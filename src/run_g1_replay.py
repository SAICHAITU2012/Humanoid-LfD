from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "vendor/mujoco_menagerie/unitree_g1/scene_with_hands.xml"


def load_csv_rows(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if not rows:
        raise SystemExit("Action CSV is empty")
    return rows


def load_model_rows(landmarks_path: Path, model_path: Path) -> list[dict]:
    """Run the trained hand mapper on extracted landmarks to produce per-frame joint targets."""
    import json

    import torch

    from train_policy import HandMapper

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    joint_names = ckpt["joint_names"]
    in_features = ckpt["state_dict"]["net.0.weight"].shape[1]
    net = HandMapper(in_features, len(joint_names))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    frames: list[tuple[float, np.ndarray | None]] = []
    with landmarks_path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            hands = item["hands"]
            flat = np.asarray(hands[0]["landmarks"], dtype=np.float32).reshape(-1) if hands else None
            frames.append((float(item["time_s"]), flat))
    if not frames:
        raise SystemExit("Landmarks file is empty")
    known = [flat for _, flat in frames if flat is not None]
    if len(known) < 20:
        raise SystemExit(f"Only {len(known)} tracked frames; not enough to drive the policy.")
    x = np.stack(known)
    with torch.no_grad():
        y = net(torch.from_numpy((x - ckpt["input_mean"]) / ckpt["input_std"])).numpy() * ckpt["target_std"] + ckpt["target_mean"]
    rows: list[dict] = []
    last = np.zeros(len(joint_names), dtype=np.float64)
    targets = iter(y)
    for time_s, flat in frames:
        if flat is not None:
            last = next(targets)
        rows.append({"frame": len(rows), "time_s": time_s, **dict(zip(joint_names, last))})
    print(f"Policy inference: {len(known)}/{len(frames)} frames tracked; mapper {model_path.name}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay retargeted hand targets on the Menagerie Unitree G1 in MuJoCo.")
    parser.add_argument("actions", type=Path, nargs="?", help="Retargeted action CSV (omit when using --model).")
    parser.add_argument("--landmarks", type=Path, help="Extracted landmarks JSONL; with --model, the trained mapper drives the hand.")
    parser.add_argument("--model", type=Path, help="Trained hand mapper checkpoint (models/hand_mapper.pt).")
    parser.add_argument("--video", type=Path, default=ROOT / "outputs/g1_hand_replay.mp4")
    parser.add_argument("--trace", type=Path, default=ROOT / "outputs/rollout.csv")
    parser.add_argument("--steps-per-frame", type=int, help="Override source timestamps with a fixed number of MuJoCo steps per frame.")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.model:
        if not args.landmarks:
            raise SystemExit("--model requires --landmarks")
        rows = load_model_rows(args.landmarks, args.model)
    elif args.actions:
        rows = load_csv_rows(args.actions)
    else:
        raise SystemExit("Provide an action CSV, or --landmarks with --model for learned-policy replay")
    sim_model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(sim_model)
    stand_id = mujoco.mj_name2id(sim_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(sim_model, data, stand_id)
    joint_names = [
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    ]
    actuator_ids = [mujoco.mj_name2id(sim_model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in joint_names]
    if min(actuator_ids) < 0:
        raise SystemExit("Could not find all G1 right-hand actuators in selected MJCF")
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    renderer = None
    writer = None
    if not args.no_video:
        try:
            renderer = mujoco.Renderer(sim_model, height=480, width=640)
        except Exception as exc:
            raise SystemExit(f"Could not initialize MuJoCo renderer ({exc}). Retry with --no-video to run physics and save the trace.") from exc
        import cv2
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
        if not writer.isOpened():
            raise SystemExit(f"Could not open output video: {args.video}")
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.82]
    camera.distance = 1.8
    camera.azimuth = 140
    camera.elevation = -12
    with args.trace.open("w", newline="", encoding="utf-8") as f:
        columns = ["sim_time_s", "source_frame", *joint_names, "pelvis_x", "pelvis_y", "pelvis_z"]
        trace = csv.DictWriter(f, fieldnames=columns)
        trace.writeheader()
        for index, row in enumerate(rows):
            ctrl = np.asarray([float(row[name]) for name in joint_names])
            for aid, target in zip(actuator_ids, ctrl):
                data.ctrl[aid] = target
            if args.steps_per_frame is not None:
                frame_steps = args.steps_per_frame
            else:
                current_t = float(row["time_s"])
                next_t = float(rows[index + 1]["time_s"]) if index + 1 < len(rows) else current_t + 1.0 / 30.0
                frame_steps = max(1, round(max(0.0, next_t - current_t) / sim_model.opt.timestep))
            for _ in range(frame_steps):
                mujoco.mj_step(sim_model, data)
            trace.writerow({"sim_time_s": data.time, "source_frame": row["frame"], **dict(zip(joint_names, ctrl)), "pelvis_x": data.qpos[0], "pelvis_y": data.qpos[1], "pelvis_z": data.qpos[2]})
            if renderer:
                renderer.update_scene(data, camera=camera)
                writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    if writer:
        writer.release()
        renderer.close()
    print(f"Simulated {len(rows)} target frames; final time={data.time:.2f}s; pelvis z={data.qpos[2]:.3f}m")
    print(f"Trace: {args.trace}")
    if not args.no_video:
        print(f"Video: {args.video}")


if __name__ == "__main__":
    main()

