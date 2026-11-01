from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from build_cup_scene import ensure_scene


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/public/g1_cup_mit"
CKPT = ROOT / "models/public_baseline.pt"
SOURCE = "lewislf/G1_Dex3_pick_white_cup_right8_1squeeze (MIT)"
ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
HAND_JOINTS = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]
TIPS = ["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"]
CUP_HALF_HEIGHT = 0.055
HORIZON = 16   # actions predicted at once
EXECUTE = 8    # actions executed before re-planning
CONTEXT = 4    # past states fed as input


def load_all() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(glob.glob(str(DATA_DIR / "data/chunk-000/*.parquet")))
    cols = ["action", "observation.state", "episode_index"]
    tables = [pq.read_table(f, columns=cols) for f in files]
    x = np.asarray(np.concatenate([t.column("observation.state").to_pylist() for t in tables]), np.float32)
    y = np.asarray(np.concatenate([t.column("action").to_pylist() for t in tables]), np.float32)
    e = np.asarray(np.concatenate([t.column("episode_index").to_pylist() for t in tables]), int)
    return x, y, e


def make_windows(states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Input: last CONTEXT states; target: the next HORIZON actions."""
    n = len(states) - HORIZON + 1
    xs = np.stack([np.concatenate([states[max(0, t - CONTEXT + 1 + j)] for j in range(CONTEXT)]) for t in range(n)])
    ys = np.stack([actions[t:t + HORIZON].reshape(-1) for t in range(n)])
    return xs.astype(np.float32), ys.astype(np.float32)


class ChunkPolicy(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, out_features))

    def forward(self, x):
        return self.net(x)


def build_scene(radius: float, center: np.ndarray) -> Path:
    text = ensure_scene().read_text(encoding="utf-8")
    text = text.replace('<geom name="target_cup_geom" type="cylinder" size="0.03 0.055"',
                        f'<geom name="target_cup_geom" type="cylinder" size="{radius:.4f} 0.055"', 1)
    top = max(center[2] - CUP_HALF_HEIGHT - 0.002, 0.02)
    pedestal = (f'    <geom name="replay_pedestal" type="cylinder" pos="{center[0]:.3f} {center[1]:.3f} {top / 2:.3f}" '
                f'size="0.08 {top / 2:.3f}" rgba="0.5 0.5 0.55 1" friction="1.0 0.01 0.001" condim="4"/>\n  ')
    text = text.replace("</worldbody>", pedestal + "</worldbody>", 1)
    out = ensure_scene().with_name("scene_with_cup_replay.xml")
    out.write_text(text, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Action-chunking BC on public G1 Dex3 episodes + closed-loop rollout.")
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--video", type=Path, default=ROOT / "outputs/chunked_baseline_rollout.mp4")
    parser.add_argument("--execute", type=int, default=1)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    x, y, episodes = load_all()
    order = np.sort(np.unique(episodes))
    split_ep = order[int(0.8 * len(order))]
    train_eps = order[order < split_ep]

    windows_x, windows_y, windows_ep = [], [], []
    for ep in train_eps:
        m = episodes == ep
        wx, wy = make_windows(x[m], y[m])
        windows_x.append(wx)
        windows_y.append(wy)
        windows_ep.append(np.full(len(wx), ep))
    wx = np.concatenate(windows_x)
    wy = np.concatenate(windows_y)
    print(f"Training windows: {len(wx)} (context {CONTEXT} states -> next {HORIZON} actions)")

    xm, xs = wx.mean(0), wx.std(0).clip(1e-4)
    ym, ys = wy.mean(0), wy.std(0).clip(1e-4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    policy = ChunkPolicy(wx.shape[1], wy.shape[1]).to(device)
    optim = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.from_numpy((wx - xm) / xs).to(device)
    yt = torch.from_numpy((wy - ym) / ys).to(device)
    rng = np.random.default_rng(0)
    batch = 4096
    for epoch in range(args.epochs):
        policy.train()
        perm = rng.permutation(len(xt))
        for start in range(0, len(xt), batch):
            idx = perm[start:start + batch]
            optim.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(policy(xt[idx]), yt[idx])
            loss.backward()
            optim.step()
        if (epoch + 1) % 100 == 0:
            policy.eval()
            with torch.no_grad():
                idx = perm[:8192]
                print(f"epoch {epoch + 1}: train batch RMSE "
                      f"{float(torch.sqrt(nn.functional.mse_loss(policy(xt[idx]), yt[idx]))):.4f}")

    torch.save({"state_dict": policy.state_dict(), "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys,
                "horizon": HORIZON, "execute": EXECUTE, "context": CONTEXT, "episode_split": int(split_ep),
                "source": SOURCE}, ROOT / "models/public_baseline_chunked.pt")

    # ---- Closed-loop rollout on a held-out episode, cup at the demo's real grasp point ----
    ep = args.episode if args.episode is not None else int(order[order >= split_ep][0])
    m = episodes == ep
    true_states, true_actions = x[m], y[m]

    scene0 = ensure_scene()
    model0 = mujoco.MjModel.from_xml_path(str(scene0))
    data0 = mujoco.MjData(model0)
    stand = mujoco.mj_name2id(model0, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model0, data0, stand)
    arm_qadr0 = model0.jnt_qposadr[[mujoco.mj_name2id(model0, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]]
    hand_qadr0 = model0.jnt_qposadr[[mujoco.mj_name2id(model0, mujoco.mjtObj.mjOBJ_JOINT, n) for n in HAND_JOINTS]]
    tips0 = [mujoco.mj_name2id(model0, mujoco.mjtObj.mjOBJ_BODY, n) for n in TIPS]
    closed_frame = int(np.argmax(true_actions[:, 7]))
    data0.qpos[arm_qadr0] = true_states[closed_frame, :7]
    data0.qpos[hand_qadr0] = true_states[closed_frame, 7:14]
    data0.qvel[:] = 0
    mujoco.mj_forward(model0, data0)
    p_thumb = data0.xpos[tips0[0]].copy()
    p_fingers = (data0.xpos[tips0[1]] + data0.xpos[tips0[2]]) / 2.0
    span = float(np.linalg.norm(p_thumb - p_fingers))
    radius = float(np.clip(span / 2.0 + 0.006, 0.030, 0.050))
    center = (p_thumb + p_fingers) / 2.0
    center[2] = float((data0.xpos[tips0[0]][2] + data0.xpos[tips0[1]][2] + data0.xpos[tips0[2]][2]) / 3.0)

    sq_tr = y[episodes < split_ep][:, 7]
    st_tr = x[episodes < split_ep][1:, 7:14]
    design = np.vstack([np.ones_like(sq_tr[:-1]), sq_tr[:-1]]).T
    finger_b, finger_w = np.linalg.lstsq(design, st_tr, rcond=None)[0]

    model = mujoco.MjModel.from_xml_path(str(build_scene(radius, center)))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, stand)
    arm_qadr = model.jnt_qposadr[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]]
    hand_qadr = model.jnt_qposadr[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in HAND_JOINTS]]
    tips = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in TIPS]
    cup_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cup")
    limits = model.jnt_range[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]]
    data.qpos[arm_qadr] = true_states[0, :7]
    data.qpos[hand_qadr] = finger_b + finger_w * true_actions[0, 7]
    mujoco.mj_forward(model, data)

    history = []

    def predict() -> np.ndarray:
        past = history[-CONTEXT:] if len(history) >= CONTEXT else (
            [history[0]] * (CONTEXT - len(history)) + history[-CONTEXT:] if history
            else [true_states[0].tolist()] * CONTEXT)
        window = np.asarray(past, np.float32).reshape(1, -1)
        with torch.no_grad():
            out = policy(torch.from_numpy((window - xm) / xs).to(device)).cpu().numpy() * ys + ym
        return out.reshape(HORIZON, 8)

    renderer = None
    writer = None
    if not args.no_video:
        renderer = mujoco.Renderer(model, height=480, width=480)
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (480, 480))
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.3, -0.05, 0.85]
    cam.distance = 1.8
    cam.azimuth = 140
    cam.elevation = -12

    errors = []
    cup_min_dist = None
    steps_per_frame = max(1, round((1.0 / 30.0) / model.opt.timestep))
    half_diverge = [0.0, 0.0]
    frame = 0
    while frame < len(true_states):
        chunk = predict()
        for k in range(args.execute):
            if frame >= len(true_states):
                break
            action = chunk[k].copy()
            action[:7] = np.clip(action[:7], limits[:, 0], limits[:, 1])
            action[7] = float(np.clip(action[7], 0.0, 1.0))
            data.qpos[arm_qadr] = action[:7]
            data.qpos[hand_qadr] = finger_b + finger_w * action[7]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            err = float(np.sqrt(((data.qpos[arm_qadr] - true_states[frame, :7]) ** 2).mean()))
            errors.append(err)
            half_diverge[0 if frame < len(true_states) // 2 else 1] += err
            history.append((data.qpos[arm_qadr].tolist() + data.qpos[hand_qadr].tolist()))
            if writer is not None:
                renderer.update_scene(data, camera=cam)
                img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                cv2.putText(img, f"chunked BC - held-out ep {ep} - frame {frame}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(img, f"arm RMSE so far {float(np.mean(errors)):.3f} rad", (12, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 2)
                writer.write(img)
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            frame += 1

    if writer is not None:
        writer.release()
        renderer.close()
    cup_z = float(data.xpos[cup_body, 2])
    n = len(errors)
    print(f"\nHeld-out episode {ep}: closed-loop arm RMSE {float(np.mean(errors)):.4f} rad "
          f"(single-state BC: 0.5048) | first half {half_diverge[0] / max(1, n // 2):.3f}, "
          f"second half {half_diverge[1] / max(1, n - n // 2):.3f}")
    print(f"Grasp-point miss distance at demo-grasp time: "
          f"{cup_min_dist if cup_min_dist is not None else float('nan'):.3f} m")
    print(f"Cup: center z start {center[2]:.3f} -> final {cup_z:.3f}")
    print(f"Video: {args.video}")


if __name__ == "__main__":
    main()
