from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

from build_cup_scene import ensure_scene


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/public/g1_cup_mit"
ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
HAND_JOINTS = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]
TIP_THUMB = "right_hand_thumb_2_link"
TIP_INDEX = "right_hand_index_1_link"
TIP_MIDDLE = "right_hand_middle_1_link"
CUP_HALF_HEIGHT = 0.055
SOURCE = "lewislf/G1_Dex3_pick_white_cup_right8_1squeeze (MIT)"


def load_episodes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(glob.glob(str(DATA_DIR / "data/chunk-000/*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files under {DATA_DIR}")
    cols = ["action", "observation.state", "episode_index", "timestamp"]
    tables = [pq.read_table(f, columns=cols) for f in files]
    states = np.concatenate([t.column("observation.state").to_pylist() for t in tables])
    actions = np.concatenate([t.column("action").to_pylist() for t in tables])
    episodes = np.concatenate([t.column("episode_index").to_pylist() for t in tables])
    return np.asarray(states, dtype=float), np.asarray(actions, dtype=float), np.asarray(episodes, dtype=int)


def pick_episode(model: mujoco.MjModel, data: mujoco.MjData, states: np.ndarray, actions: np.ndarray,
                 episodes: np.ndarray, arm_qadr: np.ndarray, hand_qadr: np.ndarray, tips: list[int],
                 requested: int | None) -> int:
    """Prefer episodes with a firm, sustained squeeze AND demonstrated lift (FK-measured)."""
    if requested is not None:
        return requested
    stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    best, best_score = 0, -1.0
    for ep in np.unique(episodes):
        m = episodes == ep
        sq = actions[m, 7]
        if sq.max() < 0.8 or int(np.sum(sq > 0.7)) < 15:
            continue
        idx = np.where(m)[0]
        closed = int(idx[np.argmax(sq)])
        late = min(int(idx[-1]), closed + 50)
        z = []
        for frame in (closed, late):
            data.qpos[arm_qadr] = states[frame, :7]
            data.qpos[hand_qadr] = states[frame, 7:14]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            z.append(float(data.xpos[tips].mean(axis=0)[2]))
        lift = z[1] - z[0]
        score = lift + 0.05 * min(int(np.sum(sq > 0.7)), 40)
        if lift > 0.04 and score > best_score:
            best, best_score = int(ep), score
    return best


def build_scene(radius: float, pedestal_xy: np.ndarray, pedestal_top: float | None) -> Path:
    """Cup scene with a resized cup and an optional pedestal under it."""
    text = ensure_scene().read_text(encoding="utf-8")
    text = text.replace('<geom name="target_cup_geom" type="cylinder" size="0.03 0.055"',
                        f'<geom name="target_cup_geom" type="cylinder" size="{radius:.4f} 0.055"', 1)
    if pedestal_top is not None:
        pedestal = (f'    <geom name="replay_pedestal" type="cylinder" '
                    f'pos="{pedestal_xy[0]:.3f} {pedestal_xy[1]:.3f} {pedestal_top / 2:.3f}" '
                    f'size="0.08 {pedestal_top / 2:.3f}" rgba="0.5 0.5 0.55 1" '
                    f'friction="1.0 0.01 0.001" condim="4"/>\n  ')
        text = text.replace("</worldbody>", pedestal + "</worldbody>", 1)
    out = ensure_scene().with_name("scene_with_cup_replay.xml")
    out.write_text(text, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Kinematically replay a real G1 Dex3 teleop episode ({SOURCE}) in the local cup scene."
    )
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--video", type=Path, default=ROOT / "outputs/replay_public_episode.mp4")
    parser.add_argument("--trace", type=Path, default=ROOT / "outputs/replay_public_trace.csv")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    states, actions, episodes = load_episodes()
    ep_mask_count = len(np.unique(episodes))
    model = mujoco.MjModel.from_xml_path(str(ensure_scene()))
    data = mujoco.MjData(model)
    stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand < 0:
        raise RuntimeError("The G1 model has no 'stand' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, stand)
    arm_qadr = model.jnt_qposadr[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]]
    hand_qadr = model.jnt_qposadr[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in HAND_JOINTS]]
    tip_thumb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TIP_THUMB)
    tip_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TIP_INDEX)
    tip_middle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TIP_MIDDLE)
    tips = [tip_thumb, tip_index, tip_middle]
    cup_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cup")
    cup_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_cup_free")]

    episode = pick_episode(model, data, states, actions, episodes, arm_qadr, hand_qadr, tips, args.episode)
    ep_states = states[episodes == episode]
    ep_actions = actions[episodes == episode]
    print(f"Episode {episode} of {ep_mask_count}: {len(ep_states)} frames ({len(ep_states) / 30:.1f}s)")

    def set_state(state: np.ndarray) -> None:
        data.qpos[arm_qadr] = state[:7]
        data.qpos[hand_qadr] = state[7:14]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

    close_idx = int(np.argmax(ep_actions[:, 7] > 0.5))
    closed_frame = int(np.argmax(ep_actions[:, 7]))
    set_state(ep_states[closed_frame])
    p_thumb = data.xpos[tip_thumb].copy()
    p_index = data.xpos[tip_index].copy()
    p_middle = data.xpos[tip_middle].copy()
    p_fingers = (p_index + p_middle) / 2.0

    # The real grasp is a pinch: thumb on one side of the mug, index+middle on the
    # other. The cup axis therefore sits on the thumb<->fingers line, and the cup
    # radius is half that span plus a pad-thickness margin.
    span = float(np.linalg.norm(p_thumb - p_fingers))
    base_radius = float(np.clip(span / 2.0 + 0.006, 0.030, 0.050))
    base_center = (p_thumb + p_fingers) / 2.0
    base_center[2] = float((p_thumb[2] + p_index[2] + p_middle[2]) / 3.0)
    print(f"Closed-frame pinch span {span:.3f} m -> base radius {base_radius:.3f}, center {np.round(base_center, 3).tolist()}")

    def score(radius: float, center: np.ndarray, overshoot: float) -> tuple[int, float]:
        scene = build_scene(radius, center[:2], max(center[2] - CUP_HALF_HEIGHT - 0.002, 0.02))
        trial_model = mujoco.MjModel.from_xml_path(str(scene))
        trial_data = mujoco.MjData(trial_model)
        mujoco.mj_resetDataKeyframe(trial_model, trial_data, stand)
        t_arm = trial_model.jnt_qposadr[[mujoco.mj_name2id(trial_model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]]
        t_hand = trial_model.jnt_qposadr[[mujoco.mj_name2id(trial_model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in HAND_JOINTS]]
        t_cup = mujoco.mj_name2id(trial_model, mujoco.mjtObj.mjOBJ_BODY, "target_cup")
        t_qadr = trial_model.jnt_qposadr[mujoco.mj_name2id(trial_model, mujoco.mjtObj.mjOBJ_JOINT, "target_cup_free")]
        trial_data.qpos[t_qadr:t_qadr + 3] = center
        trial_data.qpos[t_qadr + 3:t_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(trial_model, trial_data)
        start_z = float(center[2])
        good = 0
        for frame in range(max(0, close_idx - 5), min(len(ep_states), close_idx + 61)):
            fingers = ep_states[frame, 7:14]
            if frame >= close_idx:
                fingers = fingers * (1.0 + overshoot)  # servo overshoot re-creates the real squeeze force
            trial_data.qpos[t_arm] = ep_states[frame, :7]
            trial_data.qpos[t_hand] = fingers
            trial_data.qvel[:] = 0
            for _ in range(max(1, round((1.0 / 30.0) / trial_model.opt.timestep))):
                mujoco.mj_step(trial_model, trial_data)
            contacts = sum(
                1 for i in range(trial_data.ncon)
                if t_cup in (trial_model.geom_bodyid[trial_data.contact[i].geom1],
                             trial_model.geom_bodyid[trial_data.contact[i].geom2]))
            good += int(contacts >= 2 and float(trial_data.xpos[t_cup, 2]) > start_z + 0.015)
        return good, float(trial_data.xpos[t_cup, 2]) - start_z

    best = (None, None, -1, -1.0, 0.0)
    for overshoot in (0.2, 0.6, 1.2, 2.0):
        for radius_margin in (0.0, 0.004):
            for dz in (-0.015, -0.007, 0.0, 0.007):
                radius = base_radius + radius_margin
                center = base_center + np.array([0.0, 0.0, dz])
                good, rise = score(radius, center, overshoot)
                if (good, rise) > (best[2], best[3]):
                    best = (radius, center.copy(), good, rise, overshoot)
    radius, cup_center, peak_contacts, trial_rise, overshoot = best
    if radius is None or peak_contacts <= 0:
        raise SystemExit(f"No placement held the cup in episode {episode}; try --episode.")
    print(f"Placement search: radius {radius:.3f}, center {np.round(cup_center, 3).tolist()}, "
          f"held-lift frames {peak_contacts}, rise {trial_rise:+.3f} m, servo overshoot {overshoot:.1f}")

    start_z = float(cup_center[2])
    scene = build_scene(radius, cup_center[:2], start_z - CUP_HALF_HEIGHT - 0.002)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, stand)
    cup_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_cup_free")]
    cup_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cup")
    arm_qadr = model.jnt_qposadr[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]]
    hand_qadr = model.jnt_qposadr[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in HAND_JOINTS]]
    data.qpos[cup_qadr:cup_qadr + 3] = cup_center
    data.qpos[cup_qadr + 3:cup_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    renderer = None
    writer = None
    if not args.no_video:
        import cv2
        renderer = mujoco.Renderer(model, height=480, width=480)
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 480))
    full_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(full_camera)
    full_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    full_camera.lookat[:] = [0.35, -0.02, 0.9]
    full_camera.distance = 2.35
    full_camera.azimuth = 140
    full_camera.elevation = -12
    hand_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(hand_camera)
    hand_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    hand_camera.lookat[:] = cup_center
    hand_camera.distance = 0.5
    hand_camera.azimuth = 140
    hand_camera.elevation = -12

    steps_per_frame = max(1, round((1.0 / 30.0) / model.opt.timestep))
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    contact_frames = hold_frames = 0
    with args.trace.open("w", newline="", encoding="utf-8") as stream:
        trace = csv.writer(stream)
        trace.writerow(["time_s", "frame", "cup_x", "cup_y", "cup_z", "hand_contact_count"])
        for frame in range(len(ep_states)):
            fingers = ep_states[frame, 7:14]
            if frame >= close_idx:
                fingers = fingers * (1.0 + overshoot)
            data.qpos[arm_qadr] = ep_states[frame, :7]
            data.qpos[hand_qadr] = fingers
            data.qvel[:] = 0
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            cup_z = float(data.xpos[cup_body, 2])
            contacts = sum(
                1 for i in range(data.ncon)
                if cup_body in (model.geom_bodyid[data.contact[i].geom1], model.geom_bodyid[data.contact[i].geom2]))
            if close_idx - 5 <= frame <= close_idx + 60:
                hold_frames += 1
                contact_frames += int(contacts >= 2 and cup_z > start_z + 0.015)
            trace.writerow([f"{data.time:.4f}", frame] + [f"{v:.5f}" for v in data.xpos[cup_body]] + [contacts])
            if writer is not None:
                import cv2
                renderer.update_scene(data, camera=full_camera)
                full = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                renderer.update_scene(data, camera=hand_camera)
                close = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                writer.write(cv2.hconcat((full, close)))

    if writer is not None:
        writer.release()
        renderer.close()
    cup_z = float(data.xpos[cup_body, 2])
    print(f"Simulated {len(ep_states)} frames; finite: {bool(np.isfinite(data.qpos).all())}")
    print(f"Grasp hold: {contact_frames}/{hold_frames} sampled frames with >=2 contacts and cup lifted >1.5cm")
    print(f"Cup height: start {start_z:.3f} m -> final {cup_z:.3f} m (rise {cup_z - start_z:+.3f} m)")
    print(f"Trace: {args.trace}")
    if not args.no_video:
        print(f"Video: {args.video}")
    print(f"Source: real teleop demonstrations, {SOURCE}; kinematic replay in the local Menagerie G1 scene.")


if __name__ == "__main__":
    main()
