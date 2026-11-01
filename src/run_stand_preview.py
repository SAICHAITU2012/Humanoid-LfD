from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "vendor/mujoco_menagerie/unitree_g1/scene_with_hands.xml"
VIDEO_PATH = ROOT / "outputs/g1_stand_hand_preview.mp4"
HAND = {
    "thumb0": "right_hand_thumb_0_joint",
    "thumb1": "right_hand_thumb_1_joint",
    "thumb2": "right_hand_thumb_2_joint",
    "index0": "right_hand_index_0_joint",
    "index1": "right_hand_index_1_joint",
    "middle0": "right_hand_middle_0_joint",
    "middle1": "right_hand_middle_1_joint",
}


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, stand)
    ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint) for name, joint in HAND.items()}
    if min(ids.values()) < 0:
        raise RuntimeError("Missing G1 right-hand actuator")

    renderer = mujoco.Renderer(model, height=480, width=480)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.82]
    camera.distance = 2.5
    camera.azimuth = 140
    camera.elevation = -12
    hand_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(hand_camera)
    hand_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    hand_camera.lookat[:] = [0.0, -0.235, 0.60]
    hand_camera.distance = 0.32
    hand_camera.azimuth = 0
    hand_camera.elevation = -10
    VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(VIDEO_PATH), cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 480))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {VIDEO_PATH}")

    # Setup-only actuator preview: it is explicitly not a demonstration or training data.
    nframes = 180
    total_steps = max(1, round((1.0 / 30.0) / model.opt.timestep))
    for frame in range(nframes):
        t = frame / (nframes - 1)
        phase = min(1.0, max(0.0, (t - 0.30) / 0.22))
        if t > 0.68:
            phase = max(0.0, 1.0 - (t - 0.68) / 0.20)
        targets = {
            "thumb0": -0.55 * phase,
            "thumb1": -0.35 * phase,
            "thumb2": -1.10 * phase,
            "index0": 1.20 * phase,
            "index1": 1.35 * phase,
            "middle0": 1.20 * phase,
            "middle1": 1.35 * phase,
        }
        for name, aid in ids.items():
            data.ctrl[aid] = targets[name]
        for _ in range(total_steps):
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera)
        full_frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        renderer.update_scene(data, camera=hand_camera)
        hand_frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(full_frame, "G1 full model", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(hand_frame, "Right-hand actuator preview", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        writer.write(cv2.hconcat((full_frame, hand_frame)))

    writer.release()
    renderer.close()
    print(f"Saved G1 MuJoCo hand-actuator setup preview: {VIDEO_PATH}")
    print(f"Simulation time: {data.time:.2f}s; finite state: {bool(np.isfinite(data.qpos).all())}")
    print("This is a simulator setup preview with a programmed open-close sequence, not user data or a trained policy.")


if __name__ == "__main__":
    main()
