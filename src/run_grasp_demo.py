from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from build_task_scene import ensure_scene


ROOT = Path(__file__).resolve().parents[1]
ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
ARM_POSE = np.array([-1.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
HAND_JOINTS = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]
HAND_CLOSED = 0.75 * np.array([-0.55, -0.35, -1.10, 1.20, 1.35, 1.20, 1.35])
FINGER_TIPS = ["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"]


def named_ids(model: mujoco.MjModel, object_type: mujoco.mjtObj, names: list[str]) -> list[int]:
    ids = [mujoco.mj_name2id(model, object_type, name) for name in names]
    if min(ids) < 0:
        raise RuntimeError(f"Missing expected MuJoCo element: {names[ids.index(min(ids))]}")
    return ids


def ball_hand_contacts(model: mujoco.MjModel, data: mujoco.MjData, ball_body: int) -> list[str]:
    touching = []
    for i in range(data.ncon):
        contact = data.contact[i]
        bodies = [model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]]
        if ball_body not in bodies:
            continue
        other = bodies[1] if bodies[0] == ball_body else bodies[0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
        if "hand" in name or "wrist" in name:
            touching.append(name)
    return touching


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a scripted, prepositioned G1 hand grasp/hold/release in MuJoCo; this is not a learned policy."
    )
    parser.add_argument("--video", type=Path, default=ROOT / "outputs/g1_ball_grasp_demo.mp4")
    parser.add_argument("--trace", type=Path, default=ROOT / "outputs/grasp_trace.csv")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(ensure_scene()))
    data = mujoco.MjData(model)
    stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand < 0:
        raise RuntimeError("The G1 model has no stand keyframe")
    mujoco.mj_resetDataKeyframe(model, data, stand)

    arm_actuators = named_ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_JOINTS)
    arm_joints = named_ids(model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS)
    hand_actuators = named_ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, HAND_JOINTS)
    tip_bodies = named_ids(model, mujoco.mjtObj.mjOBJ_BODY, FINGER_TIPS)
    ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_ball")
    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_ball_free")
    if ball_body < 0 or ball_joint < 0:
        raise RuntimeError("Task scene does not contain the target ball and free joint")

    # Put the robot in a fixed, forward-reaching pose and initialize the sphere between its open fingers.
    for qpos_adr, value, actuator_id in zip(model.jnt_qposadr[arm_joints], ARM_POSE, arm_actuators):
        data.qpos[qpos_adr] = value
        data.ctrl[actuator_id] = value
    mujoco.mj_forward(model, data)
    ball_qpos_adr = model.jnt_qposadr[ball_joint]
    ball_start = data.xpos[tip_bodies].mean(axis=0) + np.array([-0.02, 0.04, 0.0])
    data.qpos[ball_qpos_adr:ball_qpos_adr + 3] = ball_start
    data.qpos[ball_qpos_adr + 3:ball_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    renderer = None
    writer = None
    if not args.no_video:
        import cv2

        args.video.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=480, width=480)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 480))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create video: {args.video}")

    full_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(full_camera)
    full_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    full_camera.lookat[:] = [0.0, 0.0, 0.90]
    full_camera.distance = 2.5
    full_camera.azimuth = 140
    full_camera.elevation = -12
    hand_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(hand_camera)
    hand_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    hand_camera.lookat[:] = ball_start
    hand_camera.distance = 0.42
    hand_camera.azimuth = 140
    hand_camera.elevation = -12

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    dt = model.opt.timestep
    frame_steps = max(1, round((1.0 / 30.0) / dt))
    frame_count = 180
    held_heights = []
    contact_frames = 0
    columns = ["time_s", "phase", "ball_x", "ball_y", "ball_z", "hand_contact_count", "contact_bodies"]
    with args.trace.open("w", newline="", encoding="utf-8") as stream:
        trace = csv.DictWriter(stream, fieldnames=columns)
        trace.writeheader()
        for frame in range(frame_count):
            # Close immediately, hold, then smoothly open and let the free ball fall.
            phase = "grasp_and_hold" if data.time < 2.30 else "release"
            amount = 1.0 if phase == "grasp_and_hold" else max(0.0, 1.0 - (data.time - 2.30) / 0.25)
            for actuator_id, target in zip(hand_actuators, HAND_CLOSED * amount):
                data.ctrl[actuator_id] = target
            for actuator_id, target in zip(arm_actuators, ARM_POSE):
                data.ctrl[actuator_id] = target
            for _ in range(frame_steps):
                mujoco.mj_step(model, data)

            touching = ball_hand_contacts(model, data, ball_body)
            ball_xyz = data.xpos[ball_body].copy()
            if data.time <= 2.30:
                held_heights.append(float(ball_xyz[2]))
                contact_frames += int(len(touching) >= 2 and ball_xyz[2] > 0.80)
            trace.writerow({
                "time_s": f"{data.time:.4f}",
                "phase": phase,
                "ball_x": f"{ball_xyz[0]:.5f}",
                "ball_y": f"{ball_xyz[1]:.5f}",
                "ball_z": f"{ball_xyz[2]:.5f}",
                "hand_contact_count": len(touching),
                "contact_bodies": ";".join(touching),
            })
            if renderer is not None:
                import cv2

                renderer.update_scene(data, camera=full_camera)
                full = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                renderer.update_scene(data, camera=hand_camera)
                close = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                cv2.putText(full, "G1 fixed reach pose", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
                cv2.putText(close, f"Scripted grasp / {phase.replace('_', ' ')}", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
                writer.write(cv2.hconcat((full, close)))

    if writer is not None:
        writer.release()
        renderer.close()
    print(f"Simulation: {frame_count} frames, {data.time:.2f}s; finite state: {bool(np.isfinite(data.qpos).all())}")
    print(f"Grasp hold: {contact_frames}/{len(held_heights)} sampled frames had at least two hand/wrist contacts and ball z > 0.80m")
    print(f"Lowest sampled ball height during hold: {min(held_heights):.3f}m; final ball height after release: {data.xpos[ball_body, 2]:.3f}m")
    print(f"Trace: {args.trace}")
    if not args.no_video:
        print(f"Video: {args.video}")
    print("This is a scripted prepositioned grasp/hold/release demo. It has no arm approach, lift/place motion, user recording, or learned policy.")


if __name__ == "__main__":
    main()
