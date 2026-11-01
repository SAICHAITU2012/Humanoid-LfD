from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from build_cup_scene import ensure_scene


ROOT = Path(__file__).resolve().parents[1]
ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
GRASP_POSE = np.array([-0.6, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
HAND_JOINTS = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]
HAND_CLOSED = np.array([-0.55, -0.35, -1.10, 1.20, 1.35, 1.20, 1.35])
FINGER_TIPS = ["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"]
TABLE_TOP_Z = 0.730
COASTER_XY = np.array([0.40, 0.01])
COASTER_TOP_Z = 0.746
CUP_HALF_HEIGHT = 0.055
LIFT_HEIGHT = 0.12


def ids(model: mujoco.MjModel, kind: mujoco.mjtObj, names: list[str]) -> list[int]:
    result = [mujoco.mj_name2id(model, kind, name) for name in names]
    if min(result) < 0:
        missing = names[result.index(min(result))]
        raise RuntimeError(f"Missing MuJoCo element: {missing}")
    return result


def solve_arm_ik(
    model: mujoco.MjModel,
    source_data: mujoco.MjData,
    arm_joint_ids: list[int],
    wrist_body: int,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Damped-least-squares IK for the G1 arm's wrist pose, with joint-limit clipping."""
    qpos_adrs = model.jnt_qposadr[arm_joint_ids]
    dof_adrs = model.jnt_dofadr[arm_joint_ids]
    data = mujoco.MjData(model)
    data.qpos[:] = source_data.qpos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(100):
        mujoco.mj_forward(model, data)
        current_position = data.xpos[wrist_body].copy()
        current_rotation = data.xmat[wrist_body].reshape(3, 3).copy()
        position_error = target_position - current_position
        rotation_error = Rotation.from_matrix(target_rotation @ current_rotation.T).as_rotvec()
        error = np.concatenate((position_error, rotation_error))
        if np.linalg.norm(position_error) < 0.002 and np.linalg.norm(rotation_error) < 0.02:
            break

        mujoco.mj_jacBody(model, data, jacp, jacr, wrist_body)
        jacobian = np.vstack((jacp[:, dof_adrs], jacr[:, dof_adrs]))
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.03 * np.eye(6), error)
        data.qpos[qpos_adrs] += np.clip(step, -0.08, 0.08)
        for joint_id, qpos_adr in zip(arm_joint_ids, qpos_adrs):
            if model.jnt_limited[joint_id]:
                low, high = model.jnt_range[joint_id]
                data.qpos[qpos_adr] = np.clip(data.qpos[qpos_adr], low, high)

    mujoco.mj_forward(model, data)
    error_m = float(np.linalg.norm(target_position - data.xpos[wrist_body]))
    return data.qpos[qpos_adrs].copy(), error_m


def wrist_goal_from_cup_goal(
    wrist_position: np.ndarray,
    wrist_rotation: np.ndarray,
    cup_position: np.ndarray,
    cup_rotation: np.ndarray,
    target_cup_position: np.ndarray,
    target_cup_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cup_offset_in_wrist = wrist_rotation.T @ (cup_position - wrist_position)
    cup_rotation_in_wrist = wrist_rotation.T @ cup_rotation
    target_wrist_rotation = target_cup_rotation @ cup_rotation_in_wrist.T
    target_wrist_position = target_cup_position - target_wrist_rotation @ cup_offset_in_wrist
    return target_wrist_position, target_wrist_rotation


def hand_contacts(model: mujoco.MjModel, data: mujoco.MjData, cup_body: int) -> list[str]:
    names = []
    for i in range(data.ncon):
        contact = data.contact[i]
        bodies = [model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]]
        if cup_body not in bodies:
            continue
        other = bodies[1] if bodies[0] == cup_body else bodies[0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
        if "hand" in name or "wrist" in name:
            names.append(name)
    return names


def cup_coaster_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    cup_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_cup_geom")
    coaster_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_coaster")
    for i in range(data.ncon):
        pair = {data.contact[i].geom1, data.contact[i].geom2}
        if pair == {cup_geom, coaster_geom}:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a scripted G1 cup pick-and-place setup simulation in MuJoCo."
    )
    parser.add_argument("--video", type=Path, default=ROOT / "outputs/g1_cup_pick_place.mp4")
    parser.add_argument("--trace", type=Path, default=ROOT / "outputs/cup_pick_place_trace.csv")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(ensure_scene()))
    data = mujoco.MjData(model)
    stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand < 0:
        raise RuntimeError("The G1 model has no stand keyframe")
    mujoco.mj_resetDataKeyframe(model, data, stand)

    arm_joints = ids(model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS)
    arm_actuators = ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_JOINTS)
    arm_qpos_adrs = model.jnt_qposadr[arm_joints]
    hand_actuators = ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, HAND_JOINTS)
    tip_bodies = ids(model, mujoco.mjtObj.mjOBJ_BODY, FINGER_TIPS)
    wrist_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    cup_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cup")
    cup_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_cup_free")
    if min(wrist_body, cup_body, cup_joint) < 0:
        raise RuntimeError("Cup scene is missing the expected G1 wrist or task object")
    cup_qpos_adr = model.jnt_qposadr[cup_joint]

    # Place the cup on the table in a calibrated grasp pose. The demo starts with the hand already at the cup.
    data.qpos[arm_qpos_adrs] = GRASP_POSE
    mujoco.mj_forward(model, data)
    cup_start = data.xpos[tip_bodies].mean(axis=0) + np.array([-0.05, -0.02, 0.0])
    cup_start[2] = TABLE_TOP_Z + CUP_HALF_HEIGHT
    data.qpos[cup_qpos_adr:cup_qpos_adr + 3] = cup_start
    data.qpos[cup_qpos_adr + 3:cup_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
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
    full_camera.lookat[:] = [0.35, -0.02, 0.82]
    full_camera.distance = 2.35
    full_camera.azimuth = 140
    full_camera.elevation = -12
    hand_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(hand_camera)
    hand_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    hand_camera.lookat[:] = [0.42, 0.0, 0.82]
    hand_camera.distance = 0.65
    hand_camera.azimuth = 140
    hand_camera.elevation = -12

    # Each stage uses a wrist-pose IK target. The cup itself remains a free body and must stay in contact.
    stage_names = ["grasp", "lift", "carry", "place", "release", "settle"]
    stage_ends = [0.50, 1.30, 2.10, 2.70, 3.10, 4.50]
    stage_durations = {"grasp": 0.0, "lift": 0.65, "carry": 0.65, "place": 0.50, "release": 0.0, "settle": 0.0}
    stage_index = -1
    stage_start_time = 0.0
    stage_start_q = data.qpos[arm_qpos_adrs].copy()
    stage_target_q = stage_start_q.copy()
    stage_ik_error = 0.0
    contact_frames = 0
    transfer_height = 0.0
    held_heights = []
    coaster_contact_frames = 0
    trace_columns = ["time_s", "phase", "cup_x", "cup_y", "cup_z", "cup_tilt_deg", "hand_contact_count", "coaster_contact"]
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    timestep = model.opt.timestep
    frame_steps = max(1, round((1.0 / 30.0) / timestep))
    frame_count = 150

    with args.trace.open("w", newline="", encoding="utf-8") as stream:
        trace = csv.DictWriter(stream, fieldnames=trace_columns)
        trace.writeheader()
        for _frame in range(frame_count):
            new_stage = next((i for i, end in enumerate(stage_ends) if data.time < end), len(stage_names) - 1)
            if new_stage != stage_index:
                stage_index = new_stage
                stage = stage_names[stage_index]
                stage_start_time = data.time
                stage_start_q = data.qpos[arm_qpos_adrs].copy()
                stage_ik_error = 0.0

                if stage == "grasp":
                    stage_target_q = GRASP_POSE.copy()
                elif stage in {"lift", "carry", "place"}:
                    wrist_position = data.xpos[wrist_body].copy()
                    wrist_rotation = data.xmat[wrist_body].reshape(3, 3).copy()
                    cup_position = data.xpos[cup_body].copy()
                    cup_rotation = data.xmat[cup_body].reshape(3, 3).copy()
                    if stage == "lift":
                        target_cup_position = cup_position + np.array([0.0, 0.0, LIFT_HEIGHT])
                        target_cup_rotation = np.eye(3)
                    elif stage == "carry":
                        transfer_height = max(cup_position[2], TABLE_TOP_Z + CUP_HALF_HEIGHT + 0.07)
                        target_cup_position = np.array([COASTER_XY[0], COASTER_XY[1], transfer_height])
                        target_cup_rotation = np.eye(3)
                    else:
                        target_cup_position = np.array([COASTER_XY[0], COASTER_XY[1], COASTER_TOP_Z + CUP_HALF_HEIGHT])
                        target_cup_rotation = np.eye(3)
                    target_wrist_position, target_wrist_rotation = wrist_goal_from_cup_goal(
                        wrist_position, wrist_rotation, cup_position, cup_rotation,
                        target_cup_position, target_cup_rotation,
                    )
                    stage_target_q, stage_ik_error = solve_arm_ik(
                        model, data, arm_joints, wrist_body,
                        target_wrist_position, target_wrist_rotation,
                    )
                else:
                    stage_target_q = stage_start_q.copy()

            stage = stage_names[stage_index]
            duration = stage_durations[stage]
            blend = 1.0 if duration <= 0 else min(1.0, max(0.0, (data.time - stage_start_time) / duration))
            arm_target = stage_start_q + blend * (stage_target_q - stage_start_q)
            hand_amount = 0.0 if stage in {"release", "settle"} else 1.0
            for actuator_id, value in zip(arm_actuators, arm_target):
                data.ctrl[actuator_id] = value
            for actuator_id, value in zip(hand_actuators, HAND_CLOSED * hand_amount):
                data.ctrl[actuator_id] = value
            for _ in range(frame_steps):
                mujoco.mj_step(model, data)

            cup_position = data.xpos[cup_body].copy()
            cup_rotation = data.xmat[cup_body].reshape(3, 3)
            cup_axis = cup_rotation[:, 2]
            tilt_deg = float(np.degrees(np.arccos(np.clip(cup_axis[2], -1.0, 1.0))))
            contacts = hand_contacts(model, data, cup_body)
            coaster_touching = cup_coaster_contact(model, data)
            if 0.55 <= data.time <= 2.70:
                held_heights.append(float(cup_position[2]))
                contact_frames += int(len(contacts) >= 2 and cup_position[2] > TABLE_TOP_Z + 0.05)
            if data.time >= 3.35:
                coaster_contact_frames += int(coaster_touching)
            trace.writerow({
                "time_s": f"{data.time:.4f}", "phase": stage,
                "cup_x": f"{cup_position[0]:.5f}", "cup_y": f"{cup_position[1]:.5f}", "cup_z": f"{cup_position[2]:.5f}",
                "cup_tilt_deg": f"{tilt_deg:.3f}", "hand_contact_count": len(contacts),
                "coaster_contact": int(coaster_touching),
            })

            if renderer is not None:
                import cv2

                renderer.update_scene(data, camera=full_camera)
                full = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                renderer.update_scene(data, camera=hand_camera)
                close = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                cv2.putText(full, "Unitree G1 / cup and coaster", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2)
                cv2.putText(close, f"Scripted {stage}", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
                writer.write(cv2.hconcat((full, close)))

    if writer is not None:
        writer.release()
        renderer.close()
    final_position = data.xpos[cup_body].copy()
    coaster_error = float(np.linalg.norm(final_position[:2] - COASTER_XY))
    final_tilt = float(np.degrees(np.arccos(np.clip(data.xmat[cup_body].reshape(3, 3)[2, 2], -1.0, 1.0))))
    print(f"Simulation: {frame_count} frames, {data.time:.2f}s; finite state: {bool(np.isfinite(data.qpos).all())}")
    print(f"Hand-contact hold: {contact_frames}/{len(held_heights)} sampled lift/carry frames met the contact and height check")
    print(f"Final cup: x={final_position[0]:.3f}m, y={final_position[1]:.3f}m, z={final_position[2]:.3f}m; coaster center error={coaster_error:.3f}m; tilt={final_tilt:.1f}deg")
    print(f"Cup/coaster contact frames after release: {coaster_contact_frames}")
    print(f"Trace: {args.trace}")
    if not args.no_video:
        print(f"Video: {args.video}")
    print("This scripted setup starts at a calibrated grasp pose; it does not include an arm approach or use personal footage.")


if __name__ == "__main__":
    main()
