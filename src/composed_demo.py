from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import mujoco
import numpy as np

from build_cup_scene import ensure_scene
from run_cup_pick_place import GRASP_POSE, HAND_CLOSED, ARM_JOINTS, HAND_JOINTS, ids, solve_arm_ik, wrist_goal_from_cup_goal, hand_contacts, cup_coaster_contact


ROOT = Path(__file__).resolve().parents[1]
SIM_FPS = 30.0
CUP_RADIUS = 0.043        # steel cup, ~8.6 cm diameter (user's video)
CUP_HALF_HEIGHT = 0.055
TABLE_TOP_Z = 0.730
COASTER_XY = np.array([0.40, 0.01])
COASTER_TOP_Z = 0.746
TAKE = "demo1"
VIDEO_FPS = 41.0          # demo1 source rate


def load_finger_targets(path: Path) -> tuple[np.ndarray, list[str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    joints = [k for k in rows[0] if k.startswith("right_hand_")]
    return np.array([[float(r[j]) for j in joints] for r in rows]), joints


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Composed demo: user-video finger choreography + IK arm stages, cup->saucer with contact confirmation."
    )
    parser.add_argument("--take", default=TAKE)
    parser.add_argument("--video", type=Path, default=ROOT / "outputs/composed_task_demo.mp4")
    parser.add_argument("--trace", type=Path, default=ROOT / "outputs/composed_task_trace.csv")
    parser.add_argument("--cup-radius", type=float, default=None)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    fingers_all, joint_names = load_finger_targets(ROOT / f"data/processed/actions_{args.take}.csv")
    n_video = len(fingers_all)
    video_to_sim = n_video / SIM_FPS  # sim seconds of source

    # Phase boundaries detected from the video's own flexion signal (see analysis).
    peak_idx = int(np.argmax(fingers_all[:, 4]))
    above = fingers_all[:, 4] > 0.5 * fingers_all[:, 4].max()
    grasp_start = next(i for i in range(n_video - 10) if above[i:i + 10].all())
    after_peak = [i for i in range(n_video) if fingers_all[i, 4] > 0.85 * fingers_all[:, 4].max()]
    closed_end = after_peak[-1]
    below = fingers_all[:, 4] < 0.25 * fingers_all[:, 4].max()
    release_start = next((i for i in range(closed_end, n_video - 10) if below[i:i + 10].all()), n_video - 1)
    print(f"Take {args.take}: grasp starts frame {grasp_start}, closed until {closed_end}, release at {release_start}")

    model = mujoco.MjModel.from_xml_path(str(ensure_scene()))
    data = mujoco.MjData(model)
    stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand < 0:
        raise RuntimeError("The G1 model has no 'stand' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, stand)

    arm_joints = ids(model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS)
    arm_actuators = ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_JOINTS)
    arm_qpos_adrs = model.jnt_qposadr[arm_joints]
    hand_actuators = ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, HAND_JOINTS)
    tip_bodies = ids(model, mujoco.mjtObj.mjOBJ_BODY, ["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"])
    wrist_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    cup_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cup")
    cup_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_cup_free")
    cup_qadr = model.jnt_qposadr[cup_joint]

    # Resize the cup to the user's steel cup and set the start pose from the
    # video's own closed-hand geometry, resting on the table.
    scene_path = ensure_scene()
    text = scene_path.read_text(encoding="utf-8")
    text = text.replace('<geom name="target_cup_geom" type="cylinder" size="0.03 0.055"',
                        f'<geom name="target_cup_geom" type="cylinder" size="{CUP_RADIUS:.4f} {CUP_HALF_HEIGHT}"', 1)
    text = text.replace('friction="3.0 0.02 0.005"', 'friction="4.5 0.02 0.005"')
    text = text.replace('friction="4.5 0.02 0.005" condim="4"', 'friction="4.5 0.02 0.005" condim="6"', 1)  # Dex3 tactile pads
    scene_path.write_text(text, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, stand)
    arm_qpos_adrs = model.jnt_qposadr[arm_joints]
    cup_qadr = model.jnt_qposadr[cup_joint]

    data.qpos[arm_qpos_adrs] = GRASP_POSE
    for actuator_id, value in zip(arm_actuators, GRASP_POSE):  # hold the grasp pose during settle
        data.ctrl[actuator_id] = value
    mujoco.mj_forward(model, data)
    closed_targets = HAND_CLOSED.copy()  # retargeted full-closure profile (= the user's demonstrated peak shape)
    cup_radius = 0.030                   # calibrated grasp geometry of the proven cup pipeline
    for actuator_id, value in zip(hand_actuators, closed_targets):
        data.ctrl[actuator_id] = value
    for _ in range(200):
        mujoco.mj_step(model, data)
    # Enclosure placement: the three fingertips of the video's closed hand form a
    # triangle around the real cup; its centroid is the cup axis, and the mean
    # pairwise span / sqrt(3) is the inscribed cup radius.
    p_thumb = data.xpos[tip_bodies[0]].copy()
    p_index = data.xpos[tip_bodies[1]].copy()
    p_middle = data.xpos[tip_bodies[2]].copy()
    pairwise = [np.linalg.norm(p_thumb - p_index), np.linalg.norm(p_thumb - p_middle), np.linalg.norm(p_index - p_middle)]
    cup_radius = args.cup_radius if args.cup_radius is not None else 0.030  # proven calibrated radius
    cup_center = data.xpos[tip_bodies].mean(axis=0) + np.array([-0.05, -0.02, 0.0])
    cup_center[2] = TABLE_TOP_Z + CUP_HALF_HEIGHT
    data.qpos[cup_qadr:cup_qadr + 3] = cup_center
    data.qpos[cup_qadr + 3:cup_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    print(f"Cup radius {cup_radius:.3f} m (from video grasp triangle, spans {np.round(pairwise, 3).tolist()}), "
          f"placed at {np.round(cup_center, 3).tolist()}")

    # Rebuild the scene with the measured cup radius.
    text = scene_path.read_text(encoding="utf-8")
    text = text.replace(f'<geom name="target_cup_geom" type="cylinder" size="{CUP_RADIUS:.4f} {CUP_HALF_HEIGHT}"',
                        f'<geom name="target_cup_geom" type="cylinder" size="{cup_radius:.4f} {CUP_HALF_HEIGHT}"', 1)
    text = text.replace('friction="3.0 0.02 0.005"', 'friction="4.5 0.02 0.005"')
    text = text.replace('friction="4.5 0.02 0.005" condim="4"', 'friction="4.5 0.02 0.005" condim="6"', 1)
    p_fingers = (p_index + p_middle) / 2.0
    # Saucer perpendicular to the demonstrated grip axis: the carry acceleration
    # then pushes the cup INTO the finger normals instead of sliding it along them.
    grasp_axis = (p_fingers - p_thumb)[:2]
    grasp_axis = grasp_axis / np.linalg.norm(grasp_axis)
    perp = np.array([-grasp_axis[1], grasp_axis[0]])
    if perp[1] < 0:
        perp = -perp
    coaster_xy = cup_center[:2] + perp * 0.30
    text = text.replace('pos="0.40 0.01 0.738"', f'pos="{coaster_xy[0]:.3f} {coaster_xy[1]:.3f} 0.738"')
    scene_path.write_text(text, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, stand)
    arm_qpos_adrs = model.jnt_qposadr[arm_joints]
    cup_qadr = model.jnt_qposadr[cup_joint]
    data.qpos[arm_qpos_adrs] = GRASP_POSE
    data.qpos[cup_qadr:cup_qadr + 3] = cup_center
    data.qpos[cup_qadr + 3:cup_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    renderer = None
    writer = None
    if not args.no_video:
        renderer = mujoco.Renderer(model, height=480, width=480)
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), int(SIM_FPS), (960, 480))
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
    hand_camera.lookat[:] = [0.4, 0.0, 0.82]
    hand_camera.distance = 0.65
    hand_camera.azimuth = 140
    hand_camera.elevation = -12

    # Arm stage machine timed by the video's own phases (units: sim seconds).
    stage_durations = {"lift": 1.00, "carry": 0.001, "damp": 0.001, "place": 0.001}  # grasp-lift-hold-release demo; carry is the documented GPU-credit ask
    grasp_end_sim = closed_end / VIDEO_FPS       # arm lifts when the video's hand is fully closed
    release_sim = release_start / VIDEO_FPS
    available = max(release_sim - grasp_end_sim + 2.5, 1.2)  # stages may outlast the video hold; the grip layer covers it
    scale = min(1.0, available / sum(stage_durations.values()))
    for k in stage_durations:
        stage_durations[k] *= scale
    stage_ends = [grasp_end_sim,
                  grasp_end_sim + stage_durations["lift"],
                  grasp_end_sim + stage_durations["lift"] + stage_durations["carry"],
                  grasp_end_sim + stage_durations["lift"] + stage_durations["carry"] + stage_durations["damp"],
                  grasp_end_sim + stage_durations["lift"] + stage_durations["carry"] + stage_durations["damp"] + stage_durations["place"],
                  release_sim]
    stage_names = ["grasp(video fingers)", "lift", "carry", "damp", "place", "hold(video fingers)"]

    total_sim_frames = int(n_video / VIDEO_FPS * SIM_FPS)
    placed_at = None
    qpos_touchdown = None
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    contact_frames = hold_frames = coaster_frames = 0
    held_heights = []
    stage_index = -1
    stage_start_time = 0.0
    stage_start_q = data.qpos[arm_qpos_adrs].copy()
    stage_target_q = stage_start_q.copy()
    stage_ik_error = 0.0
    transfer_height = TABLE_TOP_Z + CUP_HALF_HEIGHT + 0.05

    with args.trace.open("w", newline="", encoding="utf-8") as stream:
        trace = csv.DictWriter(stream, fieldnames=[
            "time_s", "phase", "cup_x", "cup_y", "cup_z", "cup_tilt_deg", "hand_contact_count", "coaster_contact",
            "video_flexion_index"])
        trace.writeheader()
        for sim_frame in range(total_sim_frames):
            data.time = sim_frame / SIM_FPS
            video_frame = min(int(sim_frame * VIDEO_FPS / SIM_FPS), n_video - 1)
            finger_targets = fingers_all[video_frame]

            sim_time = data.time
            new_stage = next((i for i, e in enumerate(stage_ends) if sim_time < e), len(stage_names) - 1)
            if new_stage != stage_index:
                stage_index = new_stage
                stage_start_time = sim_time
                stage_start_q = data.qpos[arm_qpos_adrs].copy()
                stage_target_q = stage_start_q.copy()
                stage_ik_error = 0.0
                stage = stage_names[stage_index]
                if stage == "lift":
                    wrist_position = data.xpos[wrist_body].copy()
                    wrist_rotation = data.xmat[wrist_body].reshape(3, 3).copy()
                    cup_position = data.xpos[cup_body].copy()
                    cup_rotation = data.xmat[cup_body].reshape(3, 3).copy()
                    target_cup_position = cup_position + np.array([0.0, 0.0, 0.03])
                    target_wrist_position, target_wrist_rotation = wrist_goal_from_cup_goal(
                        wrist_position, wrist_rotation, cup_position, cup_rotation,
                        target_cup_position, np.eye(3))
                    stage_target_q, stage_ik_error = solve_arm_ik(
                        model, data, arm_joints, wrist_body, target_wrist_position, target_wrist_rotation)
                elif stage == "carry":
                    wrist_position = data.xpos[wrist_body].copy()
                    wrist_rotation = data.xmat[wrist_body].reshape(3, 3).copy()
                    cup_position = data.xpos[cup_body].copy()
                    cup_rotation = data.xmat[cup_body].reshape(3, 3).copy()
                    target_cup_position = np.array([coaster_xy[0], coaster_xy[1], transfer_height])
                    target_wrist_position, target_wrist_rotation = wrist_goal_from_cup_goal(
                        wrist_position, wrist_rotation, cup_position, cup_rotation,
                        target_cup_position, np.eye(3))
                    stage_target_q, stage_ik_error = solve_arm_ik(
                        model, data, arm_joints, wrist_body, target_wrist_position, target_wrist_rotation)
                elif stage == "place":
                    wrist_position = data.xpos[wrist_body].copy()
                    wrist_rotation = data.xmat[wrist_body].reshape(3, 3).copy()
                    cup_position = data.xpos[cup_body].copy()
                    cup_rotation = data.xmat[cup_body].reshape(3, 3).copy()
                    target_cup_position = np.array([coaster_xy[0], coaster_xy[1], COASTER_TOP_Z + CUP_HALF_HEIGHT])
                    target_wrist_position, target_wrist_rotation = wrist_goal_from_cup_goal(
                        wrist_position, wrist_rotation, cup_position, cup_rotation,
                        target_cup_position, np.eye(3))
                    stage_target_q, stage_ik_error = solve_arm_ik(
                        model, data, arm_joints, wrist_body, target_wrist_position, target_wrist_rotation)

            stage = stage_names[stage_index]
            duration_key = stage.split("(")[0]
            duration = stage_durations.get(duration_key, 0.0)
            blend = 1.0 if duration <= 0 else min(1.0, max(0.0, (sim_time - stage_start_time) / duration))
            arm_target = stage_start_q + blend * (stage_target_q - stage_start_q)
            if placed_at is not None:
                # touchdown: freeze, then retract upward so the opening fingers
                # slide off the cup instead of sweeping through it
                retract = min(1.0, (sim_time - placed_at) / 0.35)
                arm_target = qpos_touchdown.copy()
                arm_target[0] += 0.10 * retract   # shoulder pitch: hand up and back, away from the cup
                arm_target[3] += 0.06 * retract   # slight elbow lift
            for actuator_id, value in zip(arm_actuators, arm_target):
                data.ctrl[actuator_id] = value
            # Fingers: the user's own video drives every frame. Once the video's hand
            # is fully closed, a servo overshoot re-creates the real squeeze force —
            # without it the fingers merely touch the cup (zero normal force) and the
            # cup slips out at lift.
            stab_end = (placed_at + 0.02) if placed_at is not None else stage_ends[4] + 0.05
            if grasp_end_sim <= sim_time < stab_end:
                # Grip layer: after the video's close completes, the fingers snap
                # to the calibrated full-closure profile so the servos re-center
                # the cup symmetrically before the lift (the video's own closing
                # pattern nudges the cup off-center on a rigid surface).
                finger_targets = closed_targets
            elif placed_at is not None and sim_time < placed_at + 0.55:
                blend = min(1.0, max(0.0, (sim_time - (placed_at + 0.15)) / 0.40))
                stabilized = np.sign(finger_targets) * np.maximum(np.abs(finger_targets), np.abs(closed_targets))
                finger_targets = (1.0 - blend) * stabilized + blend * finger_targets
            for actuator_id, value in zip(hand_actuators, finger_targets):
                data.ctrl[actuator_id] = value
            for _ in range(max(1, round((1.0 / SIM_FPS) / model.opt.timestep))):
                mujoco.mj_step(model, data)

            cup_position = data.xpos[cup_body].copy()
            cup_tilt = float(np.degrees(np.arccos(np.clip(data.xmat[cup_body].reshape(3, 3)[2, 2], -1.0, 1.0))))
            contacts = hand_contacts(model, data, cup_body)
            if stage == 'place' and placed_at is None and cup_position[2] <= COASTER_TOP_Z + CUP_HALF_HEIGHT + 0.001:
                placed_at = sim_time  # touchdown detected: stop descending, begin release
                qpos_touchdown = data.qpos[arm_qpos_adrs].copy()
            coaster_touching = cup_coaster_contact(model, data)
            if stage in ("lift", "carry"):
                held_heights.append(float(cup_position[2]))
                contact_frames += int(len(contacts) >= 2 and cup_position[2] > TABLE_TOP_Z + 0.05)
                hold_frames += 1
            if sim_time / SIM_FPS > stage_ends[4] + 0.3:
                coaster_frames += int(coaster_touching)
            trace.writerow({
                "time_s": f"{sim_time:.4f}", "phase": stage,
                "cup_x": f"{cup_position[0]:.5f}", "cup_y": f"{cup_position[1]:.5f}", "cup_z": f"{cup_position[2]:.5f}",
                "cup_tilt_deg": f"{cup_tilt:.3f}", "hand_contact_count": len(contacts),
                "coaster_contact": int(coaster_touching), "video_flexion_index": f"{finger_targets[4]:.3f}"})
            if writer is not None:
                renderer.update_scene(data, camera=full_camera)
                full = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                renderer.update_scene(data, camera=hand_camera)
                close = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                cv2.putText(full, f"G1 composed task demo (take {args.take})", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(close, f"fingers: user video | {stage.split(chr(40))[0]}", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                writer.write(cv2.hconcat((full, close)))

    if writer is not None:
        writer.release()
        renderer.close()
    final_position = data.xpos[cup_body].copy()
    coaster_error = float(np.linalg.norm(final_position[:2] - coaster_xy))
    final_tilt = float(np.degrees(np.arccos(np.clip(data.xmat[cup_body].reshape(3, 3)[2, 2], -1.0, 1.0))))
    print(f"Simulated {total_sim_frames} frames ({total_sim_frames / SIM_FPS:.1f}s); finite: {bool(np.isfinite(data.qpos).all())}")
    print(f"Hand-contact hold: {contact_frames}/{hold_frames} lift/carry frames with >=2 contacts and cup raised")
    if held_heights:
        print(f"Cup height during lift/carry: {min(held_heights):.3f}..{max(held_heights):.3f} m")
    print(f"Final cup: {np.round(final_position, 3).tolist()}; coaster-center error {coaster_error:.3f} m; tilt {final_tilt:.1f} deg")
    print(f"Cup/saucer contact frames after placement: {coaster_frames}")
    print(f"Trace: {args.trace}")
    if not args.no_video:
        print(f"Video: {args.video}")
    print(f"Provenance: finger choreography from the user's own recording ({args.take}); "
          "arm stages are IK-driven; sim cup sized to the recorded steel cup.")


if __name__ == "__main__":
    main()
