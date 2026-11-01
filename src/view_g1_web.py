from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import mujoco
import numpy as np
from dataclasses import replace

from gait import GaitParams, HIP0, KNEE0, walk_targets

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "vendor/mujoco_menagerie/unitree_g1/scene_with_hands.xml"
SPEC_PATH = ROOT / "outputs/g1_control_spec.json"
STATE_PATH = ROOT / "outputs/g1_control_state.json"

ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
HAND_JOINTS = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]
CONTROL_JOINTS = ARM_JOINTS + HAND_JOINTS
WALK_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_knee_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "waist_roll_joint", "waist_pitch_joint",
]
WALK_WAIST_PITCH = 0.08  # forward trunk bias; counters the gait's backward drift
SETTLE_S = 0.8   # stand -> crouch before stepping
RAMP_S = 1.2     # step amplitudes ramp in after settle
HALT_S = 0.6     # walk targets -> stand
FALL_Z = 0.45    # auto-halt when the pelvis drops this low


def main() -> None:
    parser = argparse.ArgumentParser(
        description="G1 viewer controlled by src/g1_control_server.py via outputs/g1_control_state.json."
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand < 0:
        raise SystemExit("The G1 model has no 'stand' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, stand)

    def hold_stand() -> None:
        # Hold every joint actuator at its stand-pose angle so the robot stays stable.
        for aid in range(model.nu):
            jid = model.actuator_trnid[aid, 0]
            data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]

    hold_stand()
    mujoco.mj_forward(model, data)

    name2aid = {}
    actuators = []
    for name in CONTROL_JOINTS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise SystemExit(f"Missing actuator: {name}")
        lo, hi = model.actuator_ctrlrange[aid]
        name2aid[name] = aid
        actuators.append({"name": name, "lo": float(lo), "hi": float(hi),
                          "value": float(data.ctrl[aid]), "hand": name in HAND_JOINTS})
    SPEC_PATH.write_text(json.dumps({"actuators": actuators}), encoding="utf-8")
    STATE_PATH.write_text(json.dumps({
        "targets": {a["name"]: a["value"] for a in actuators},
        "reset": False,
        "mode": "stand",
        "walk_speed": 1.0,
        "seq": 0,
    }), encoding="utf-8")

    walk_aids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in WALK_JOINTS}
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    foot_body = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")
                 for s in ("left", "right")}
    gait = GaitParams()

    def foot_contact(side: str) -> bool:
        body = foot_body[side]
        for c in range(data.ncon):
            g1, g2 = data.contact[c].geom1, data.contact[c].geom2
            if (model.geom_bodyid[g1] == body or model.geom_bodyid[g2] == body) and data.contact[c].dist < 0.005:
                return True
        return False

    def crouch_pose() -> dict[str, float]:
        pose = {n: 0.0 for n in WALK_JOINTS}
        for s in ("left", "right"):
            pose[f"{s}_hip_pitch_joint"] = HIP0
            pose[f"{s}_knee_joint"] = KNEE0
            pose[f"{s}_ankle_pitch_joint"] = gait.ankle_level
            pose[f"{s}_hip_roll_joint"] = gait.spread if s == "left" else -gait.spread
        pose["waist_pitch_joint"] = WALK_WAIST_PITCH
        return pose

    def stand_pose() -> dict[str, float]:
        return {n: 0.0 for n in WALK_JOINTS}

    # Interactive viewer; on macOS launch this script with .venv/bin/mjpython.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.80]
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 140.0
        viewer.cam.elevation = -12.0
        last_mtime = None
        targets = {a["name"]: a["value"] for a in actuators}
        # Walking state: "stand" or "walk". The walk branch is a verbatim port
        # of the headless harness in tune trials (single time-based loop), so
        # live behavior matches the validated +0.55 m / ~5 s walks exactly.
        walk_phase = "stand"
        walk_t0 = 0.0
        halt_from: dict[str, float] = {}
        halt_t0 = 0.0
        bal = {"pr": 0.0, "pp": 0.0, "rr": 0.0, "rp": 0.0}
        speed = 1.0
        while viewer.is_running():
            try:
                mtime = STATE_PATH.stat().st_mtime
            except FileNotFoundError:
                mtime = None
            if mtime != last_mtime:
                last_mtime = mtime
                command = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                speed = max(0.8, min(1.0, float(command.get("walk_speed", 1.0))))
                if command.get("reset"):
                    mujoco.mj_resetDataKeyframe(model, data, stand)
                    hold_stand()
                    targets = {a["name"]: float(data.ctrl[name2aid[a["name"]]]) for a in actuators}
                    walk_phase = "stand"
                    # Publish the reset pose back so the browser sliders follow.
                    STATE_PATH.write_text(json.dumps({"targets": targets, "reset": False,
                                                      "mode": "stand", "walk_speed": speed,
                                                      "seq": command.get("seq", 0) + 1}), encoding="utf-8")
                    last_mtime = STATE_PATH.stat().st_mtime
                else:
                    targets = {name: float(value) for name, value in command.get("targets", targets).items()}
                    mode = command.get("mode", "stand")
                    if mode == "walk" and walk_phase != "walk":
                        if float(data.xpos[pelvis][2]) < 0.55 or not np.isfinite(data.qpos).all():
                            # Fallen: stand the robot back up before walking.
                            mujoco.mj_resetDataKeyframe(model, data, stand)
                            hold_stand()
                            targets = {a["name"]: float(data.ctrl[name2aid[a["name"]]]) for a in actuators}
                        walk_phase = "walk"
                        walk_t0 = data.time
                        bal = {"pr": 0.0, "pp": 0.0, "rr": 0.0, "rp": 0.0}
                    elif mode == "stand" and walk_phase == "walk":
                        walk_phase = "halt"
                        halt_t0 = data.time
                        halt_from = {n: float(data.ctrl[a]) for n, a in walk_aids.items()}

            if walk_phase == "walk":
                t = data.time - walk_t0
                crouch = crouch_pose()
                if t < SETTLE_S:
                    # Harness parity: legs snap to the crouch and hold it.
                    for n, a in walk_aids.items():
                        data.ctrl[a] = crouch[n]
                else:
                    wt = t - SETTLE_S
                    step_scale = min(1.0, wt / RAMP_S)
                    leg_targets = walk_targets(wt, replace(gait, freq=gait.freq * speed), step_scale)
                    roll = float(data.xquat[pelvis][1])
                    pitch = float(data.xquat[pelvis][2])
                    dt = model.opt.timestep
                    bal["rr"] = 0.7 * bal["rr"] + 0.3 * (roll - bal["pr"]) / dt
                    bal["rp"] = 0.7 * bal["rp"] + 0.3 * (pitch - bal["pp"]) / dt
                    bal["pr"], bal["pp"] = roll, pitch
                    u_roll = max(-0.08, min(0.08, 1.2 * roll + 0.25 * bal["rr"]))
                    u_pit = max(-0.12, min(0.12, 1.5 * pitch + 0.3 * bal["rp"]))
                    l_c, r_c = foot_contact("left"), foot_contact("right")
                    if l_c and r_c:      # double support: hold the trunk
                        for s in ("left", "right"):
                            leg_targets[f"{s}_hip_pitch_joint"] += u_pit
                    elif l_c:            # single support: rotate about stance foot
                        leg_targets["left_hip_roll_joint"] -= u_roll
                        leg_targets["left_hip_pitch_joint"] -= u_pit
                    elif r_c:
                        leg_targets["right_hip_roll_joint"] -= u_roll
                        leg_targets["right_hip_pitch_joint"] -= u_pit
                    for n, a in walk_aids.items():
                        data.ctrl[a] = leg_targets.get(n, crouch[n])
                if float(data.xpos[pelvis][2]) < FALL_Z or not np.isfinite(data.qpos).all():
                    walk_phase = "stand"  # stumbled: stop commanding the legs
                    try:
                        command = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                        command["mode"] = "stand"
                        STATE_PATH.write_text(json.dumps(command), encoding="utf-8")
                        last_mtime = STATE_PATH.stat().st_mtime
                    except (FileNotFoundError, json.JSONDecodeError):
                        pass
            elif walk_phase == "halt":
                blend = min(1.0, (data.time - halt_t0) / HALT_S)
                stand_p = stand_pose()
                for n, a in walk_aids.items():
                    data.ctrl[a] = (1.0 - blend) * halt_from[n] + blend * stand_p[n]
                if blend >= 1.0:
                    walk_phase = "stand"

            for name, value in targets.items():
                data.ctrl[name2aid[name]] = value
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)  # Cap the loop near real-time pace and keep the Mac cool.
    print("Web-controls viewer closed.")


if __name__ == "__main__":
    main()
