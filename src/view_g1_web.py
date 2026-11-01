from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mujoco


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
        "seq": 0,
    }), encoding="utf-8")

    # Interactive viewer; on macOS launch this script with .venv/bin/mjpython.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.80]
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 140.0
        viewer.cam.elevation = -12.0
        last_mtime = None
        targets = {a["name"]: a["value"] for a in actuators}
        while viewer.is_running():
            try:
                mtime = STATE_PATH.stat().st_mtime
            except FileNotFoundError:
                mtime = None
            if mtime != last_mtime:
                last_mtime = mtime
                command = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                if command.get("reset"):
                    mujoco.mj_resetDataKeyframe(model, data, stand)
                    hold_stand()
                    targets = {a["name"]: float(data.ctrl[name2aid[a["name"]]]) for a in actuators}
                    # Publish the reset pose back so the browser sliders follow.
                    STATE_PATH.write_text(json.dumps({"targets": targets, "reset": False,
                                                      "seq": command.get("seq", 0) + 1}), encoding="utf-8")
                    last_mtime = STATE_PATH.stat().st_mtime
                else:
                    targets = {name: float(value) for name, value in command.get("targets", targets).items()}
            for name, value in targets.items():
                data.ctrl[name2aid[name]] = value
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)  # Cap the loop near real-time pace and keep the Mac cool.
    print("Web-controls viewer closed.")


if __name__ == "__main__":
    main()
