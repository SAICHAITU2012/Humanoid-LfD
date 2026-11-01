from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "vendor/mujoco_menagerie/unitree_g1/scene_with_hands.xml"
HAND_JOINTS = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]
HAND_CLOSED = np.array([-0.55, -0.35, -1.10, 1.20, 1.35, 1.20, 1.35])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive MuJoCo viewer: G1 stand pose with a live hand open/close cycle."
    )
    parser.add_argument("--cycle-seconds", type=float, default=4.0, help="Duration of one open/close cycle.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand < 0:
        raise SystemExit("The G1 model has no 'stand' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, stand)
    actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in HAND_JOINTS]
    if min(actuator_ids) < 0:
        raise SystemExit("Could not find all G1 right-hand actuators")

    # Interactive viewer; on macOS launch this script with .venv/bin/mjpython.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Start wide; the viewer default camera is quite zoomed in.
        viewer.cam.lookat[:] = [0.0, 0.0, 0.80]
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 140.0
        viewer.cam.elevation = -12.0
        while viewer.is_running():
            phase = 0.5 * (1.0 - np.cos(2.0 * np.pi * data.time / args.cycle_seconds))
            for actuator_id, closed in zip(actuator_ids, HAND_CLOSED):
                data.ctrl[actuator_id] = closed * phase
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.004)  # Cap the loop so long viewer sessions stay cool.
    print("Viewer closed.")


if __name__ == "__main__":
    main()
