from __future__ import annotations

import platform
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "vendor/mujoco_menagerie/unitree_g1/scene_with_hands.xml"


def main() -> None:
    if not MODEL.exists():
        raise FileNotFoundError(f"G1 model missing: {MODEL}")
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand_id < 0:
        raise RuntimeError("The selected G1 model has no 'stand' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, stand_id)
    mujoco.mj_forward(model, data)
    for _ in range(3000):
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all():
        raise RuntimeError("MuJoCo state diverged during the stand-pose check")
    print(f"Platform: {platform.platform()} ({platform.machine()})")
    print(f"MuJoCo: {mujoco.__version__}")
    print(f"G1 model: {MODEL.relative_to(ROOT)}")
    print(f"nq={model.nq}, nv={model.nv}, actuators={model.nu}, bodies={model.nbody}")
    print(f"Stand-pose simulation: 3000 steps, finite state, root z={data.qpos[2]:.3f} m")
    print("G1 three-finger hand actuators:")
    for name in (
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    ):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"Required hand actuator missing: {name}")
        print(f"  {name}")


if __name__ == "__main__":
    main()

