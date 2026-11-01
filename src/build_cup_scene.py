from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
G1_DIR = ROOT / "vendor/mujoco_menagerie/unitree_g1"
SOURCE = G1_DIR / "scene_with_hands.xml"
TASK_SCENE = G1_DIR / "scene_with_cup.xml"

CUP_SCENE = """
    <geom name="task_table_top" type="box" pos="0.55 -0.05 0.705" size="0.48 0.40 0.025"
          rgba="0.36 0.24 0.14 1" friction="1.2 0.01 0.001" condim="4"/>
    <geom name="task_table_leg_front_left" type="box" pos="0.10 -0.38 0.35" size="0.025 0.025 0.35"
          rgba="0.30 0.20 0.12 1"/>
    <geom name="task_table_leg_front_right" type="box" pos="0.10 0.28 0.35" size="0.025 0.025 0.35"
          rgba="0.30 0.20 0.12 1"/>
    <geom name="task_table_leg_back_left" type="box" pos="0.98 -0.38 0.35" size="0.025 0.025 0.35"
          rgba="0.30 0.20 0.12 1"/>
    <geom name="task_table_leg_back_right" type="box" pos="0.98 0.28 0.35" size="0.025 0.025 0.35"
          rgba="0.30 0.20 0.12 1"/>
    <geom name="target_coaster" type="cylinder" pos="0.40 0.01 0.738" size="0.075 0.008"
          rgba="0.12 0.48 0.78 1" friction="1.4 0.01 0.001" condim="4"/>
    <body name="target_cup" pos="0.39 -0.05 0.785">
      <freejoint name="target_cup_free"/>
      <geom name="target_cup_geom" type="cylinder" size="0.03 0.055" mass="0.04"
            rgba="0.92 0.72 0.20 1" friction="3.0 0.02 0.005" condim="4"/>
      <geom name="cup_handle_upper" type="capsule" fromto="0.025 0 0.028 0.058 0 0.028" size="0.005"
            rgba="0.92 0.72 0.20 1" density="0" contype="0" conaffinity="0"/>
      <geom name="cup_handle_side" type="capsule" fromto="0.058 0 0.028 0.058 0 -0.028" size="0.005"
            rgba="0.92 0.72 0.20 1" density="0" contype="0" conaffinity="0"/>
      <geom name="cup_handle_lower" type="capsule" fromto="0.058 0 -0.028 0.025 0 -0.028" size="0.005"
            rgba="0.92 0.72 0.20 1" density="0" contype="0" conaffinity="0"/>
    </body>
"""


def ensure_scene() -> Path:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Menagerie model not found at {SOURCE}; run scripts/setup_menagerie.sh")
    text = SOURCE.read_text(encoding="utf-8")
    marker = "</worldbody>"
    if text.count(marker) != 1:
        raise RuntimeError("Unexpected upstream scene structure; could not add cup task safely")
    text = text.replace(marker, CUP_SCENE + "\n  " + marker, 1)
    TASK_SCENE.write_text(text, encoding="utf-8")
    return TASK_SCENE


if __name__ == "__main__":
    print(ensure_scene())
