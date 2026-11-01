from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
G1_DIR = ROOT / "vendor/mujoco_menagerie/unitree_g1"
SOURCE = G1_DIR / "scene_with_hands.xml"
TASK_SCENE = G1_DIR / "scene_with_ball.xml"

BALL = """
    <body name="target_ball" pos="0.0 -0.242 0.553">
      <freejoint name="target_ball_free"/>
      <geom name="target_ball_geom" type="sphere" size="0.035" mass="0.08"
            rgba="0.95 0.22 0.12 1" friction="1.2 0.01 0.001" condim="4"/>
    </body>
"""


def ensure_scene() -> Path:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Menagerie model not found at {SOURCE}; run scripts/setup_menagerie.sh")
    text = SOURCE.read_text(encoding="utf-8")
    marker = "</worldbody>"
    if text.count(marker) != 1:
        raise RuntimeError("Unexpected upstream scene structure; could not add task object safely")
    text = text.replace(marker, BALL + "\n  " + marker, 1)
    TASK_SCENE.write_text(text, encoding="utf-8")
    return TASK_SCENE


if __name__ == "__main__":
    print(ensure_scene())
