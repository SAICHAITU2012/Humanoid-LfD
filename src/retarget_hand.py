from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


JOINTS = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]


def flexion(points: list[list[float]], a: int, b: int, c: int) -> float:
    """Joint-angle flexion from hand-relative 3D landmarks; 0 straight, 1 bent."""
    dims = min(3, len(points[a]))
    p0 = np.asarray(points[a][:dims], dtype=float)
    p1 = np.asarray(points[b][:dims], dtype=float)
    p2 = np.asarray(points[c][:dims], dtype=float)
    u, v = p0 - p1, p2 - p1
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom < 1e-7:
        return 0.0
    angle = math.acos(float(np.clip(np.dot(u, v) / denom, -1.0, 1.0)))
    return float(np.clip((math.pi - angle) / (math.pi * 0.65), 0.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert detected human finger flexion to G1 three-finger targets.")
    parser.add_argument("landmarks", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right", "any"), default="any")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    previous = [0.0] * len(JOINTS)
    tracked_count = 0
    with args.landmarks.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            hands = item["hands"]
            hand = next((h for h in hands if args.side == "any" or h["handedness"] == args.side), None)
            tracked = hand is not None
            if tracked:
                p = hand.get("world_landmarks") or hand["landmarks"]
                thumb = flexion(p, 2, 3, 4)
                index = flexion(p, 5, 6, 8)
                middle = flexion(p, 9, 10, 12)
                # G1's right-hand joints have asymmetric sign/range conventions.
                previous = [
                    -0.55 * thumb, -0.35 * thumb, -1.10 * thumb,
                    1.20 * index, 1.35 * index,
                    1.20 * middle, 1.35 * middle,
                ]
                tracked_count += 1
            rows.append({"frame": item["frame"], "time_s": item["time_s"], "tracked": int(tracked), **dict(zip(JOINTS, previous))})
    if len(rows) < 2 or tracked_count < 2:
        raise SystemExit("Need at least two frames with a detected hand; inspect the video/overlay and try again.")
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} frames ({tracked_count} tracked) to {args.out}; missing frames hold the last pose.")
    print("Review target signs/ranges in MuJoCo. This is hand-shape retargeting, not global wrist tracking or contact-aware IK.")


if __name__ == "__main__":
    main()

