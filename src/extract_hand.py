from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp


CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract timestamped hand landmarks from your own video using MediaPipe Tasks.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, help="Optional annotated MP4 for reviewing tracking quality.")
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[1] / "models/hand_landmarker.task")
    parser.add_argument("--max-hands", type=int, default=1)
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"Hand-landmarker model missing: {args.model}. Run scripts/download_hand_model.sh first.")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.overlay:
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise SystemExit(f"Could not create overlay video: {args.overlay}")

    tasks = mp.tasks
    options = tasks.vision.HandLandmarkerOptions(
        base_options=tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=tasks.vision.RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    frame_index = 0
    detected_frames = 0
    with tasks.vision.HandLandmarker.create_from_options(options) as landmarker, args.out.open("w", encoding="utf-8") as out:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = round(frame_index * 1000.0 / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, timestamp_ms)
            entry = {"frame": frame_index, "time_s": frame_index / fps, "width": width, "height": height, "hands": []}
            for i, landmarks in enumerate(result.hand_landmarks):
                category = "unknown"
                if i < len(result.handedness) and result.handedness[i]:
                    category = getattr(result.handedness[i][0], "category_name", "unknown").lower()
                normalized = [[float(p.x), float(p.y), float(p.z)] for p in landmarks]
                world = []
                if i < len(result.hand_world_landmarks):
                    world = [[float(p.x), float(p.y), float(p.z)] for p in result.hand_world_landmarks[i]]
                entry["hands"].append({"handedness": category, "landmarks": normalized, "world_landmarks": world})
                detected_frames += 1
                if writer:
                    pts = [(round(p[0] * width), round(p[1] * height)) for p in normalized]
                    for a, b in CONNECTIONS:
                        cv2.line(frame, pts[a], pts[b], (40, 220, 40), 2, cv2.LINE_AA)
                    for x, y in pts:
                        cv2.circle(frame, (x, y), 3, (30, 60, 240), -1, cv2.LINE_AA)
            out.write(json.dumps(entry, separators=(",", ":")) + "\n")
            if writer:
                writer.write(frame)
            frame_index += 1
    cap.release()
    if writer:
        writer.release()
    if frame_index == 0:
        raise SystemExit("The video contained no readable frames")
    print(f"Frames: {frame_index}; frames with a detected hand: {detected_frames}; source FPS: {fps:.2f}")
    print(f"Landmarks: {args.out}")
    if args.overlay:
        print(f"Review tracking quality in: {args.overlay}")
    print("World landmarks describe hand-relative shape, not the camera's metric wrist position.")


if __name__ == "__main__":
    main()

