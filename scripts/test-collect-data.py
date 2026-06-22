import argparse
import json
import signal
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

# MediaPipe 33 landmarks -> COCO 17 keypoints
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

COCO_BONES = [
    (5, 7), (7, 9), (6, 8), (8, 10),   # arms
    (5, 6),                              # shoulders
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (11, 12),                            # hips
    (5, 11), (6, 12),                    # torso
    (0, 1), (0, 2), (1, 3), (2, 4),     # face
]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
MODEL_FILENAME = "pose_landmarker_lite.task"


def ensure_model(cache_dir: Path) -> Path:
    model_path = cache_dir / MODEL_FILENAME
    if model_path.exists():
        return model_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_FILENAME} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, str(model_path))
        print(f"  saved to {model_path}")
    except Exception as exc:
        print(f"ERROR: Failed to download model: {exc}", file=sys.stderr)
        sys.exit(1)
    return model_path


def post_json(url: str, payload: dict | None = None, timeout: float = 5.0) -> bool:
    data = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        print(f"WARNING: POST {url} failed: {exc}", file=sys.stderr)
        return False


def draw_skeleton(frame: np.ndarray, persons_keypoints: list, w: int, h: int):
    colors = [(0, 255, 0), (0, 165, 255), (255, 100, 0)]
    for p_idx, keypoints in enumerate(persons_keypoints):
        color = colors[p_idx % len(colors)]
        pts = []
        for x, y in keypoints:
            px, py = int(x * w), int(y * h)
            pts.append((px, py))
            cv2.circle(frame, (px, py), 4, color, -1)

        for i, j in COCO_BONES:
            if i < len(pts) and j < len(pts):
                cv2.line(frame, pts[i], pts[j], color, 2)


def main():
    parser = argparse.ArgumentParser(description="Collect MULTI-PERSON ground-truth keypoints")
    parser.add_argument("--server", default="http://localhost:8080", help="Sensing server URL")
    parser.add_argument("--preview", action="store_true", help="Show live skeleton overlay")
    parser.add_argument("--duration", type=int, default=300, help="Recording duration in seconds")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--output", default="data/ground-truth", help="Output directory")
    parser.add_argument("--max-poses", type=int, default=3, help="Maximum number of people to detect")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = repo_root / "data" / ".cache"

    model_path = ensure_model(cache_dir)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Cannot open camera", file=sys.stderr)
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {frame_w}x{frame_h}")

    # === Multi-person support ===
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_poses=args.max_poses,                    # Cho phép nhiều người
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"keypoints_multi_{timestamp_str}.jsonl"
    out_file = open(out_path, "w", encoding="utf-8")
    print(f"Output: {out_path}")

    recording_url_start = f"{args.server}/api/v1/recording/start"
    recording_url_stop = f"{args.server}/api/v1/recording/stop"
    csi_started = post_json(recording_url_start)

    shutdown_requested = False
    def _handle_signal(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    start_time = time.monotonic()
    frame_count = 0
    total_confidence = 0.0
    total_visible = 0

    print(f"Collecting for {args.duration}s (max {args.max_poses} persons)... (press 'q' to stop)")

    try:
        while not shutdown_requested:
            elapsed = time.monotonic() - start_time
            if elapsed >= args.duration:
                break

            ret, frame = cap.read()
            if not ret:
                continue

            ts_ns = time.time_ns()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect(mp_image)

            persons_data = []
            frame_confidence = 0.0
            frame_visible = 0

            for landmarks in result.pose_landmarks:
                keypoints = []
                visibilities = []
                for coco_idx in range(17):
                    mp_idx = MP_TO_COCO[coco_idx]
                    lm = landmarks[mp_idx]
                    keypoints.append([round(lm.x, 5), round(lm.y, 5)])
                    vis = lm.visibility if hasattr(lm, 'visibility') else 0.0
                    visibilities.append(vis)

                confidence = float(np.mean(visibilities))
                n_visible = int(sum(1 for v in visibilities if v > 0.5))

                persons_data.append({
                    "keypoints": keypoints,
                    "confidence": round(confidence, 4),
                    "n_visible": n_visible
                })

                frame_confidence += confidence
                frame_visible += n_visible

            # Average per frame
            n_persons = len(persons_data)
            avg_conf = frame_confidence / n_persons if n_persons > 0 else 0.0

            record = {
                "ts_ns": ts_ns,
                "persons": persons_data,
                "n_persons": n_persons,
                "confidence": round(avg_conf, 4)
            }
            out_file.write(json.dumps(record) + "\n")

            frame_count += 1
            total_confidence += avg_conf
            total_visible += frame_visible

            # Preview
            if args.preview:
                keypoints_list = [p["keypoints"] for p in persons_data]
                draw_skeleton(frame, keypoints_list, frame_w, frame_h)
                
                cv2.putText(frame, f"Persons: {n_persons}  Visible: {frame_visible}  Time: {int(args.duration - elapsed)}s", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Multi-Person Ground Truth Collection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        out_file.close()
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()
        landmarker.close()

        if csi_started:
            post_json(recording_url_stop)

        avg_conf = total_confidence / frame_count if frame_count > 0 else 0.0
        avg_vis = total_visible / frame_count if frame_count > 0 else 0.0
        print("\n=== Collection Summary ===")
        print(f"  Total frames:       {frame_count}")
        print(f"  Avg confidence:     {avg_conf:.3f}")
        print(f"  Avg visible joints: {avg_vis:.1f} / 17")
        print(f"  Output:             {out_path}")


if __name__ == "__main__":
    main()