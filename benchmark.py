import argparse
import time
from pathlib import Path

import cv2

from device_support import print_device_status, resolve_device
from tracking.person_tracker import PersonTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mide FPS y latencia aproximada del seguimiento.",
    )
    parser.add_argument("source", help="Video MP4 o indice de camara.")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tracker", default="tracking/bytetrack_stable.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frames", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_device_status()

    source: str | int
    if args.source.isdigit():
        source = int(args.source)
    else:
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"No existe el video: {args.source}")
            return 2
        source = str(source_path)

    try:
        device = resolve_device(args.device)
    except (RuntimeError, ValueError) as error:
        print(error)
        return 3

    tracker = PersonTracker(
        model_name="yolo11n.pt",
        confidence=args.conf,
        device=device,
        tracker=args.tracker,
        trajectory_length=30,
        imgsz=args.imgsz,
    )
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print("No se pudo abrir la fuente.")
        return 4

    latencies: list[float] = []
    track_counts: list[int] = []
    frame_count = 0
    try:
        started = time.perf_counter()
        while frame_count < args.frames:
            success, frame = capture.read()
            if not success:
                break
            inference_started = time.perf_counter()
            tracks = tracker.update(frame)
            latencies.append(time.perf_counter() - inference_started)
            track_counts.append(len(tracks))
            frame_count += 1
    finally:
        capture.release()

    elapsed = time.perf_counter() - started
    fps = frame_count / elapsed if elapsed > 0 else 0
    average_latency = (
        sum(latencies) / len(latencies) if latencies else 0
    )
    p95_latency = 0.0
    if latencies:
        ordered = sorted(latencies)
        p95_latency = ordered[max(0, int(len(ordered) * 0.95) - 1)]

    print(f"Fotogramas procesados: {frame_count}")
    print(f"FPS promedio: {fps:.2f}")
    print(f"Latencia promedio: {average_latency * 1000:.1f} ms")
    print(f"Latencia p95: {p95_latency * 1000:.1f} ms")
    average_tracks = (
        sum(track_counts) / len(track_counts) if track_counts else 0
    )
    print(f"Promedio de tracks por fotograma: {average_tracks:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
