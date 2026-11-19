import argparse
import sys
from pathlib import Path

import cv2

from detection.person_detector import PersonDetector
from tracking.person_tracker import PersonTracker
from visualization.renderer import draw_person_detections
from visualization.tracking_renderer import draw_tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demostracion base de deteccion de personas con YOLO.",
    )
    parser.add_argument(
        "--source",
        default="webcam",
        help="Ruta de imagen/video, indice de webcam (0) o 'webcam'.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Modelo YOLO preentrenado.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Umbral de confianza entre 0 y 1.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Dispositivo de inferencia: cpu o cuda.",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Activar seguimiento mediante tracking integrado.",
    )
    parser.add_argument(
        "--tracker",
        default="tracking/bytetrack_stable.yaml",
        choices=("tracking/bytetrack_stable.yaml", "tracking/botsort_stable.yaml"),
        help="Algoritmo de seguimiento.",
    )
    parser.add_argument(
        "--trajectory",
        type=int,
        default=30,
        help="Cantidad maxima de puntos mostrados por trayectoria.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Ruta opcional para guardar una imagen procesada.",
    )
    return parser.parse_args()


def resolve_source(source: str) -> str | int:
    if source.lower() in {"webcam", "camera", "camara"}:
        return 0

    if source.isdigit():
        return int(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo o fuente: {source}")
    return str(path)


def process_image(
    detector: PersonDetector,
    image_path: str,
    output_path: Path | None,
) -> bool:
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"No se pudo leer la imagen: {image_path}", file=sys.stderr)
        return False

    detections = detector.detect(frame)
    rendered = draw_person_detections(frame, detections)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), rendered)
        print(f"Imagen procesada guardada en: {output_path}")

    cv2.imshow("Deteccion de personas - imagen", rendered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return True


def process_video(
    detector: PersonDetector,
    source: str | int,
    tracker: PersonTracker | None = None,
) -> bool:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"No se pudo abrir la fuente: {source}", file=sys.stderr)
        return False

    window_name = "Deteccion de personas"

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if tracker is None:
                detections = detector.detect(frame)
                rendered = draw_person_detections(frame, detections)
            else:
                tracks = tracker.update(frame)
                rendered = draw_tracks(frame, tracks, tracker)
            cv2.imshow(window_name, rendered)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return True


def main() -> int:
    args = parse_args()

    try:
        source = resolve_source(args.source)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 2

    try:
        detector = PersonDetector(
            model_name=args.model,
            confidence=args.conf,
            device=args.device,
        )
    except (ImportError, ValueError, RuntimeError) as error:
        print(f"No se pudo cargar el detector: {error}", file=sys.stderr)
        return 3

    source_suffix = Path(source).suffix.lower() if isinstance(source, str) else ""
    if source_suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return 0 if process_image(detector, str(source), args.output) else 4

    active_tracker: PersonTracker | None = None
    if args.track:
        try:
            active_tracker = PersonTracker(
                model_name=args.model,
                confidence=args.conf,
                device=args.device,
                tracker=args.tracker,
                trajectory_length=args.trajectory,
            )
        except (ImportError, ValueError, RuntimeError) as error:
            print(f"No se pudo iniciar el seguimiento: {error}", file=sys.stderr)
            return 6

    return 0 if process_video(detector, source, active_tracker) else 5


if __name__ == "__main__":
    raise SystemExit(main())
