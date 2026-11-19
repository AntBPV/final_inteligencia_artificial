import argparse
import sys
import uuid
from pathlib import Path

import cv2

from detection.person_detector import PersonDetector
from counting.line_counter import LineCounter
from analytics.assistant import OccupancyAssistant
from analytics.statistics import OccupancyStatistics
from storage.repository import CountRepository
from tracking.person_tracker import PersonTracker
from visualization.renderer import draw_person_detections
from visualization.tracking_renderer import draw_tracks
from visualization.counting_renderer import draw_counting_overlay


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
        "--count",
        action="store_true",
        help="Activar conteo de entradas y salidas al cruzar una linea.",
    )
    parser.add_argument("--line-x1", type=float, default=0.0)
    parser.add_argument("--line-y1", type=float, default=0.50)
    parser.add_argument("--line-x2", type=float, default=1.0)
    parser.add_argument("--line-y2", type=float, default=0.50)
    parser.add_argument(
        "--reverse-direction",
        action="store_true",
        help="Invierte que lado representa entrada y salida.",
    )
    parser.add_argument(
        "--confirmation",
        type=int,
        default=3,
        help="Fotogramas consecutivos necesarios para confirmar un cruce.",
    )
    parser.add_argument(
        "--line-margin",
        type=float,
        default=8.0,
        help="Margen en pixeles alrededor de la linea.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Ruta opcional para guardar una imagen procesada.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/occupancy.sqlite3"),
        help="Ruta de la base SQLite.",
    )
    parser.add_argument("--ask", help="Consultar una estadistica y finalizar.")
    parser.add_argument("--stats", action="store_true", help="Mostrar resumen hoy.")
    parser.add_argument("--interactive", action="store_true", help="Abrir asistente interactivo.")
    parser.add_argument("--export", type=Path, help="Exportar eventos a CSV.")
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
    count: bool = False,
    line_points: tuple[float, float, float, float] = (0.0, 0.50, 1.0, 0.50),
    reverse_direction: bool = False,
    confirmation_frames: int = 3,
    line_margin_pixels: float = 8.0,
    repository: CountRepository | None = None,
    session_id: str | None = None,
) -> bool:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"No se pudo abrir la fuente: {source}", file=sys.stderr)
        return False

    window_name = "Deteccion de personas"
    counter: LineCounter | None = None
    recent_events: list[str] = []

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            tracks = []
            if tracker is None:
                detections = detector.detect(frame)
                rendered = draw_person_detections(frame, detections)
            else:
                tracks = tracker.update(frame)
                rendered = draw_tracks(frame, tracks, tracker)

            if count:
                if counter is None:
                    height, width = frame.shape[:2]
                    counter = LineCounter(
                        frame_width=width,
                        frame_height=height,
                        line_start=(line_points[0], line_points[1]),
                        line_end=(line_points[2], line_points[3]),
                        confirmation_frames=confirmation_frames,
                        line_margin_pixels=line_margin_pixels,
                        reverse_direction=reverse_direction,
                    )

                events = counter.update(tracks)
                for event in events:
                    label = f"{event.event_type.upper()} ID {event.track_id}"
                    recent_events.append(label)
                    print(label)
                    if repository is not None:
                        repository.save_event(
                            session_id=session_id or "unknown",
                            track_id=event.track_id,
                            event_type=event.event_type,
                            occupancy=event.occupancy,
                        )

                rendered = draw_counting_overlay(
                    rendered,
                    counter,
                    recent_events[-3:],
                )
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

    repository = CountRepository(args.database)
    statistics = OccupancyStatistics(repository.connection)
    assistant = OccupancyAssistant(statistics)

    if args.ask:
        print(assistant.ask(args.ask))
        repository.close()
        return 0

    if args.stats:
        summary = statistics.summarize_today()
        print(
            f"Hoy: {summary['entradas']} entradas, {summary['salidas']} salidas, "
            f"{summary['ocupacion']} personas actualmente, "
            f"{summary['ocupacion_maxima']} ocupacion maxima."
        )
        repository.close()
        return 0

    if args.export:
        exported = repository.export_csv(args.export)
        print(f"Se exportaron {exported} eventos a {args.export}")
        repository.close()
        return 0

    if args.interactive:
        print("Asistente de ocupacion. Escribe 'salir' para finalizar.")
        while True:
            try:
                question = input("Consulta: ").strip()
            except EOFError:
                break
            if question.lower() in {"salir", "exit", "q"}:
                break
            print(assistant.ask(question))
        repository.close()
        return 0

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

    if args.count and not args.track:
        print("El conteo requiere activar --track.", file=sys.stderr)
        return 7

    line_points = (
        args.line_x1,
        args.line_y1,
        args.line_x2,
        args.line_y2,
    )
    session_id = uuid.uuid4().hex
    success = process_video(
        detector,
        source,
        active_tracker,
        args.count,
        line_points,
        args.reverse_direction,
        args.confirmation,
        args.line_margin,
        repository,
        session_id,
    )
    repository.close()
    return 0 if success else 5


if __name__ == "__main__":
    raise SystemExit(main())
