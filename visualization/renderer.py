import cv2
import numpy as np

from detection.person_detector import Detection


def draw_person_detections(
    frame: np.ndarray,
    detections: list[Detection],
    show_confidence: bool = True,
) -> np.ndarray:
    rendered = frame.copy()

    for detection in detections:
        cv2.rectangle(
            rendered,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            (0, 255, 0),
            2,
        )

        label = "person"
        if show_confidence:
            label += f" {detection.confidence:.2f}"

        text_x = max(detection.x1, 0)
        text_y = max(detection.y1 - 8, 18)
        cv2.putText(
            rendered,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        rendered,
        f"Personas detectadas: {len(detections)}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return rendered
