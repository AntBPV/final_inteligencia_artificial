import cv2
import numpy as np

from counting.line_counter import LineCounter


def draw_counting_overlay(
    frame: np.ndarray,
    counter: LineCounter,
    recent_events: list[str] | None = None,
) -> np.ndarray:
    rendered = frame.copy()
    cv2.line(
        rendered,
        counter.start,
        counter.end,
        (0, 255, 255),
        3,
    )

    cv2.putText(
        rendered,
        f"Entradas: {counter.entries}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        rendered,
        f"Salidas: {counter.exits}",
        (12, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        rendered,
        f"Ocupacion: {counter.occupancy}",
        (12, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for index, event in enumerate(recent_events or (), start=1):
        cv2.putText(
            rendered,
            event,
            (12, 114 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return rendered
