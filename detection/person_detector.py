from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class PersonDetector:
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.40,
        device: str = "cpu",
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence debe estar entre 0 y 1")

        from ultralytics import YOLO

        self.confidence = confidence
        self.model_name = Path(model_name).as_posix()
        self.model = YOLO(self.model_name)
        self.device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=[0],
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        detections: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            confidence = float(box.conf.item())
            coordinates = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = (int(round(value)) for value in coordinates)
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                )
            )

        return detections


def filter_person_detections(
    detections: Iterable[Detection],
    minimum_confidence: float,
) -> list[Detection]:
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence debe estar entre 0 y 1")

    return [
        detection
        for detection in detections
        if detection.confidence >= minimum_confidence
    ]
