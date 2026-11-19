from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PersonTrack:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    lost: bool = False

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class PersonTracker:
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.40,
        device: str = "cpu",
        tracker: str = "tracking/bytetrack_stable.yaml",
        trajectory_length: int = 30,
        lost_grace_frames: int = 15,
        imgsz: int = 640,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence debe estar entre 0 y 1")
        if trajectory_length < 1:
            raise ValueError("trajectory_length debe ser mayor que cero")
        if lost_grace_frames < 0:
            raise ValueError("lost_grace_frames no puede ser negativo")
        if imgsz < 160:
            raise ValueError("imgsz debe ser al menos 160")

        from ultralytics import YOLO

        self.confidence = confidence
        self.device = device
        self.tracker = tracker
        self.trajectory_length = trajectory_length
        self.imgsz = imgsz
        self.model = YOLO(model_name)
        self.trajectories: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.trajectory_length)
        )
        self.lost_grace_frames = lost_grace_frames
        self._frame_shape: tuple[int, int] | None = None

    @staticmethod
    def _iou(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = max(1, first[2] - first[0]) * max(1, first[3] - first[1])
        second_area = max(1, second[2] - second[0]) * max(1, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union

    def update(self, frame: np.ndarray) -> list[PersonTrack]:
        if frame is None or frame.size == 0:
            return []
        self._frame_shape = frame.shape[:2]

        results = self.model.track(
            source=frame,
            conf=min(self.confidence, 0.05),
            classes=[0],
            device=self.device,
            tracker=self.tracker,
            persist=True,
            imgsz=self.imgsz,
            verbose=False,
        )

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []

        tracks: list[PersonTrack] = []
        ids = boxes.id.detach().cpu().numpy()
        coordinates = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()

        height, width = frame.shape[:2]
        candidates = [
            (float(raw_confidence), int(raw_id), raw_coordinates)
            for raw_id, raw_coordinates, raw_confidence in zip(
                ids,
                coordinates,
                confidences,
            )
            if float(raw_confidence) >= self.confidence
        ]

        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        unique_candidates: list[tuple[float, int, np.ndarray]] = []
        for candidate in candidates:
            candidate_confidence, candidate_id, candidate_coordinates = candidate
            candidate_box = tuple(
                int(round(value))
                for value in candidate_coordinates
            )
            if any(
                self._iou(candidate_box, tuple(
                    int(round(value))
                    for value in selected_coordinates
                )) >= 0.50
                for _, _, selected_coordinates in unique_candidates
            ):
                continue
            unique_candidates.append(candidate)

        for raw_confidence, raw_id, raw_coordinates in unique_candidates:
            track_id = int(raw_id)
            x1 = max(0, min(width - 1, int(round(raw_coordinates[0]))))
            y1 = max(0, min(height - 1, int(round(raw_coordinates[1]))))
            x2 = max(0, min(width - 1, int(round(raw_coordinates[2]))))
            y2 = max(0, min(height - 1, int(round(raw_coordinates[3]))))
            track = PersonTrack(
                track_id=track_id,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(raw_confidence),
            )
            tracks.append(track)
            self.trajectories[track_id].append(track.center)

        return tracks

    def get_trajectory(
        self,
        track_id: int,
    ) -> list[tuple[int, int]]:
        return list(self.trajectories.get(track_id, ()))

    def reset(self) -> None:
        self.trajectories.clear()
