from dataclasses import dataclass
from typing import Iterable

from tracking.person_tracker import PersonTrack


@dataclass(frozen=True)
class CountEvent:
    track_id: int
    event_type: str
    occupancy: int


@dataclass
class CountState:
    side: int
    pending_side: int | None = None
    pending_frames: int = 0


class LineCounter:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        line_start: tuple[float, float] = (0.0, 0.50),
        line_end: tuple[float, float] = (1.0, 0.50),
        confirmation_frames: int = 3,
        line_margin_pixels: float = 8.0,
        reverse_direction: bool = False,
    ) -> None:
        if frame_width < 1 or frame_height < 1:
            raise ValueError("Las dimensiones del fotograma deben ser positivas")
        if confirmation_frames < 1:
            raise ValueError("confirmation_frames debe ser mayor que cero")
        if line_margin_pixels < 0:
            raise ValueError("line_margin_pixels no puede ser negativo")

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.start = (
            int(round(line_start[0] * frame_width)),
            int(round(line_start[1] * frame_height)),
        )
        self.end = (
            int(round(line_end[0] * frame_width)),
            int(round(line_end[1] * frame_height)),
        )
        if self.start == self.end:
            raise ValueError("La linea debe tener dos puntos diferentes")

        self.confirmation_frames = confirmation_frames
        self.line_margin_pixels = line_margin_pixels
        self.reverse_direction = reverse_direction
        self.entries = 0
        self.exits = 0
        self.states: dict[int, CountState] = {}

    @property
    def occupancy(self) -> int:
        return max(0, self.entries - self.exits)

    def _signed_distance(self, point: tuple[int, int]) -> float:
        line_x = self.end[0] - self.start[0]
        line_y = self.end[1] - self.start[1]
        return (
            line_x * (point[1] - self.start[1])
            - line_y * (point[0] - self.start[0])
        ) / ((line_x * line_x + line_y * line_y) ** 0.5)

    def _side(self, point: tuple[int, int]) -> int:
        distance = self._signed_distance(point)
        if distance > self.line_margin_pixels:
            return 1
        if distance < -self.line_margin_pixels:
            return -1
        return 0

    def reset(self) -> None:
        self.entries = 0
        self.exits = 0
        self.states.clear()

    def update(self, tracks: Iterable[PersonTrack]) -> list[CountEvent]:
        events: list[CountEvent] = []

        active_ids = set()
        for track in tracks:
            active_ids.add(track.track_id)
            side = self._side(track.center)
            if side == 0:
                continue

            state = self.states.get(track.track_id)
            if state is None:
                self.states[track.track_id] = CountState(side=side)
                continue

            if side == state.side:
                state.pending_side = None
                state.pending_frames = 0
                continue

            if state.pending_side != side:
                state.pending_side = side
                state.pending_frames = 1
            else:
                state.pending_frames += 1
            if state.pending_frames < self.confirmation_frames:
                continue

            previous_side = state.side
            is_entry = (previous_side == -1 and side == 1)
            if self.reverse_direction:
                is_entry = not is_entry

            if is_entry:
                self.entries += 1
                event_type = "entrada"
            else:
                self.exits += 1
                event_type = "salida"

            state.side = side
            state.pending_side = None
            state.pending_frames = 0
            events.append(
                CountEvent(
                    track_id=track.track_id,
                    event_type=event_type,
                    occupancy=self.occupancy,
                )
            )

        for track_id in list(self.states):
            if track_id not in active_ids:
                state = self.states[track_id]
                state.pending_side = None
                state.pending_frames = 0

        return events
