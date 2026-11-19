import cv2
import numpy as np

from tracking.person_tracker import PersonTrack, PersonTracker


TRACK_COLORS = (
    (0, 255, 0),
    (255, 180, 0),
    (0, 180, 255),
    (255, 0, 255),
)


def draw_tracks(
    frame: np.ndarray,
    tracks: list[PersonTrack],
    tracker: PersonTracker,
    show_confidence: bool = True,
) -> np.ndarray:
    rendered = frame.copy()

    for track in tracks:
        color = TRACK_COLORS[track.track_id % len(TRACK_COLORS)]
        cv2.rectangle(
            rendered,
            (track.x1, track.y1),
            (track.x2, track.y2),
            color,
            2,
        )

        label = f"ID {track.track_id}"
        if show_confidence:
            if not track.lost:
                label += f" person {track.confidence:.2f}"

        text_x = max(track.x1, 0)
        text_y = max(track.y1 - 8, 18)
        cv2.putText(
            rendered,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        trajectory = tracker.get_trajectory(track.track_id)
        for point_a, point_b in zip(trajectory, trajectory[1:]):
            cv2.line(rendered, point_a, point_b, color, 2)

    cv2.putText(
        rendered,
        f"Personas seguidas: {len(tracks)}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return rendered
