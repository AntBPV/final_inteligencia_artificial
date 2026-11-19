import numpy as np

from detection.person_detector import Detection
from visualization.renderer import draw_person_detections
from visualization.tracking_renderer import draw_tracks
from tracking.person_tracker import PersonTrack, PersonTracker


def test_renderer_returns_copy_and_draws_detection() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detection = Detection(20, 30, 100, 180, 0.82)

    rendered = draw_person_detections(frame, [detection])

    assert rendered is not frame
    assert rendered.shape == frame.shape
    assert np.any(rendered != frame)


def test_tracking_renderer_draws_track_without_mutating_frame() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    track = PersonTrack(3, 20, 30, 100, 180, 0.82)
    tracker = object.__new__(PersonTracker)

    from collections import defaultdict, deque

    tracker.trajectory_length = 30
    tracker.trajectories = defaultdict(
        lambda: deque(maxlen=tracker.trajectory_length)
    )
    tracker.trajectories[3].append(track.center)

    rendered = draw_tracks(frame, [track], tracker)

    assert rendered is not frame
    assert rendered.shape == frame.shape
    assert np.any(rendered != frame)
