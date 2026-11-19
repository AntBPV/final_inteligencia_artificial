import numpy as np

from tracking.person_tracker import PersonTrack, PersonTracker


def test_person_track_center_and_box() -> None:
    track = PersonTrack(
        track_id=7,
        x1=20,
        y1=40,
        x2=60,
        y2=100,
        confidence=0.90,
    )

    assert track.box == (20, 40, 60, 100)
    assert track.center == (40, 70)


def test_tracker_rejects_invalid_configuration() -> None:
    try:
        PersonTracker(confidence=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("Se esperaba ValueError")


def test_tracker_trajectory_is_bounded_and_resettable() -> None:
    tracker = object.__new__(PersonTracker)
    tracker.trajectory_length = 3
    tracker.trajectories = {}

    from collections import defaultdict, deque

    tracker.trajectories = defaultdict(
        lambda: deque(maxlen=tracker.trajectory_length)
    )
    for point in ((10, 10), (20, 20), (30, 30), (40, 40)):
        tracker.trajectories[2].append(point)

    assert tracker.get_trajectory(2) == [(20, 20), (30, 30), (40, 40)]

    tracker.reset()
    assert tracker.get_trajectory(2) == []


def test_iou_matches_overlapping_boxes() -> None:
    iou = PersonTracker._iou(
        (10, 10, 50, 50),
        (20, 20, 60, 60),
    )

    assert 0.3 < iou < 0.4


def test_iou_rejects_mostly_overlapping_boxes() -> None:
    iou = PersonTracker._iou(
        (10, 10, 100, 100),
        (12, 12, 98, 98),
    )

    assert iou >= 0.85
