from counting.line_counter import LineCounter
from tracking.person_tracker import PersonTrack


def track(track_id: int, center_x: int, center_y: int) -> PersonTrack:
    half_width = 20
    half_height = 40
    return PersonTrack(
        track_id=track_id,
        x1=center_x - half_width,
        y1=center_y - half_height,
        x2=center_x + half_width,
        y2=center_y + half_height,
        confidence=0.90,
    )


def test_horizontal_entry_and_exit() -> None:
    counter = LineCounter(
        frame_width=200,
        frame_height=100,
        confirmation_frames=1,
        line_margin_pixels=0,
    )

    counter.update([track(1, 100, 20)])
    entry_events = counter.update([track(1, 100, 80)])

    assert counter.entries == 1
    assert counter.exits == 0
    assert counter.occupancy == 1
    assert entry_events[0].event_type == "entrada"

    exit_events = counter.update([track(1, 100, 20)])

    assert counter.entries == 1
    assert counter.exits == 1
    assert counter.occupancy == 0
    assert exit_events[0].event_type == "salida"


def test_cross_requires_confirmation_frames() -> None:
    counter = LineCounter(
        frame_width=200,
        frame_height=100,
        confirmation_frames=3,
        line_margin_pixels=0,
    )

    counter.update([track(1, 100, 20)])
    counter.update([track(1, 100, 80)])
    assert counter.entries == 0

    counter.update([track(1, 100, 80)])
    assert counter.entries == 0

    counter.update([track(1, 100, 82)])
    assert counter.entries == 1


def test_track_cannot_count_twice_on_same_side() -> None:
    counter = LineCounter(
        frame_width=200,
        frame_height=100,
        confirmation_frames=1,
        line_margin_pixels=0,
    )

    counter.update([track(1, 100, 20)])
    counter.update([track(1, 100, 80)])
    counter.update([track(1, 100, 82)])

    assert counter.entries == 1


def test_reverse_direction_swaps_entry_and_exit() -> None:
    counter = LineCounter(
        frame_width=200,
        frame_height=100,
        confirmation_frames=1,
        line_margin_pixels=0,
        reverse_direction=True,
    )

    counter.update([track(1, 100, 20)])
    events = counter.update([track(1, 100, 80)])

    assert counter.entries == 0
    assert counter.exits == 1
    assert events[0].event_type == "salida"


def test_reset_clears_totals_and_states() -> None:
    counter = LineCounter(
        frame_width=200,
        frame_height=100,
        confirmation_frames=1,
        line_margin_pixels=0,
    )
    counter.update([track(1, 100, 20)])
    counter.update([track(1, 100, 80)])

    counter.reset()

    assert counter.entries == 0
    assert counter.exits == 0
    assert counter.occupancy == 0
    assert counter.states == {}
