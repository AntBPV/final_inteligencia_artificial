import pytest

from detection.person_detector import Detection, filter_person_detections


def test_filter_person_detections_applies_confidence_threshold() -> None:
    detections = [
        Detection(10, 20, 30, 80, 0.35),
        Detection(40, 20, 70, 90, 0.75),
    ]

    filtered = filter_person_detections(detections, 0.40)

    assert len(filtered) == 1
    assert filtered[0].confidence == 0.75


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_filter_person_detections_rejects_invalid_threshold(
    threshold: float,
) -> None:
    with pytest.raises(ValueError):
        filter_person_detections([], threshold)
