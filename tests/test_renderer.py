import numpy as np

from detection.person_detector import Detection
from visualization.renderer import draw_person_detections


def test_renderer_returns_copy_and_draws_detection() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detection = Detection(20, 30, 100, 180, 0.82)

    rendered = draw_person_detections(frame, [detection])

    assert rendered is not frame
    assert rendered.shape == frame.shape
    assert np.any(rendered != frame)
