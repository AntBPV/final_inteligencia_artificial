import pytest

from device_support import resolve_device


def test_auto_or_cpu_resolves_without_gpu() -> None:
    assert resolve_device("cpu") == "cpu"


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_device("metal")
