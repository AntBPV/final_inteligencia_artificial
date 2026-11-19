import importlib.util
import platform
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceStatus:
    value: str
    available: bool
    description: str


def get_device_statuses() -> list[DeviceStatus]:
    cuda_available = torch.cuda.is_available()
    cuda_name = torch.cuda.get_device_name(0) if cuda_available else None
    dml_available = importlib.util.find_spec("torch_directml") is not None
    openvino_available = importlib.util.find_spec("openvino") is not None

    statuses = [
        DeviceStatus(
            "cpu",
            True,
            "CPU (dispositivo compatible y actualmente activo)",
        ),
        DeviceStatus(
            "auto",
            cuda_available,
            "CUDA disponible" if cuda_available else "Sin CUDA; auto seleccionara CPU",
        ),
        DeviceStatus(
            "cuda",
            cuda_available,
            cuda_name or "CUDA no disponible",
        ),
        DeviceStatus(
            "dml",
            dml_available,
            "torch-directml instalado" if dml_available else "torch-directml no instalado",
        ),
        DeviceStatus(
            "openvino",
            openvino_available,
            "OpenVINO instalado" if openvino_available else "OpenVINO no instalado",
        ),
    ]
    return statuses


def resolve_device(requested: str):
    normalized = requested.strip().lower()
    if normalized in {"", "auto"}:
        if torch.cuda.is_available():
            return "0"
        return "cpu"
    if normalized in {"cpu", "cuda", "0"}:
        if normalized == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA fue solicitado, pero PyTorch no detecta una GPU NVIDIA.")
        return "0" if normalized == "cuda" else normalized
    if normalized == "dml":
        if importlib.util.find_spec("torch_directml") is None:
            raise RuntimeError(
                "DirectML no está instalado. Se requiere un entorno compatible "
                "como Python 3.11/3.12 y torch-directml."
            )
        try:
            import torch_directml
            return torch_directml.device()
        except Exception as error:
            raise RuntimeError(f"No se pudo inicializar DirectML: {error}") from error
    if normalized == "openvino":
        if importlib.util.find_spec("openvino") is None:
            raise RuntimeError(
                "OpenVINO no está instalado. Puede instalarse con "
                "`python -m pip install openvino`."
            )
        return "openvino"
    raise ValueError(
        "Dispositivo no reconocido. Use cpu, auto, cuda, dml o openvino."
    )


def print_device_status() -> None:
    print(f"Sistema: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    for status in get_device_statuses():
        marker = "disponible" if status.available else "no disponible"
        print(f"- {status.value}: {marker} - {status.description}")
