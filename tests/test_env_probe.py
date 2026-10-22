"""Hardware probing, against a stand-in for NVML.

Step 1 must report something sensible on a machine with no NVIDIA driver at
all -- that is the case the screen exists to explain -- so none of this may
raise, and every failure has to turn into a sentence rather than an exception.
"""

import types

import pytest

from yolo_assistant.app.core import env_probe
from yolo_assistant.app.core.env_probe import DriverInfo, GpuInfo


class FakeNvml:
    """The handful of NVML calls this module makes."""

    NVML_TEMPERATURE_GPU = 0

    def __init__(self, *, init_error=None, devices=1, capability=(12, 0), name=b"RTX 5070 Ti"):
        self.init_error = init_error
        self.devices = devices
        self.capability = capability
        self.name = name
        self.shutdown_called = False

    def nvmlInit(self):
        if self.init_error:
            raise self.init_error

    def nvmlShutdown(self):
        self.shutdown_called = True

    def nvmlSystemGetDriverVersion(self):
        return b"595.84"

    def nvmlSystemGetCudaDriverVersion(self):
        return 13020

    def nvmlDeviceGetCount(self):
        return self.devices

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetMemoryInfo(self, _handle):
        return types.SimpleNamespace(total=16 * 1024**3, used=0)

    def nvmlDeviceGetCudaComputeCapability(self, _handle):
        if self.capability is None:
            raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")
        return self.capability

    def nvmlDeviceGetName(self, _handle):
        return self.name


@pytest.fixture
def nvml(monkeypatch):
    def install(fake):
        monkeypatch.setattr(env_probe, "pynvml", fake)
        return fake

    return install


def test_a_working_driver_is_reported_in_full(nvml):
    nvml(FakeNvml())
    report = env_probe.probe_system()

    assert report.driver.present
    assert report.driver.version == "595.84"
    assert report.driver.cuda_version == "13.2"
    assert len(report.gpus) == 1
    assert report.primary_gpu.name == "RTX 5070 Ti"
    assert report.primary_gpu.vram_gb == pytest.approx(16.0)
    assert report.primary_gpu.compute_capability_value == 12.0


def test_nvml_is_always_shut_down(nvml):
    fake = nvml(FakeNvml())
    env_probe.probe_system()
    assert fake.shutdown_called, "NVML was left initialised"


def test_no_driver_produces_a_sentence_not_an_exception(nvml):
    nvml(FakeNvml(init_error=RuntimeError("NVML_ERROR_DRIVER_NOT_LOADED")))
    report = env_probe.probe_system()

    assert not report.driver.present
    assert report.gpus == []
    assert "NVML_ERROR_DRIVER_NOT_LOADED" in report.driver.reason
    # The rest of the machine is still described, because it is still useful.
    assert report.cpu.model and report.memory.total_gb > 0


def test_a_driver_with_no_gpu_is_distinguished_from_no_driver(nvml):
    """An eGPU that has been unplugged, or a VM with the driver but no device."""
    nvml(FakeNvml(devices=0))
    report = env_probe.probe_system()

    assert report.driver.present
    assert report.gpus == []
    assert "no NVIDIA GPU" in report.driver.reason


def test_an_unreadable_compute_capability_does_not_lose_the_gpu(nvml):
    nvml(FakeNvml(capability=None))
    report = env_probe.probe_system()

    assert report.primary_gpu.compute_capability == "unknown"
    assert report.primary_gpu.compute_capability_value is None
    assert report.driver.present


def test_nvml_strings_may_be_bytes_or_str(nvml):
    nvml(FakeNvml(name="RTX 4090"))
    assert env_probe.probe_system().primary_gpu.name == "RTX 4090"


def test_missing_nvidia_ml_py_is_survivable(monkeypatch):
    monkeypatch.setattr(env_probe, "pynvml", None)
    report = env_probe.probe_system()
    assert not report.driver.present
    assert "nvidia-ml-py" in report.driver.reason


@pytest.mark.parametrize(
    "version,expected",
    [("595.84", 595.84), ("535.104.05", 535.104), ("", None), ("unknown", None)],
)
def test_driver_versions_parse_for_comparison(version, expected):
    assert DriverInfo(version=version).version_value == expected


@pytest.mark.parametrize("capability", ["", "unknown", "12"])
def test_unparseable_compute_capabilities_are_none(capability):
    assert GpuInfo(0, "x", 8.0, capability).compute_capability_value is None


def test_descriptions_are_written_for_the_hardware_panel(nvml):
    nvml(FakeNvml())
    report = env_probe.probe_system()
    assert "threads" in report.cpu.describe()
    assert "GB total" in report.memory.describe()
    assert "VRAM" in report.primary_gpu.describe() and "compute" in report.primary_gpu.describe()


def test_the_cpu_is_named_rather_than_left_as_the_architecture():
    """platform.processor() returns the useless string "x86_64" on Linux."""
    model = env_probe._cpu_model()
    assert model and model not in ("x86_64", "AMD64")
    # On Windows, platform.processor() is PROCESSOR_IDENTIFIER.
    assert "Family" not in model and "Stepping" not in model


@pytest.mark.skipif(env_probe.os.name != "nt", reason="reads the Windows registry")
def test_windows_reports_the_brand_string_trimmed():
    name = env_probe._windows_cpu_name()
    assert name and name == name.strip() and "  " not in name
