"""GPU telemetry, sampled by the GUI rather than reported by the worker.

Deliberate design choice: the gauges stay alive while the worker is importing
torch, downloading a model, or wedged inside a native call -- which is exactly
when the user is staring at them. It also means one panel serves autolabeling,
training and inference identically, and telemetry never touches the message
protocol.

``nvidia-ml-py`` is pure ctypes over ``nvml.dll`` / ``libnvidia-ml.so.1``, so
none of this needs torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QThread, Signal

from .i18n import tr

try:
    import pynvml
except ImportError:  # pragma: no cover - dependency is declared
    pynvml = None


@dataclass(frozen=True)
class GpuSample:
    """One reading. ``None`` means "this GPU cannot report that field"."""

    index: int
    name: str
    utilisation_pct: float | None = None
    memory_used_gb: float | None = None
    memory_total_gb: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    temperature_c: float | None = None
    unsupported: frozenset[str] = field(default_factory=frozenset)

    @property
    def memory_fraction(self) -> float:
        if not self.memory_total_gb:
            return 0.0
        return min(1.0, (self.memory_used_gb or 0.0) / self.memory_total_gb)

    @property
    def power_fraction(self) -> float:
        if not self.power_limit_w:
            return 0.0
        return min(1.0, (self.power_w or 0.0) / self.power_limit_w)


@dataclass(frozen=True)
class GpuStatic:
    """Facts that do not change between samples."""

    index: int
    name: str
    memory_total_gb: float
    driver_version: str
    cuda_driver_version: str
    compute_capability: str


class TelemetrySampler(QThread):
    """Polls NVML on its own thread and emits samples.

    Power and temperature are genuinely unavailable on some laptop and vGPU
    parts, so every field is probed independently and a failure disables just
    that field rather than the whole panel.
    """

    sampled = Signal(object)        # GpuSample
    unavailable = Signal(str)       # human-readable reason
    devices_discovered = Signal(list)  # list[GpuStatic]

    def __init__(self, interval_ms: int = 500, parent=None) -> None:
        super().__init__(parent)
        self._interval_ms = interval_ms
        self._device_index = 0
        self._stop = False
        self._unsupported: set[str] = set()

    def set_device(self, index: int) -> None:
        self._device_index = index

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: D102 - QThread entry point
        if pynvml is None:
            self.unavailable.emit(tr("nvidia-ml-py is not installed."))
            return
        try:
            pynvml.nvmlInit()
        except Exception as exc:  # noqa: BLE001 - NVML raises its own types
            self.unavailable.emit(tr("No NVIDIA driver detected ({error}).", error=_short(exc)))
            return

        try:
            statics = self._discover()
            if not statics:
                self.unavailable.emit(tr("No NVIDIA GPU found."))
                return
            self.devices_discovered.emit(statics)

            while not self._stop:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
                    self.sampled.emit(self._read(handle, statics))
                except Exception as exc:  # noqa: BLE001
                    self.unavailable.emit(
                        tr("Telemetry read failed ({error}).", error=_short(exc))
                    )
                    return
                self.msleep(self._interval_ms)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001, S110
                pass

    # -- internals --------------------------------------------------------- #

    def _discover(self) -> list[GpuStatic]:
        driver = _decode(pynvml.nvmlSystemGetDriverVersion())
        try:
            raw = pynvml.nvmlSystemGetCudaDriverVersion()
            cuda = f"{raw // 1000}.{(raw % 1000) // 10}"
        except Exception:  # noqa: BLE001
            cuda = tr("unknown")

        statics: list[GpuStatic] = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                capability = f"{major}.{minor}"
            except Exception:  # noqa: BLE001
                capability = tr("unknown")
            statics.append(
                GpuStatic(
                    index=index,
                    name=_decode(pynvml.nvmlDeviceGetName(handle)),
                    memory_total_gb=round(mem.total / 1024**3, 2),
                    driver_version=driver,
                    cuda_driver_version=cuda,
                    compute_capability=capability,
                )
            )
        return statics

    def _read(self, handle, statics: list[GpuStatic]) -> GpuSample:
        static = statics[min(self._device_index, len(statics) - 1)]
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return GpuSample(
            index=self._device_index,
            name=static.name,
            utilisation_pct=self._field(
                "utilisation", lambda: float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            ),
            memory_used_gb=round(mem.used / 1024**3, 2),
            memory_total_gb=round(mem.total / 1024**3, 2),
            power_w=self._field(
                "power", lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            ),
            power_limit_w=self._field(
                "power_limit",
                lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0,
            ),
            temperature_c=self._field(
                "temperature",
                lambda: float(
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                ),
            ),
            unsupported=frozenset(self._unsupported),
        )

    def _field(self, name: str, read):
        """Read one field, remembering permanent failures so we stop retrying."""
        if name in self._unsupported:
            return None
        try:
            return read()
        except Exception:  # noqa: BLE001
            self._unsupported.add(name)
            return None


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _short(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or type(exc).__name__
