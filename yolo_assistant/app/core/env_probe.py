"""What this machine is, answered without torch.

Step 1 has to report the hardware and decide whether the driver gate passes
*before* anything heavy exists -- on a first run there is no managed environment
at all. So everything here reads from psutil and NVML, both of which are pure
Python over OS interfaces.

The deliberate division of labour: this module answers "what hardware and driver
are present?", and ``worker/commands/probe.py`` answers "can the installed torch
actually use it?". Only the second question can be answered from inside the
venv, and only the first can be answered before the venv exists.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass, field

from .i18n import N_, tr

try:
    import psutil
except ImportError:  # pragma: no cover - declared dependency
    psutil = None

try:
    import pynvml
except ImportError:  # pragma: no cover - declared dependency
    pynvml = None


@dataclass(frozen=True)
class CpuInfo:
    model: str = N_("Unknown CPU")
    physical_cores: int | None = None
    logical_cores: int | None = None

    def describe(self) -> str:
        model = tr(self.model)
        if self.physical_cores and self.logical_cores:
            return tr(
                "{model} - {cores} cores / {threads} threads",
                model=model,
                cores=self.physical_cores,
                threads=self.logical_cores,
            )
        if self.logical_cores:
            return tr("{model} - {threads} threads", model=model, threads=self.logical_cores)
        return model


@dataclass(frozen=True)
class MemoryInfo:
    total_gb: float = 0.0
    available_gb: float = 0.0

    def describe(self) -> str:
        if not self.total_gb:
            return tr("Unknown")
        return tr(
            "{total:.1f} GB total, {free:.1f} GB free",
            total=self.total_gb,
            free=self.available_gb,
        )


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    vram_gb: float
    compute_capability: str = N_("unknown")

    @property
    def compute_capability_value(self) -> float | None:
        """Compute capability as a number, for comparisons like ``< 7.0``."""
        try:
            major, minor = self.compute_capability.split(".")
            return float(f"{int(major)}.{int(minor)}")
        except (ValueError, AttributeError):
            return None

    def describe(self) -> str:
        return tr(
            "{name} - {vram:.1f} GB VRAM, compute {capability}",
            name=self.name,
            vram=self.vram_gb,
            capability=tr(self.compute_capability),
        )


@dataclass(frozen=True)
class DriverInfo:
    """The NVIDIA driver, and the newest CUDA it can run.

    ``cuda_version`` is *not* an installed CUDA Toolkit. It is the highest CUDA
    runtime this driver supports -- which is the number that actually decides
    which torch build will work, and the number ``nvidia-smi`` prints top right.
    """

    present: bool = False
    version: str = ""
    cuda_version: str = ""
    reason: str = ""

    @property
    def version_value(self) -> float | None:
        match = re.match(r"(\d+)\.(\d+)", self.version or "")
        return float(f"{match.group(1)}.{match.group(2)}") if match else None


@dataclass(frozen=True)
class SystemReport:
    cpu: CpuInfo = field(default_factory=CpuInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    driver: DriverInfo = field(default_factory=DriverInfo)
    gpus: list[GpuInfo] = field(default_factory=list)
    os_description: str = ""
    #: Windows only. ``None`` elsewhere, or when the setting cannot be read.
    long_paths: bool | None = None

    @property
    def primary_gpu(self) -> GpuInfo | None:
        return self.gpus[0] if self.gpus else None


def probe_system() -> SystemReport:
    """Read everything Step 1 displays. Cheap enough to call on page entry."""
    driver, gpus = _probe_nvidia()
    return SystemReport(
        cpu=_probe_cpu(),
        memory=_probe_memory(),
        driver=driver,
        gpus=gpus,
        os_description=f"{platform.system()} {platform.release()} ({platform.machine()})",
        long_paths=windows_long_paths_enabled(),
    )


# -- CPU / RAM ------------------------------------------------------------- #


def _probe_cpu() -> CpuInfo:
    physical = logical = None
    if psutil is not None:
        try:
            physical = psutil.cpu_count(logical=False)
            logical = psutil.cpu_count(logical=True)
        except Exception:  # noqa: BLE001
            pass
    return CpuInfo(model=_cpu_model(), physical_cores=physical, logical_cores=logical or os.cpu_count())


def _cpu_model() -> str:
    """A human-readable CPU name.

    ``platform.processor()`` is no help on either target: on Linux it is
    ``"x86_64"``, and on Windows it is ``PROCESSOR_IDENTIFIER`` -- "AMD64 Family
    26 Model 68 Stepping 0, AuthenticAMD" rather than "AMD Ryzen 9 9950X". The
    marketing name lives in ``/proc/cpuinfo`` and in the registry respectively,
    and the platform string is only the last resort.
    """
    if os.name == "nt":
        name = _windows_cpu_name()
        if name:
            return name
    if os.name == "posix":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith(("model name", "Model")):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    name = platform.processor() or platform.machine()
    return name or N_("Unknown CPU")


def _windows_cpu_name() -> str:
    """The CPU's brand string, as Task Manager shows it, or "" if unreadable."""
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            value, _kind = winreg.QueryValueEx(key, "ProcessorNameString")
    except (OSError, ImportError):
        return ""
    # Intel pads the brand string with leading spaces, AMD with trailing ones.
    return " ".join(str(value).split())


def _probe_memory() -> MemoryInfo:
    if psutil is None:
        return MemoryInfo()
    try:
        virtual = psutil.virtual_memory()
    except Exception:  # noqa: BLE001
        return MemoryInfo()
    return MemoryInfo(
        total_gb=virtual.total / 1024**3,
        available_gb=virtual.available / 1024**3,
    )


# -- NVIDIA ---------------------------------------------------------------- #


def _probe_nvidia() -> tuple[DriverInfo, list[GpuInfo]]:
    if pynvml is None:  # pragma: no cover - declared dependency
        return DriverInfo(reason=tr("nvidia-ml-py is not installed.")), []
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 - NVML raises its own exception types
        return DriverInfo(reason=_no_driver_reason(exc)), []

    try:
        driver_version = _text(pynvml.nvmlSystemGetDriverVersion())
        try:
            raw = pynvml.nvmlSystemGetCudaDriverVersion()
            cuda_version = f"{raw // 1000}.{(raw % 1000) // 10}"
        except Exception:  # noqa: BLE001
            cuda_version = ""

        gpus: list[GpuInfo] = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                capability = f"{major}.{minor}"
            except Exception:  # noqa: BLE001
                capability = N_("unknown")
            gpus.append(
                GpuInfo(
                    index=index,
                    name=_text(pynvml.nvmlDeviceGetName(handle)),
                    vram_gb=memory.total / 1024**3,
                    compute_capability=capability,
                )
            )
    except Exception as exc:  # noqa: BLE001
        return DriverInfo(reason=tr("NVML query failed: {error}", error=exc)), []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001, S110
            pass

    if not gpus:
        return (
            DriverInfo(
                present=True,
                version=driver_version,
                cuda_version=cuda_version,
                reason=tr("The driver is installed but reports no NVIDIA GPU."),
            ),
            [],
        )
    return DriverInfo(present=True, version=driver_version, cuda_version=cuda_version), gpus


def _no_driver_reason(exc: BaseException) -> str:
    detail = str(exc).strip() or type(exc).__name__
    return tr("No usable NVIDIA driver was found ({detail}).", detail=detail)


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def windows_long_paths_enabled() -> bool | None:
    """Whether Windows accepts paths longer than 260 characters.

    Returns ``None`` on every other platform, and on Windows when the setting
    cannot be read. This matters because a CUDA-enabled PyTorch install nests
    deeply -- ``venvs/<fingerprint>/Lib/site-packages/nvidia/<library>/...`` --
    and with long paths disabled the install fails partway through with a
    file-not-found error that says nothing about path length.
    """
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem"
        ) as key:
            value, _kind = winreg.QueryValueEx(key, "LongPathsEnabled")
        return bool(value)
    except (OSError, ImportError, ValueError):
        return None


LONG_PATHS_WARNING = N_(
    "Windows long path support is turned off. Installing PyTorch creates files with very "
    "long names and the install may fail partway through. Enable it with the Group Policy "
    'setting "Enable Win32 long paths", or set LongPathsEnabled to 1 under '
    "HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\FileSystem."
)


def nvidia_smi_query() -> str:
    """Raw ``nvidia-smi`` output, for the support-detail view only.

    NVML is the source of truth everywhere else; this exists because a support
    request is far easier to answer with the user's own ``nvidia-smi`` in front
    of you.
    """
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            # A console window flashing up from the installed, windowed app.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return tr("nvidia-smi could not be run: {error}", error=exc)
    return completed.stdout or completed.stderr
