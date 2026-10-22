"""Decide which CUDA build of PyTorch to install.

This is the single most consequential decision the app makes, and the one a
beginner has no way to make correctly. Get it wrong and nothing fails at install
time: the environment builds cleanly, the app reports success, and then the
first training step dies with ``no kernel image is available for execution``.

Three independent constraints have to hold at once, and each is checked here:

1. **The driver must support the CUDA runtime.** A CUDA 13.x build needs an
   r580 driver; CUDA 12.x builds run on anything from r525 thanks to CUDA's
   minor-version compatibility.
2. **The torch build must contain kernels for this GPU.** Newer builds drop old
   architectures -- the cu128 and later builds start at compute 7.5, so a
   Pascal card needs cu126 even though its driver may be brand new. This is the
   constraint ``uv pip install --torch-backend=auto`` does not check, which is
   why the app makes its own decision.
3. **A wheel must actually exist** for this backend, Python version and
   platform. Backends are not interchangeable here: at the time of writing
   cu128 stops at torch 2.11 while cu126, cu130 and cu132 go to 2.14.

The candidate list is a pinned table rather than a version string formatted into
a URL, because the obvious guess is wrong: there is no ``cu131`` index -- it
returns 403 -- and a driver reporting CUDA 13.1 would be sent straight to it.
"""

from __future__ import annotations

import os
import platform
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ..env_probe import SystemReport
from ..i18n import N_, tr

TORCH_INDEX_BASE = "https://download.pytorch.org/whl"

#: The Python we install into the managed venv. Pinned, not inherited: it is the
#: version with full wheel coverage across every index below, and pinning it
#: means the wheel-availability check has a single answer rather than one per
#: user machine.
MANAGED_PYTHON = "3.12"
PYTHON_TAG = "cp312"


@dataclass(frozen=True)
class Backend:
    """One installable PyTorch flavour."""

    key: str
    label: str
    cuda: str
    #: Lowest driver that can run this CUDA runtime, per platform.
    min_driver_linux: float
    min_driver_windows: float
    #: Lowest GPU compute capability this build ships kernels for.
    min_compute: float
    #: Version floor. The index itself carries only compatible versions, so the
    #: newest build on it is the right one; the floor only rules out ancient
    #: releases that predate features we rely on.
    torch_spec: str

    @property
    def index_url(self) -> str:
        return f"{TORCH_INDEX_BASE}/{self.key}"

    @property
    def is_cuda(self) -> bool:
        return self.key != "cpu"

    def min_driver(self, windows: bool | None = None) -> float:
        if windows is None:
            windows = os.name == "nt"
        return self.min_driver_windows if windows else self.min_driver_linux


#: Verified against the live indexes, newest first. Adding an entry means
#: checking that the index exists *and* that it carries a cp312 wheel.
BACKENDS: tuple[Backend, ...] = (
    Backend("cu132", "CUDA 13.2", "13.2", 580.65, 580.88, 7.5, "torch>=2.12"),
    Backend("cu130", "CUDA 13.0", "13.0", 580.65, 580.88, 7.5, "torch>=2.9"),
    Backend("cu129", "CUDA 12.9", "12.9", 525.60, 527.41, 7.5, "torch>=2.8"),
    Backend("cu128", "CUDA 12.8", "12.8", 525.60, 527.41, 7.5, "torch>=2.7"),
    Backend("cu126", "CUDA 12.6", "12.6", 525.60, 527.41, 5.0, "torch>=2.6"),
    Backend("cpu", N_("CPU only"), "", 0.0, 0.0, 0.0, "torch>=2.6"),
)

BY_KEY = {backend.key: backend for backend in BACKENDS}
CPU = BY_KEY["cpu"]

#: Compute capabilities below this get an explicit warning: they work, but they
#: are outside what current Ultralytics releases are tested against.
LEGACY_COMPUTE = 7.0


@dataclass(frozen=True)
class BackendPlan:
    """The chosen backend and, in plain English, why."""

    backend: Backend
    reason: str
    warnings: list[str] = field(default_factory=list)
    #: Backends the user could pick instead, best first. Offered in the UI so a
    #: wrong automatic choice is recoverable without editing anything.
    alternatives: list[Backend] = field(default_factory=list)

    @property
    def is_cpu_fallback(self) -> bool:
        return not self.backend.is_cuda


def usable_backends(report: SystemReport, *, windows: bool | None = None) -> list[Backend]:
    """CUDA backends this machine's driver and GPU can both actually run."""
    if not report.driver.present or not report.gpus:
        return []

    driver_value = report.driver.version_value
    # The weakest GPU decides: torch loads one build for every device, so a
    # second, older card would be unusable with a build chosen for the first.
    computes = [gpu.compute_capability_value for gpu in report.gpus]
    known = [value for value in computes if value is not None]
    compute = min(known) if known else None

    usable = []
    for backend in BACKENDS:
        if not backend.is_cuda:
            continue
        if driver_value is not None and driver_value < backend.min_driver(windows):
            continue
        if compute is not None and compute < backend.min_compute:
            continue
        usable.append(backend)
    return usable


def plan_backend(
    report: SystemReport,
    *,
    preferred: str | None = None,
    availability=None,
    windows: bool | None = None,
) -> BackendPlan:
    """Choose a backend for ``report``.

    ``preferred`` is a backend key the user picked explicitly; it is honoured
    even when it looks wrong, with a warning, because the table here can be out
    of date and the user's machine cannot.

    ``availability`` is an optional ``callable(Backend) -> bool | None`` used to
    confirm a wheel exists before committing to a multi-gigabyte download. It
    returns ``None`` when the network could not answer, which is treated as "do
    not step down" -- a proxy failure must not silently downgrade someone to
    CPU.
    """
    warnings: list[str] = []
    usable = usable_backends(report, windows=windows)

    if preferred and preferred in BY_KEY:
        backend = BY_KEY[preferred]
        if backend.is_cuda and usable and backend not in usable:
            warnings.append(
                tr(
                    "{backend} was selected manually, but this machine's driver or GPU "
                    "may not support it. If training fails with a kernel error, choose "
                    "{alternative} instead.",
                    backend=tr(backend.label),
                    alternative=tr(usable[0].label),
                )
            )
        return BackendPlan(
            backend=backend,
            reason=tr("{backend} was selected manually.", backend=tr(backend.label)),
            warnings=warnings,
            alternatives=[other for other in BACKENDS if other is not backend],
        )

    if not report.driver.present:
        return BackendPlan(
            backend=CPU,
            reason=report.driver.reason or tr("No NVIDIA driver was found."),
            warnings=[
                tr(
                    "Without an NVIDIA GPU, autolabeling and training will be too slow to be "
                    "practical. Install an NVIDIA driver and run this step again."
                )
            ],
            alternatives=list(BACKENDS[:-1]),
        )

    if not usable:
        gpu = report.primary_gpu
        if gpu:
            reason = tr(
                "No CUDA build of PyTorch supports this GPU ({gpu}, compute {capability}).",
                gpu=gpu.name,
                capability=tr(gpu.compute_capability),
            )
        else:
            reason = tr("No CUDA build of PyTorch supports this GPU.")
        return BackendPlan(
            backend=CPU,
            reason=reason,
            warnings=[
                tr(
                    "The GPU is too old for current PyTorch releases, or the driver is older "
                    "than any supported CUDA runtime. Updating the NVIDIA driver may help."
                )
            ],
            alternatives=list(BACKENDS[:-1]),
        )

    chosen = usable[0]
    if availability is not None:
        for candidate in usable:
            verdict = availability(candidate)
            if verdict is False:
                warnings.append(
                    tr(
                        "No {backend} package is published for Python {python}.",
                        backend=tr(candidate.label),
                        python=MANAGED_PYTHON,
                    )
                )
                continue
            chosen = candidate
            break
        else:
            return BackendPlan(
                backend=CPU,
                reason=tr("No CUDA package could be found for this machine."),
                warnings=warnings,
                alternatives=list(usable),
            )

    gpu = report.primary_gpu
    compute = gpu.compute_capability_value if gpu else None
    if compute is not None and compute < LEGACY_COMPUTE:
        warnings.append(
            tr(
                "This GPU (compute {capability}) is older than the hardware current "
                "releases are tested on. Training will work but may be slow, and mixed precision "
                "is unlikely to help.",
                capability=gpu.compute_capability,
            )
        )

    reason = tr(
        "Driver {driver} supports CUDA {cuda} and this GPU is compute {capability}, "
        "so {backend} is the newest usable build.",
        driver=report.driver.version,
        cuda=report.driver.cuda_version or tr("unknown"),
        capability=tr(gpu.compute_capability) if gpu else tr("unknown"),
        backend=tr(chosen.label),
    )
    return BackendPlan(
        backend=chosen,
        reason=reason,
        warnings=warnings,
        alternatives=[other for other in BACKENDS if other is not chosen],
    )


# -- wheel availability ---------------------------------------------------- #


def platform_tag() -> str:
    """The wheel platform tag this machine needs."""
    machine = platform.machine().lower()
    if os.name == "nt":
        return "win_arm64" if machine in ("arm64", "aarch64") else "win_amd64"
    if machine in ("arm64", "aarch64"):
        return "manylinux_2_28_aarch64"
    return "manylinux_2_28_x86_64"


def wheel_available(
    backend: Backend,
    *,
    python_tag: str = PYTHON_TAG,
    tag: str | None = None,
    timeout: float = 20.0,
    opener=None,
) -> bool | None:
    """Whether the index carries a torch wheel for this backend and platform.

    Returns ``None`` when the question could not be answered -- an offline
    machine or a corporate proxy -- which callers must not treat as "no".
    """
    if not backend.is_cuda:
        return True
    url = f"{backend.index_url}/torch/"
    tag = tag or platform_tag()
    try:
        fetch = opener or urllib.request.urlopen
        with fetch(url, timeout=timeout) as response:  # noqa: S310 - fixed https host
            if getattr(response, "status", 200) != 200:
                return False
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        # 403 is what a non-existent index returns, and is a definite "no".
        return False if exc.code in (403, 404) else None
    except (OSError, urllib.error.URLError, ValueError):
        return None
    return bool(_wheel_pattern(python_tag, tag).search(body))


def newest_torch(backend: Backend, body: str, *, python_tag: str = PYTHON_TAG, tag: str | None = None) -> str | None:
    """Highest torch version on an already-fetched index page. Diagnostic only."""
    versions = [
        tuple(int(part) for part in match.group(1).split("."))
        for match in _wheel_pattern(python_tag, tag or platform_tag()).finditer(body)
    ]
    return ".".join(str(part) for part in max(versions)) if versions else None


def _wheel_pattern(python_tag: str, tag: str) -> re.Pattern[str]:
    # The index escapes the local-version '+' as %2B on some entries and not on
    # others, so both spellings have to match.
    return re.compile(
        rf"torch-(\d+\.\d+\.\d+)(?:%2B|\+)cu\d+-{python_tag}-{python_tag}t?-{re.escape(tag)}\.whl"
    )
