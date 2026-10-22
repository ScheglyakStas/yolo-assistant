"""The managed environment: where it lives, whether it works, and what is in it.

Two rules shape this module.

**A directory is only "ready" when something has proved it.** ``ready.json`` is
written last, after ``worker probe`` has run inside the environment and reported
that torch can see the GPU. An interrupted install therefore leaves a directory
with no ``ready.json``, which reads as unfinished rather than as working -- the
failure mode this exists to prevent is an environment that installed cleanly,
reports success, and then dies at the first training step.

**The environment is addressed by fingerprint, not by name.** Changing the CUDA
backend or the Python version produces a different directory, so a new
environment is built alongside the old one and the old one keeps working until
the new one proves itself. Only an Ultralytics version change mutates an
existing environment in place, because rebuilding 10 GB to move one package
would make the update button unusable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path

from ..i18n import N_, tr
from ..paths import venvs_dir
from .planner import BACKENDS, BY_KEY, MANAGED_PYTHON, Backend
from .uv_manager import venv_python

READY_FILENAME = "ready.json"
REQUIREMENTS_FILENAME = "requirements.txt"

#: Bump when the requirement set changes in a way that an existing environment
#: cannot be upgraded into. That forces a fresh directory rather than an
#: in-place install on top of incompatible packages.
ENV_SCHEMA = 1

#: Rough guide shown before the user commits to the download. Deliberately a
#: range and deliberately honest: torch alone is ~3 GB, TensorRT ~2 GB, and
#: OWLv2 adds ~1.5 GB at first use.
DOWNLOAD_ESTIMATE = "10-14 GB"

REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_REQUIREMENTS = REPO_ROOT / "requirements" / "heavy.txt"

#: Modules the later steps import, and what stops working without each. The
#: probe is asked to import exactly these inside the environment, and the answer
#: goes into ``ready.json``.
#:
#: This list exists because "the install exited 0" and "the work can start" are
#: different claims. OWLv2's image processor needs scipy, which nothing else in
#: the requirements pulls in; before this check, the way to find that out was an
#: error on the first photo of Step 3, one step and several gigabytes after the
#: mistake could have been fixed. Anything added to ``requirements/heavy.txt``
#: because a step imports it belongs here too.
REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("torch", N_("anything at all")),
    ("ultralytics", N_("training and export")),
    ("transformers", N_("autolabeling")),
    ("scipy", N_("autolabeling")),
    ("cv2", N_("reading photos, video and cameras")),
)

#: Probed and recorded, but never a reason to call an environment unusable.
#: ``onnxruntime-gpu`` can fail to import over a system library the app does not
#: install and a reinstall cannot conjure, and it is only wanted in Step 7 --
#: locking Steps 2 to 6 behind it would be a worse bug than the one it reports.
#: Step 7 raises its own coded error if it is really missing.
OPTIONAL_PACKAGES: tuple[tuple[str, str], ...] = (
    ("onnxruntime", N_("testing an exported model")),
    ("modelopt", N_("training a model for INT8")),
)

PACKAGE_NAMES: tuple[str, ...] = tuple(
    name for name, _ in REQUIRED_PACKAGES + OPTIONAL_PACKAGES
)


@dataclass(frozen=True)
class EnvSpec:
    """Identifies one managed environment."""

    backend: Backend
    python_version: str = MANAGED_PYTHON

    @property
    def fingerprint(self) -> str:
        material = f"{ENV_SCHEMA}|{self.backend.key}|{self.python_version}"
        return hashlib.sha256(material.encode()).hexdigest()[:12]

    @property
    def path(self) -> Path:
        return venvs_dir() / f"{self.backend.key}-py{self.python_version}-{self.fingerprint}"

    @property
    def python(self) -> Path:
        return venv_python(self.path)

    @property
    def ready_file(self) -> Path:
        return self.path / READY_FILENAME

    @property
    def requirements_file(self) -> Path:
        return self.path / REQUIREMENTS_FILENAME


@dataclass(frozen=True)
class ReadyInfo:
    """What ``worker probe`` found, recorded at the moment the install finished."""

    fingerprint: str
    backend: str
    python_version: str
    torch: str | None = None
    cuda_build: str | None = None
    cuda_available: bool = False
    ultralytics: str | None = None
    transformers: str | None = None
    devices: list[dict] = field(default_factory=list)
    #: GPU architectures this torch build ships kernels for, e.g. ``sm_120``.
    arch_list: list[str] = field(default_factory=list)
    #: Module name -> version, or ``None`` where the import failed. Empty means
    #: the environment was proved by a version of the app that did not ask.
    packages: dict[str, str | None] = field(default_factory=dict)
    installed_at: float = 0.0
    schema: int = ENV_SCHEMA

    def to_obj(self) -> dict:
        return {
            "schema": self.schema,
            "fingerprint": self.fingerprint,
            "backend": self.backend,
            "python_version": self.python_version,
            "torch": self.torch,
            "cuda_build": self.cuda_build,
            "cuda_available": self.cuda_available,
            "ultralytics": self.ultralytics,
            "transformers": self.transformers,
            "devices": self.devices,
            "arch_list": self.arch_list,
            "packages": self.packages,
            "installed_at": self.installed_at,
        }

    @classmethod
    def from_obj(cls, obj: dict) -> "ReadyInfo":
        return cls(
            fingerprint=str(obj.get("fingerprint", "")),
            backend=str(obj.get("backend", "")),
            python_version=str(obj.get("python_version", "")),
            torch=obj.get("torch"),
            cuda_build=obj.get("cuda_build"),
            cuda_available=bool(obj.get("cuda_available", False)),
            ultralytics=obj.get("ultralytics"),
            transformers=obj.get("transformers"),
            devices=list(obj.get("devices") or []),
            arch_list=list(obj.get("arch_list") or []),
            packages=dict(obj.get("packages") or {}),
            installed_at=float(obj.get("installed_at") or 0.0),
            schema=int(obj.get("schema", 0)),
        )


#: What the UI shows, and what decides whether Step 2 unlocks.
MISSING = "missing"
INCOMPLETE = "incomplete"
STALE = "stale"
DEGRADED = "degraded"
READY = "ready"


@dataclass(frozen=True)
class VenvStatus:
    state: str
    message: str
    ready: ReadyInfo | None = None
    spec: EnvSpec | None = None

    @property
    def is_usable(self) -> bool:
        """Whether work can proceed. A degraded environment still runs on CPU."""
        return self.state in (READY, DEGRADED)


def render_requirements(backend: Backend, *, base: str | None = None) -> str:
    """The exact requirements text for ``backend``.

    Written into the environment so that a support request can be answered with
    "show me ``requirements.txt`` from your venv" rather than guesswork.
    """
    if base is None:
        base = BASE_REQUIREMENTS.read_text(encoding="utf-8")
    header = (
        f"# Generated for the {backend.label} backend.\n"
        f"# Installed from {backend.index_url if backend.is_cuda else 'PyPI'}\n"
    )
    torch_lines = f"{backend.torch_spec}\ntorchvision\n"
    return f"{header}{torch_lines}\n{base}"


def status(spec: EnvSpec) -> VenvStatus:
    """Decide, from the filesystem alone, what state the environment is in."""
    if not spec.path.exists() or not spec.python.exists():
        return VenvStatus(
            MISSING,
            tr(
                "No environment for {backend} yet. "
                "Setting one up downloads roughly {estimate}.",
                backend=tr(spec.backend.label),
                estimate=DOWNLOAD_ESTIMATE,
            ),
            spec=spec,
        )

    try:
        raw = json.loads(spec.ready_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return VenvStatus(
            INCOMPLETE,
            tr(
                "An environment was started but never finished. Setting up again resumes it; "
                "packages already downloaded are reused."
            ),
            spec=spec,
        )

    info = ReadyInfo.from_obj(raw)
    if info.schema != ENV_SCHEMA or info.fingerprint != spec.fingerprint:
        return VenvStatus(
            STALE,
            tr("This environment was built by an older version of the app and needs rebuilding."),
            ready=info,
            spec=spec,
        )
    if not info.packages:
        # Proved by a version of the app that only asked about torch. Repairing
        # re-runs the install, which reuses everything already downloaded, and
        # then asks the fuller question.
        return VenvStatus(
            INCOMPLETE,
            tr(
                "This environment was checked by an earlier version of the app, which did "
                "not confirm that everything the later steps need is installed. Repairing "
                "it checks, and installs anything missing -- it does not download it all "
                "again."
            ),
            ready=info,
            spec=spec,
        )

    missing = missing_packages(info)
    if missing:
        return VenvStatus(INCOMPLETE, _missing_message(missing), ready=info, spec=spec)

    if spec.backend.is_cuda and info.torch and not info.cuda_build:
        # Not "degraded": nothing about this machine stops the GPU working. The
        # wrong wheel was installed, and repairing replaces it.
        return VenvStatus(
            INCOMPLETE,
            tr(
                "PyTorch {torch} is the CPU-only build, not the {backend} build this "
                "environment needs. Repairing replaces it with the GPU build.",
                torch=info.torch,
                backend=tr(spec.backend.label),
            ),
            ready=info,
            spec=spec,
        )

    if spec.backend.is_cuda and not info.cuda_available:
        return VenvStatus(
            DEGRADED,
            tr(
                "The environment is installed, but PyTorch cannot use the GPU. "
                "Training will fall back to the CPU and be very slow."
            ),
            ready=info,
            spec=spec,
        )

    device = info.devices[0]["name"] if info.devices else "CPU"
    return VenvStatus(
        READY,
        tr(
            "PyTorch {torch} ({build}) and Ultralytics {ultralytics} on {device}.",
            torch=info.torch,
            build=info.cuda_build or "CPU",
            ultralytics=info.ultralytics,
            device=device,
        ),
        ready=info,
        spec=spec,
    )


def missing_packages(info: ReadyInfo | None) -> list[str]:
    """Which of :data:`REQUIRED_PACKAGES` the environment could not import.

    Nothing is reported when the record predates the check: absence of an
    answer is not a missing package, and ``status`` handles that case on its
    own terms.
    """
    if info is None or not info.packages:
        return []
    return [name for name, _ in REQUIRED_PACKAGES if info.packages.get(name) is None]


def _missing_message(missing: list[str]) -> str:
    purpose = dict(REQUIRED_PACKAGES + OPTIONAL_PACKAGES)
    needs = [tr(need) for need in sorted({purpose.get(name, N_("some steps")) for name in missing})]
    return tr(
        "This environment is missing {packages}, so {purposes} cannot run. "
        "Repairing installs what is missing; nothing already downloaded is fetched again.",
        packages=", ".join(missing),
        purposes=reduce(
            lambda first, second: tr("{first} and {second}", first=first, second=second), needs
        ),
    )


def missing_kernels(info: ReadyInfo | None) -> str:
    """Explain a GPU this torch build has no kernels for, or return "".

    ``torch.cuda.is_available()`` answers "is there a driver and a device", not
    "can this build run on it". A Blackwell card under a cu126 build passes the
    first question and fails the second, and the only symptom is
    ``no kernel image is available for execution`` at the first training step --
    ten minutes and several gigabytes after the mistake was made. Comparing the
    device's compute capability against the build's architecture list catches it
    here instead.
    """
    if info is None or not info.cuda_available or not info.arch_list:
        return ""
    for device in info.devices:
        capability = str(device.get("compute_capability") or "")
        major, _, minor = capability.partition(".")
        if not major.isdigit() or not minor.isdigit():
            continue
        if f"sm_{major}{minor}" not in info.arch_list:
            if "name" not in device:
                return tr(
                    "PyTorch {torch} contains no kernels for this GPU (compute {capability}). "
                    'Training would fail with "no kernel image is available". Choose a '
                    "different PyTorch build below and set the environment up again.",
                    torch=info.torch,
                    capability=capability,
                )
            return tr(
                "PyTorch {torch} contains no kernels for {gpu} (compute {capability}). "
                'Training would fail with "no kernel image is available". Choose a '
                "different PyTorch build below and set the environment up again.",
                torch=info.torch,
                gpu=device["name"],
                capability=capability,
            )
    return ""


def has_cpu_torch(spec: EnvSpec) -> bool:
    """Whether a CUDA environment holds a CPU-only torch, read from disk.

    Repairing has to know this, because ``uv pip install`` leaves an installed
    torch alone when its version satisfies the requirement -- and the CPU wheel
    is the same version as the CUDA one. The package metadata cannot tell them
    apart either (both say ``2.14.1``); only ``torch/version.py`` records the
    CUDA version the build targets, as ``None`` for CPU. Read from the file
    rather than ``ready.json``, which an interrupted repair has already removed.
    """
    if not spec.backend.is_cuda:
        return False
    for version_py in spec.path.glob("[Ll]ib/**/site-packages/torch/version.py"):
        try:
            text = version_py.read_text(encoding="utf-8")
        except OSError:
            continue
        return re.search(r"^cuda\b[^=\n]*=\s*None\s*$", text, re.MULTILINE) is not None
    return False


def write_ready(spec: EnvSpec, probe: dict) -> ReadyInfo:
    """Record a successful probe. This is what makes an environment "ready"."""
    info = ReadyInfo(
        fingerprint=spec.fingerprint,
        backend=spec.backend.key,
        python_version=spec.python_version,
        torch=probe.get("torch"),
        cuda_build=probe.get("cuda_build"),
        cuda_available=bool(probe.get("cuda_available")),
        ultralytics=probe.get("ultralytics"),
        transformers=probe.get("transformers"),
        devices=list(probe.get("devices") or []),
        arch_list=list(probe.get("arch_list") or []),
        packages=dict(probe.get("packages") or {}),
        installed_at=time.time(),
    )
    _atomic_write(spec.ready_file, json.dumps(info.to_obj(), indent=2))
    return info


def clear_ready(spec: EnvSpec) -> None:
    """Mark an environment unproven again, before mutating it."""
    try:
        spec.ready_file.unlink()
    except OSError:
        pass


def write_requirements(spec: EnvSpec, text: str) -> Path:
    spec.path.mkdir(parents=True, exist_ok=True)
    _atomic_write(spec.requirements_file, text)
    return spec.requirements_file


def discover() -> list[tuple[EnvSpec, VenvStatus]]:
    """Every environment this app could use, for the diagnostics view."""
    found = []
    for backend in BACKENDS:
        spec = EnvSpec(backend=backend)
        if spec.path.exists():
            found.append((spec, status(spec)))
    return found


def remove(spec: EnvSpec) -> None:
    """Delete an environment. Used to recover from a corrupt install."""
    shutil.rmtree(spec.path, ignore_errors=True)


def spec_for(backend_key: str, *, python_version: str = MANAGED_PYTHON) -> EnvSpec:
    return EnvSpec(backend=BY_KEY[backend_key], python_version=python_version)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
