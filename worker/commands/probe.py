"""Report what the managed environment actually contains.

Run inside the venv, this is the authority for Step 1's "CUDA" gate: not whether
a CUDA Toolkit is installed system-wide (torch ships its own runtime and does not
need one), but whether *this* torch can see *this* GPU.

It also answers the other half of that question, which is easy to forget and
expensive to learn later: whether the packages the later steps import are
actually importable. An install that resolved cleanly can still be missing
something no dependency declares -- OWLv2's image processor needs scipy, and
nothing in the requirements pulls it in by itself -- and the symptom is an
error at the first photo of Step 3 rather than at the end of Step 1. The caller
says which modules it cares about, so this list never has to be kept in step
with the app's by hand.
"""

from __future__ import annotations

import importlib
import platform
import sys

from worker.channel import Channel
from worker.i18n import tr
from worker.protocol import Phase

#: Used when the command is run by hand with no job attached. The GUI always
#: sends its own list.
DEFAULT_PACKAGES = (
    "torch",
    "ultralytics",
    "transformers",
    "scipy",
    "cv2",
    "onnxruntime",
    "modelopt",
)


def run(channel: Channel, job: dict) -> dict:
    channel.progress(Phase.SETUP, 0, 3, unit="checks")
    info: dict = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }

    # The imports first, and unconditionally. An environment with no torch is
    # still worth describing -- "torch, ultralytics and scipy cannot be
    # imported" names the problem, where stopping at the first failure only
    # reports that the GPU could not be reached.
    wanted = [str(name) for name in (job.get("packages") or DEFAULT_PACKAGES)]
    packages: dict[str, str | None] = {}
    for name in wanted:
        version = _version_of(name)
        packages[name] = version
        # Kept as flat keys too: ready.json has recorded them that way since M1
        # and the diagnostics view reads them from there.
        info[name] = version
        if version is None:
            channel.log(
                tr("{name} cannot be imported in this environment.", name=name),
                level="warning",
            )
    info["packages"] = packages
    channel.progress(Phase.SETUP, 1, 3, unit="checks")

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        info["torch"] = None
        info["torch_import_error"] = str(exc)
        channel.progress(Phase.SETUP, 3, 3, unit="checks")
        return info

    info["torch"] = torch.__version__
    info["cuda_build"] = torch.version.cuda
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["devices"] = []
    # Which GPU architectures this build actually contains kernels for. This is
    # the one fact that explains "no kernel image is available for execution":
    # torch.cuda.is_available() can be True while this list has nothing for the
    # installed card, and the failure then only appears at the first training
    # step, far from its cause.
    info["arch_list"] = list(torch.cuda.get_arch_list()) if info["cuda_available"] else []
    channel.progress(Phase.SETUP, 2, 3, unit="checks")
    if info["cuda_available"]:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info["devices"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )

    channel.progress(Phase.SETUP, 3, 3, unit="checks")
    return info


def _version_of(name: str) -> str | None:
    """The module's version, or ``None`` if it cannot be imported at all.

    Imported rather than looked up in the package metadata: a wheel whose
    files were half-written, or a native extension built against a different
    numpy, is installed as far as ``pip list`` is concerned and still fails
    the moment anything uses it.
    """
    try:
        module = importlib.import_module(name)
    except Exception:  # noqa: BLE001 - a broken install raises anything at all
        return None
    return str(getattr(module, "__version__", "unknown"))
