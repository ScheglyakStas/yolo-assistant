"""An environment is only "ready" once something has proved it.

These tests walk the states a real installation passes through -- missing, half
built, built by an older app, built but unable to see the GPU, working -- because
each one has a different remedy and telling them apart is the whole point of
``ready.json``.
"""

import json
import os

import pytest

from yolo_assistant.app.core import paths
from yolo_assistant.app.core.bootstrap import venv_manager as vm


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path))
    return tmp_path


@pytest.fixture
def spec(isolated):
    return vm.spec_for("cu128")


def make_venv(spec):
    spec.python.parent.mkdir(parents=True, exist_ok=True)
    spec.python.write_text("#!/bin/sh\n")
    return spec


PROBE = {
    "torch": "2.11.0+cu128",
    "cuda_build": "12.8",
    "cuda_available": True,
    "ultralytics": "8.4.154",
    "devices": [{"index": 0, "name": "RTX 5070 Ti"}],
    "packages": {name: "1.0" for name in vm.PACKAGE_NAMES},
}


def test_missing_environment_says_what_it_will_cost(spec):
    status = vm.status(spec)
    assert status.state == vm.MISSING
    assert not status.is_usable
    assert vm.DOWNLOAD_ESTIMATE in status.message


def test_an_interrupted_install_is_incomplete_not_ready(spec):
    """The failure this design exists to prevent: a half-built environment
    that looks finished because the directory is there."""
    make_venv(spec)
    status = vm.status(spec)
    assert status.state == vm.INCOMPLETE
    assert not status.is_usable


def test_a_proved_environment_is_ready(spec):
    make_venv(spec)
    vm.write_ready(spec, PROBE)
    status = vm.status(spec)
    assert status.state == vm.READY
    assert status.is_usable
    assert "8.4.154" in status.message and "RTX 5070 Ti" in status.message


def test_an_environment_missing_a_package_is_not_ready(spec):
    """The bug this check exists for.

    OWLv2's image processor imports scipy, nothing in the requirements pulls it
    in transitively, and an install without it exits 0 and looks perfect. Before
    this, the way to find out was an error on the first photo of Step 3.
    """
    make_venv(spec)
    vm.write_ready(spec, {**PROBE, "packages": {**PROBE["packages"], "scipy": None}})

    status = vm.status(spec)

    assert status.state == vm.INCOMPLETE
    assert not status.is_usable
    assert "scipy" in status.message and "autolabeling" in status.message
    assert "already downloaded" in status.message, "it must not sound like a 12 GB rebuild"
    assert vm.missing_packages(status.ready) == ["scipy"]


def test_an_environment_proved_before_the_check_existed_is_rechecked(spec):
    make_venv(spec)
    vm.write_ready(spec, {k: v for k, v in PROBE.items() if k != "packages"})

    status = vm.status(spec)

    # Not "broken": nobody ever asked it the question. Repairing asks.
    assert status.state == vm.INCOMPLETE
    assert "earlier version of the app" in status.message
    assert vm.missing_packages(status.ready) == [], "an unanswered question is not a failure"


def test_the_toolkit_the_qat_toggle_needs_is_installed_and_probed():
    """Step 5's INT8 toggle imports modelopt, so it has to be installed -- and
    probed, because an environment that cannot import it is still perfectly good
    for every other step."""
    requirements = vm.BASE_REQUIREMENTS.read_text(encoding="utf-8")

    assert "nvidia-modelopt" in requirements
    assert "modelopt" in vm.PACKAGE_NAMES
    assert "modelopt" not in dict(vm.REQUIRED_PACKAGES)


def test_every_required_package_names_what_needs_it(spec):
    # The message a user reads is built from these, so a bare module name with
    # no explanation would be a tooltip-less parameter by another route.
    assert all(purpose.strip() for _name, purpose in vm.REQUIRED_PACKAGES)
    assert "scipy" in vm.PACKAGE_NAMES


def test_an_optional_package_is_recorded_but_does_not_lock_the_app(spec):
    """onnxruntime-gpu can fail to import over a system library no reinstall
    provides, and only Step 7 wants it. Reporting it is right; refusing to let
    anyone train because of it is not."""
    make_venv(spec)
    vm.write_ready(spec, {**PROBE, "packages": {**PROBE["packages"], "onnxruntime": None}})

    status = vm.status(spec)

    assert status.state == vm.READY
    assert status.ready.packages["onnxruntime"] is None, "still recorded, for diagnostics"


def test_installed_but_blind_to_the_gpu_is_degraded(spec):
    make_venv(spec)
    vm.write_ready(spec, {**PROBE, "cuda_available": False})
    status = vm.status(spec)
    assert status.state == vm.DEGRADED
    # Still usable: it will train on the CPU, slowly, which beats refusing to run.
    assert status.is_usable


def test_a_cpu_torch_in_a_cuda_environment_needs_repair_not_patience(spec):
    """The wrong wheel, not a limit of the machine: "degraded" would tell the
    user to live with CPU training when repairing fixes it."""
    make_venv(spec)
    vm.write_ready(
        spec, {**PROBE, "torch": "2.14.1+cpu", "cuda_build": None, "cuda_available": False}
    )
    status = vm.status(spec)
    assert status.state == vm.INCOMPLETE
    assert "2.14.1+cpu" in status.message


def write_torch_version(spec, cuda):
    site = spec.path / ("Lib" if os.name == "nt" else "lib/python3.12") / "site-packages"
    (site / "torch").mkdir(parents=True, exist_ok=True)
    (site / "torch" / "version.py").write_text(
        f"__version__ = '2.14.1'\ncuda: Optional[str] = {cuda!r}\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("backend", "cuda", "expected"),
    [("cu132", None, True), ("cu132", "13.2", False), ("cpu", None, False)],
)
def test_a_cpu_torch_is_recognised_on_disk(isolated, backend, cuda, expected):
    """Its package metadata says plain ``2.14.1`` either way, so the repair has
    to look at what the build was compiled for."""
    spec = vm.spec_for(backend)
    write_torch_version(spec, cuda)
    assert vm.has_cpu_torch(spec) is expected


def test_no_torch_at_all_is_not_a_cpu_torch(spec):
    make_venv(spec)
    assert vm.has_cpu_torch(spec) is False


def test_an_environment_from_an_older_app_is_stale(spec):
    make_venv(spec)
    vm.write_ready(spec, PROBE)
    raw = json.loads(spec.ready_file.read_text())
    raw["schema"] = vm.ENV_SCHEMA - 1
    spec.ready_file.write_text(json.dumps(raw))
    assert vm.status(spec).state == vm.STALE


def test_a_ready_file_from_a_different_fingerprint_is_stale(spec):
    """Guards against a directory being copied or a fingerprint scheme change."""
    make_venv(spec)
    vm.write_ready(spec, PROBE)
    raw = json.loads(spec.ready_file.read_text())
    raw["fingerprint"] = "0" * 12
    spec.ready_file.write_text(json.dumps(raw))
    assert vm.status(spec).state == vm.STALE


def test_corrupt_ready_file_reads_as_unfinished(spec):
    make_venv(spec)
    spec.ready_file.write_text("{not json")
    assert vm.status(spec).state == vm.INCOMPLETE


def test_clearing_ready_marks_it_unproven_again(spec):
    make_venv(spec)
    vm.write_ready(spec, PROBE)
    vm.clear_ready(spec)
    assert vm.status(spec).state == vm.INCOMPLETE
    vm.clear_ready(spec)  # idempotent: an update may run twice


def test_a_build_without_kernels_for_this_gpu_is_caught(spec):
    """torch.cuda.is_available() is True here. It answers "is there a driver and
    a device", not "can this build run on it" -- and the difference only shows
    up as a kernel error at the first training step."""
    make_venv(spec)
    info = vm.write_ready(
        spec,
        {
            **PROBE,
            "arch_list": ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"],
            "devices": [{"index": 0, "name": "RTX 5070 Ti", "compute_capability": "12.0"}],
        },
    )
    message = vm.missing_kernels(info)
    assert "no kernel image" in message
    assert "RTX 5070 Ti" in message and "12.0" in message


def test_a_matching_build_raises_no_complaint(spec):
    make_venv(spec)
    info = vm.write_ready(
        spec,
        {
            **PROBE,
            "arch_list": ["sm_90", "sm_100", "sm_120"],
            "devices": [{"index": 0, "name": "RTX 5070 Ti", "compute_capability": "12.0"}],
        },
    )
    assert vm.missing_kernels(info) == ""


def test_no_arch_list_means_no_opinion(spec):
    """Environments built by an older version of this app have no arch list, and
    guessing from its absence would be worse than staying quiet."""
    make_venv(spec)
    assert vm.missing_kernels(vm.write_ready(spec, PROBE)) == ""
    assert vm.missing_kernels(None) == ""


def test_the_arch_list_survives_a_round_trip(spec):
    make_venv(spec)
    vm.write_ready(spec, {**PROBE, "arch_list": ["sm_120"]})
    assert vm.status(spec).ready.arch_list == ["sm_120"]


def test_backends_get_separate_directories(isolated):
    """So a new environment can be built while the old one still works."""
    assert vm.spec_for("cu128").path != vm.spec_for("cu126").path
    assert vm.spec_for("cu128").fingerprint != vm.spec_for("cu126").fingerprint


def test_the_fingerprint_is_stable_across_calls(isolated):
    assert vm.spec_for("cu128").fingerprint == vm.spec_for("cu128").fingerprint


def test_requirements_name_the_index_and_the_torch_floor():
    text = vm.render_requirements(vm.BY_KEY["cu128"], base="ultralytics==8.4.154\n")
    assert "download.pytorch.org/whl/cu128" in text
    assert "torch>=2.7" in text and "torchvision" in text
    assert "ultralytics==8.4.154" in text


def test_cpu_requirements_do_not_claim_an_index():
    text = vm.render_requirements(vm.BY_KEY["cpu"], base="")
    assert "download.pytorch.org" not in text
    assert "PyPI" in text


def test_discover_and_remove(spec, isolated):
    make_venv(spec)
    vm.write_ready(spec, PROBE)
    assert [found.backend.key for found, _status in vm.discover()] == ["cu128"]
    vm.remove(spec)
    assert vm.discover() == []
