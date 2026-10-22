"""Backend selection: the decision that fails silently if it is wrong.

A wrong choice here does not raise at install time. The environment builds, the
app reports success, and the first training run dies with "no kernel image is
available for execution on the device" -- so these tests exist to pin down the
three constraints that must all hold, especially the two that ``uv``'s own
``--torch-backend=auto`` does not check: compute capability and wheel existence.
"""

import urllib.error

import pytest

from yolo_assistant.app.core.bootstrap import planner
from yolo_assistant.app.core.env_probe import DriverInfo, GpuInfo, SystemReport


def report(*, driver="595.84", cuda="13.2", compute="12.0", name="RTX 5070 Ti", gpus=None):
    if gpus is None:
        gpus = [GpuInfo(0, name, 16.0, compute)] if driver else []
    return SystemReport(
        driver=DriverInfo(present=bool(driver), version=driver or "", cuda_version=cuda),
        gpus=gpus,
    )


def test_modern_gpu_gets_the_newest_backend():
    plan = planner.plan_backend(report(), windows=False)
    assert plan.backend.key == "cu132"
    assert "595.84" in plan.reason and "12.0" in plan.reason


def test_old_driver_is_limited_to_cuda_12():
    """An r535 driver predates CUDA 13, so the 13.x indexes must not be offered."""
    usable = planner.usable_backends(report(driver="535.104", cuda="12.2"), windows=False)
    keys = [backend.key for backend in usable]
    assert keys == ["cu129", "cu128", "cu126"]


def test_pascal_falls_back_despite_a_new_driver():
    """The trap: a current driver, but a GPU newer torch builds have no kernels for."""
    plan = planner.plan_backend(report(compute="6.1", name="GTX 1080 Ti"), windows=False)
    assert plan.backend.key == "cu126"
    assert any("older than the hardware" in warning for warning in plan.warnings)


def test_no_driver_falls_back_to_cpu_with_a_warning():
    plan = planner.plan_backend(report(driver="", cuda="", gpus=[]), windows=False)
    assert plan.is_cpu_fallback
    assert plan.backend.key == "cpu"
    assert any("too slow to be" in warning for warning in plan.warnings)


def test_windows_needs_a_higher_driver_for_cuda_12():
    machine = report(driver="526.00", cuda="12.0")
    assert planner.usable_backends(machine, windows=False)
    assert not planner.usable_backends(machine, windows=True)


def test_the_weakest_gpu_decides():
    """torch loads one build for all devices, so a second, older card constrains it."""
    machine = report(
        gpus=[GpuInfo(0, "RTX 5070 Ti", 16.0, "12.0"), GpuInfo(1, "GTX 1080", 8.0, "6.1")]
    )
    assert [b.key for b in planner.usable_backends(machine, windows=False)] == ["cu126"]


def test_a_manual_choice_is_honoured_but_flagged():
    plan = planner.plan_backend(
        report(compute="6.1"), preferred="cu132", windows=False
    )
    assert plan.backend.key == "cu132"
    assert any("selected manually" in warning for warning in plan.warnings)


def test_missing_wheels_step_down_to_the_next_backend():
    seen = []

    def availability(backend):
        seen.append(backend.key)
        return backend.key == "cu129"

    plan = planner.plan_backend(report(), availability=availability, windows=False)
    assert plan.backend.key == "cu129"
    assert seen == ["cu132", "cu130", "cu129"]


def test_an_unanswerable_availability_check_does_not_downgrade():
    """A proxy failure must not quietly hand someone a CPU build."""
    plan = planner.plan_backend(report(), availability=lambda _b: None, windows=False)
    assert plan.backend.key == "cu132"


def test_all_wheels_missing_ends_at_cpu():
    plan = planner.plan_backend(report(), availability=lambda _b: False, windows=False)
    assert plan.backend.key == "cpu"
    assert len(plan.warnings) == len(planner.usable_backends(report(), windows=False))


# -- index parsing --------------------------------------------------------- #


class FakeResponse:
    def __init__(self, body: str, status: int = 200):
        self._body = body.encode()
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


INDEX = """<a href="torch-2.14.0%2Bcu132-cp312-cp312-manylinux_2_28_x86_64.whl">x</a>
<a href="torch-2.13.0+cu132-cp312-cp312-win_amd64.whl">y</a>
<a href="torch-2.14.0+cu132-cp311-cp311-manylinux_2_28_x86_64.whl">z</a>"""


def test_wheel_found_for_this_platform():
    assert planner.wheel_available(
        planner.BY_KEY["cu132"],
        tag="manylinux_2_28_x86_64",
        opener=lambda *_a, **_k: FakeResponse(INDEX),
    ) is True


def test_wheel_absent_for_an_unsupported_python():
    assert planner.wheel_available(
        planner.BY_KEY["cu132"],
        python_tag="cp39",
        tag="manylinux_2_28_x86_64",
        opener=lambda *_a, **_k: FakeResponse(INDEX),
    ) is False


def test_a_nonexistent_index_is_a_definite_no():
    """download.pytorch.org answers 403 for an index that was never published."""

    def opener(*_a, **_k):
        raise urllib.error.HTTPError("url", 403, "Forbidden", {}, None)

    assert planner.wheel_available(planner.BY_KEY["cu132"], opener=opener) is False


def test_a_network_failure_is_unknown_not_no():
    def opener(*_a, **_k):
        raise OSError("connection reset")

    assert planner.wheel_available(planner.BY_KEY["cu132"], opener=opener) is None


def test_newest_torch_reads_both_plus_spellings():
    assert planner.newest_torch(
        planner.BY_KEY["cu132"], INDEX, tag="manylinux_2_28_x86_64"
    ) == "2.14.0"


@pytest.mark.parametrize("backend", planner.BACKENDS)
def test_every_backend_is_self_consistent(backend):
    assert backend.torch_spec.startswith("torch")
    if backend.is_cuda:
        assert backend.index_url.endswith(backend.key)
        assert backend.min_driver_windows >= backend.min_driver_linux
