"""Export, driven against a stand-in for Ultralytics.

The exporter itself cannot run here -- no torch, no GPU, and a TensorRT build
is minutes of it -- and it is not the interesting part. The interesting parts
are all around it: which keyword the requested precision is spelled with, where
the artifact ends up, what is recorded beside it, and what a failure is turned
into when the model was trained for INT8.

``load_yolo`` is replaced wholesale, which is why it is a module function; the
``torch`` and ``ultralytics.cfg`` imports inside the command are satisfied with
stand-ins registered in ``sys.modules``, because what they are asked is small
and entirely about branching.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from worker.commands import export
from worker.util.errors import (
    ExportFailed,
    GpuUnavailable,
    QatFormatMismatch,
)

from test_train_worker import FakeChannel


class StubYOLO:
    """Records what it was asked for and writes something that looks exported."""

    def __init__(
        self,
        *,
        weights: Path,
        produces: str = "best.onnx",
        is_directory: bool = False,
        raises: Exception | None = None,
        returns=None,
        size: int = 4 * 1024 * 1024,
    ) -> None:
        self.kwargs: dict = {}
        self._weights = weights
        self._produces = produces
        self._is_directory = is_directory
        self._raises = raises
        self._returns = returns
        self._size = size

    def export(self, **kwargs):
        self.kwargs = kwargs
        if self._raises is not None:
            raise self._raises
        if self._returns is not None:
            return self._returns
        # Ultralytics writes beside the checkpoint, which is the whole reason
        # the command has to move the result anywhere.
        produced = self._weights.parent / self._produces
        if self._is_directory:
            produced.mkdir(parents=True, exist_ok=True)
            (produced / "model.xml").write_bytes(b"x" * self._size)
        else:
            produced.write_bytes(b"x" * self._size)
        return str(produced)


@pytest.fixture
def project(tmp_path) -> Path:
    weights = tmp_path / "project" / "runs" / "bottles" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"weights")
    (tmp_path / "project" / "exports").mkdir(parents=True)
    return tmp_path / "project"


def job_for(project: Path, **extra) -> dict:
    job = {
        "project": str(project),
        "weights": str(project / "runs" / "bottles" / "weights" / "best.pt"),
        "run": "bottles",
        "which": "best",
        "exports": str(project / "exports"),
        "output": "bottles_best_fp32.onnx",
        "format": "onnx",
        "precision": "fp32",
        "imgsz_hw": [640, 640],
        "deepstream": False,
        "qat": False,
        "args": {},
        "schema": 1,
    }
    job.update(extra)
    return job


@pytest.fixture
def yolo(monkeypatch, project):
    stub = StubYOLO(weights=project / "runs" / "bottles" / "weights" / "best.pt")
    monkeypatch.setattr(export, "load_yolo", lambda *a, **k: stub)
    return stub


@pytest.fixture
def gpu(monkeypatch):
    """A machine with a working CUDA torch, as far as this command asks."""
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


@pytest.fixture
def no_gpu(monkeypatch):
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


def install_cfg(monkeypatch, keys) -> None:
    """Stand in for ``ultralytics.cfg.DEFAULT_CFG_DICT``."""
    package = types.ModuleType("ultralytics")
    cfg = types.ModuleType("ultralytics.cfg")
    cfg.DEFAULT_CFG_DICT = dict.fromkeys(keys, None)
    package.cfg = cfg
    monkeypatch.setitem(sys.modules, "ultralytics", package)
    monkeypatch.setitem(sys.modules, "ultralytics.cfg", cfg)


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_the_artifact_lands_in_the_exports_folder(project, yolo, gpu):
    result = export.run(FakeChannel(), job_for(project))

    landed = project / "exports" / "bottles_best_fp32.onnx"
    assert landed.is_file()
    assert result["export"] == str(landed)
    # A move, not a copy: a second engine inside the run folder fills disks.
    assert not (project / "runs" / "bottles" / "weights" / "best.onnx").exists()


def test_a_record_is_written_beside_it(project, yolo, gpu):
    export.run(
        FakeChannel(),
        job_for(
            project,
            precision="fp16",
            imgsz_hw=[480, 640],
            output="bottles_best_fp16.onnx",
        ),
    )

    record = json.loads(
        (project / "exports" / "bottles_best_fp16.onnx.umla.json").read_text()
    )
    # The file itself knows none of this, and Step 7 needs all of it.
    assert record["run"] == "bottles"
    assert record["which"] == "best"
    assert record["precision"] == "fp16"
    assert record["imgsz_hw"] == [480, 640]


def test_the_record_is_not_mistaken_for_an_export(project, yolo, gpu):
    export.run(FakeChannel(), job_for(project))

    names = sorted(p.name for p in (project / "exports").iterdir())
    assert names == ["bottles_best_fp32.onnx", "bottles_best_fp32.onnx.umla.json"]


def test_a_directory_export_is_moved_whole(project, monkeypatch, gpu):
    stub = StubYOLO(
        weights=project / "runs" / "bottles" / "weights" / "best.pt",
        produces="best_openvino_model",
        is_directory=True,
    )
    monkeypatch.setattr(export, "load_yolo", lambda *a, **k: stub)

    export.run(
        FakeChannel(),
        job_for(project, format="openvino", output="bottles_best_fp32_openvino_model"),
    )

    landed = project / "exports" / "bottles_best_fp32_openvino_model"
    assert (landed / "model.xml").is_file()


def test_the_settings_the_spec_fixes_are_always_sent(project, yolo, gpu):
    export.run(FakeChannel(), job_for(project))

    assert yolo.kwargs["simplify"] is True
    assert yolo.kwargs["opset"] == 19
    assert yolo.kwargs["format"] == "onnx"
    # Both edges, unlike training: export is where the pair means something.
    assert yolo.kwargs["imgsz"] == [640, 640]


def test_the_device_is_the_apps_choice_not_the_users(project, yolo, gpu):
    export.run(FakeChannel(), job_for(project))

    assert yolo.kwargs["device"] == 0


def test_an_export_replaces_the_one_it_is_named_after(project, yolo, gpu):
    stale = project / "exports" / "bottles_best_fp32.onnx"
    stale.write_bytes(b"old")

    export.run(FakeChannel(), job_for(project))

    assert stale.read_bytes() != b"old"


# --------------------------------------------------------------------------- #
# Precision, whichever way this Ultralytics spells it
# --------------------------------------------------------------------------- #


def test_fp32_needs_no_keyword_at_all(project, yolo, gpu):
    export.run(FakeChannel(), job_for(project, precision="fp32"))

    assert "half" not in yolo.kwargs
    assert "int8" not in yolo.kwargs
    assert "quantize" not in yolo.kwargs


def test_the_older_spelling_is_used_when_that_is_what_is_installed(
    project, yolo, gpu, monkeypatch
):
    install_cfg(monkeypatch, ["half", "int8", "simplify", "opset"])

    export.run(
        FakeChannel(),
        job_for(
            project, precision="int8", format="engine", output="bottles_best_int8.engine"
        ),
    )

    assert yolo.kwargs["int8"] is True
    assert "quantize" not in yolo.kwargs


def test_the_newer_spelling_is_used_when_that_is_what_is_installed(
    project, yolo, gpu, monkeypatch
):
    """Getting this wrong is the quiet kind of wrong: an unknown keyword is
    carried through the config and a full-precision model comes out labelled
    INT8."""
    install_cfg(monkeypatch, ["quantize", "simplify", "opset"])

    export.run(
        FakeChannel(), job_for(project, precision="fp16", output="bottles_best_fp16.onnx")
    )

    assert yolo.kwargs["quantize"] == "fp16"
    assert "half" not in yolo.kwargs


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def test_a_setting_this_ultralytics_does_not_know_stops_the_export(
    project, yolo, gpu, monkeypatch
):
    install_cfg(monkeypatch, ["dynamic", "simplify", "opset"])

    with pytest.raises(ExportFailed, match="dynammic"):
        export.run(FakeChannel(), job_for(project, args={"dynammic": True}))


def test_a_deepstream_setting_this_release_dropped_is_a_warning_not_a_failure(
    project, yolo, gpu, monkeypatch
):
    """The user ticked a box; they did not type "agnostic_nms". Refusing the
    export because the app's own shorthand aged is the app's problem being
    charged to them."""
    install_cfg(monkeypatch, ["nms", "simplify", "opset"])
    channel = FakeChannel()

    export.run(
        channel,
        job_for(project, deepstream=True, args={"nms": False, "agnostic_nms": True}),
    )

    assert yolo.kwargs["nms"] is False
    assert "agnostic_nms" not in yolo.kwargs
    assert any("agnostic_nms" in msg for _code, msg in channel.warnings)


def test_settings_pass_through_when_the_defaults_cannot_be_read(project, yolo, gpu):
    """No ultralytics to ask, so the exporter itself has the final say."""
    export.run(FakeChannel(), job_for(project, args={"dynamic": True}))

    assert yolo.kwargs["dynamic"] is True


# --------------------------------------------------------------------------- #
# What it refuses
# --------------------------------------------------------------------------- #


def test_a_missing_checkpoint_is_reported_before_anything_loads(project, yolo, gpu):
    with pytest.raises(ExportFailed, match="is not there"):
        export.run(FakeChannel(), job_for(project, weights=str(project / "gone.pt")))

    assert yolo.kwargs == {}


def test_a_size_off_the_detection_grid_is_refused(project, yolo, gpu):
    with pytest.raises(ExportFailed, match="multiple of 32"):
        export.run(FakeChannel(), job_for(project, imgsz_hw=[640, 700]))


def test_tensorrt_without_a_gpu_is_an_error_rather_than_a_slow_route(
    project, yolo, no_gpu
):
    with pytest.raises(GpuUnavailable, match="cannot see one"):
        export.run(FakeChannel(), job_for(project, format="engine"))


def test_onnx_without_a_gpu_simply_uses_the_processor(project, yolo, no_gpu):
    export.run(FakeChannel(), job_for(project))

    assert yolo.kwargs["device"] == "cpu"


# --------------------------------------------------------------------------- #
# What a failure is turned into
# --------------------------------------------------------------------------- #


def test_a_qat_model_at_the_wrong_precision_is_told_to_choose_int8(
    project, monkeypatch, gpu
):
    """The specification's case. The exporter's own complaint is about a dtype;
    the connection to the QAT toggle two steps back has to be supplied here."""
    stub = StubYOLO(
        weights=project / "runs" / "bottles" / "weights" / "best.pt",
        raises=RuntimeError("expected scalar type Float but found QInt8"),
    )
    monkeypatch.setattr(export, "load_yolo", lambda *a, **k: stub)

    with pytest.raises(QatFormatMismatch) as caught:
        export.run(FakeChannel(), job_for(project, qat=True, precision="fp16"))

    assert caught.value.code == "qat_format_mismatch"
    assert "Choose INT8" in caught.value.hint


def test_an_ordinary_export_failure_keeps_its_own_code(project, monkeypatch, gpu):
    stub = StubYOLO(
        weights=project / "runs" / "bottles" / "weights" / "best.pt",
        raises=RuntimeError("onnxslim is not installed"),
    )
    monkeypatch.setattr(export, "load_yolo", lambda *a, **k: stub)

    with pytest.raises(ExportFailed) as caught:
        export.run(FakeChannel(), job_for(project))

    assert caught.value.code == "export_failed"
    assert "onnxslim" in str(caught.value)


def test_an_export_that_produced_nothing_is_a_failure(project, monkeypatch, gpu):
    """It "succeeded" and left nothing findable, which is not a success."""
    stub = StubYOLO(
        weights=project / "runs" / "bottles" / "weights" / "best.pt",
        returns="/nowhere/at/all.onnx",
    )
    monkeypatch.setattr(export, "load_yolo", lambda *a, **k: stub)

    with pytest.raises(ExportFailed, match="did not leave a file"):
        export.run(FakeChannel(), job_for(project))


def test_a_suspiciously_small_result_is_flagged_rather_than_hidden(
    project, monkeypatch, gpu
):
    stub = StubYOLO(
        weights=project / "runs" / "bottles" / "weights" / "best.pt", size=1024
    )
    monkeypatch.setattr(export, "load_yolo", lambda *a, **k: stub)
    channel = FakeChannel()

    export.run(channel, job_for(project))

    assert any("small for a detector" in msg for _code, msg in channel.warnings)


# --------------------------------------------------------------------------- #
# What it says while it runs
# --------------------------------------------------------------------------- #


def test_the_artifact_is_announced_so_the_page_can_name_it(project, yolo, gpu):
    channel = FakeChannel()

    export.run(channel, job_for(project))

    kinds = [kind for kind, _path in channel.artifacts]
    assert kinds == ["export"]


def test_progress_reaches_the_end(project, yolo, gpu):
    channel = FakeChannel()

    export.run(channel, job_for(project))

    last = channel.progress_calls[-1]
    assert (last["current"], last["total"]) == (4, 4)
    assert last["phase"] == "export"


def test_the_quiet_stretch_is_announced_before_it_starts(project, yolo, gpu):
    """An export has no callbacks to report from, so the one honest thing to
    do is say so before going quiet."""
    channel = FakeChannel()

    export.run(channel, job_for(project, format="engine"))

    assert "no progress to report" in channel.text
