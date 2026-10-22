"""Training, driven against a stand-in for Ultralytics.

The trainer itself is not the interesting part and cannot run here -- no GPU, no
torch, and a checkpoint download. Everything around it is interesting and is
where a mistake would be invisible for hours: which folder the results land in,
what the progress and metric messages say, when the estimate is allowed to
appear, and what is left behind when the user stops halfway.

``load_yolo`` is therefore replaced wholesale, which is why it is a module
function and not an import at the top of the command. The stand-in calls the
callbacks the command registered, in the order Ultralytics calls them, with an
object that answers the same attributes -- so the reporting logic is exercised
for real.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worker.channel import Cancelled
from worker.commands import train
from worker.util.errors import DatasetError, QatUnavailable

from test_autolabel_worker import build_dataset


class FakeChannel:
    """The subset of ``Channel`` a command touches, with a cancel switch."""

    def __init__(self, cancel_after: int | None = None) -> None:
        self.logs: list[tuple[str, str]] = []
        self.warnings: list[tuple[str, str]] = []
        self.artifacts: list[tuple[str, str]] = []
        self.progress_calls: list[dict] = []
        self.metric_calls: list[tuple[int, dict, bool]] = []
        self.ready_info: dict = {}
        #: Everything in the order it was sent, so a test can ask what was known
        #: at a given moment rather than only what was sent overall.
        self.events: list[tuple[str, object]] = []
        self._cancel_after = cancel_after
        self._checks = 0

    def progress(self, phase, current, total=None, *, unit="items", eta_s=None, detail=""):
        call = {
            "phase": getattr(phase, "value", phase),
            "current": current,
            "total": total,
            "unit": unit,
            "eta_s": eta_s,
            "detail": detail,
        }
        self.progress_calls.append(call)
        self.events.append(("progress", call))

    def metrics(self, epoch, values, *, best=False):
        self.metric_calls.append((epoch, dict(values), best))
        self.events.append(("metrics", epoch))

    def ready(self, **info):
        self.ready_info = info

    def log(self, msg, *, level="info"):
        self.logs.append((level, msg))

    def warning(self, code, msg, *, hint=""):
        self.warnings.append((code, msg))

    def artifact(self, kind, path, **meta):
        self.artifacts.append((kind, str(path)))

    def check_cancelled(self):
        self._checks += 1
        if self._cancel_after is not None and self._checks > self._cancel_after:
            raise Cancelled()

    # -- helpers for the assertions ---------------------------------------- #

    @property
    def text(self) -> str:
        return "\n".join(msg for _level, msg in self.logs)


class FakeTrainer:
    """The handful of attributes the reporter reads off Ultralytics' trainer."""

    def __init__(self, save_dir: Path, *, epochs: int, batches: int) -> None:
        self.save_dir = save_dir
        self.epochs = epochs
        self.epoch = 0
        self.train_loader = [object()] * batches
        self.metrics: dict = {}
        self.lr = {"lr/pg0": 0.01}
        self.tloss = [1.0, 0.5]
        self.fitness = 0.0
        self.best_fitness = 0.0

    def label_loss_items(self, losses, prefix="train"):
        return {f"{prefix}/box_loss": 1.0, f"{prefix}/cls_loss": 0.5}


class StubYOLO:
    """Calls the registered callbacks the way ``_do_train`` does.

    ``fitness`` climbs for ``improving_epochs`` and then falls, so that the
    "best epoch was a while ago" reporting has something real to describe.
    """

    def __init__(
        self,
        *,
        batches: int = 4,
        improving_epochs: int = 99,
        write_weights=("best.pt", "last.pt"),
        oom_after: int | None = None,
    ):
        self.callbacks: dict[str, list] = {}
        self.train_kwargs: dict = {}
        self.trainer: FakeTrainer | None = None
        self._batches = batches
        self._improving = improving_epochs
        self._write_weights = write_weights
        #: Batches into the first epoch at which to "run out of memory", halve
        #: the batch and restart the epoch, as Ultralytics' ``_do_train`` does.
        self._oom_after = oom_after

    def add_callback(self, event, handler):
        self.callbacks.setdefault(event, []).append(handler)

    def _fire(self, event):
        for handler in self.callbacks.get(event, []):
            handler(self.trainer)

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        epochs = int(kwargs["epochs"])
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        self.trainer = FakeTrainer(save_dir, epochs=epochs, batches=self._batches)

        self._fire("on_pretrain_routine_end")
        self._fire("on_train_start")
        if self._oom_after is not None:
            self.trainer.epoch = 0
            self._fire("on_train_epoch_start")
            for _ in range(self._oom_after):
                self._fire("on_train_batch_end")
            self._batches *= 2
            self.trainer.batch_size = 8
            self.trainer.train_loader = [object()] * self._batches
        for epoch in range(epochs):
            self.trainer.epoch = epoch
            self._fire("on_train_epoch_start")
            for _ in range(self._batches):
                self._fire("on_train_batch_end")
            fitness = (
                0.1 * (epoch + 1) if epoch < self._improving else 0.1 * self._improving - 0.01
            )
            self.trainer.fitness = fitness
            self.trainer.best_fitness = max(self.trainer.best_fitness, fitness)
            self.trainer.metrics = {
                "metrics/mAP50(B)": round(fitness + 0.2, 4),
                "metrics/mAP50-95(B)": round(fitness, 4),
            }
            self._fire("on_fit_epoch_end")
        self._fire("on_train_end")

        if self._write_weights:
            weights = save_dir / "weights"
            weights.mkdir(parents=True, exist_ok=True)
            for name in self._write_weights:
                (weights / name).write_bytes(b"weights")
        return {}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def labelled_project(root: Path, classes=(("cat", 3), ("dog", 2))) -> Path:
    """A prepared dataset with every photo labelled, as Step 3 would leave it."""
    build_dataset(root, classes=classes)
    dataset = root / "dataset"
    (dataset / "data.yaml").write_text("path: .\ntrain: train.txt\n", encoding="utf-8")
    for name, count in classes:
        for number in range(1, count + 1):
            (dataset / "labels" / name / f"{name}_{number:06d}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
    return root


def job_for(root: Path, **extra) -> dict:
    job = {
        "project": str(root),
        "data": str(root / "dataset" / "data.yaml"),
        "runs": str(root / "runs"),
        "name": "train",
        "model": "yolo11n.pt",
        "epochs": 3,
        "imgsz": 640,
        "imgsz_hw": [640, 640],
        "device": "cpu",
        "args": {},
    }
    job.update(extra)
    return job


@pytest.fixture
def project(tmp_path):
    return labelled_project(tmp_path / "project")


@pytest.fixture
def yolo(monkeypatch):
    stub = StubYOLO()
    monkeypatch.setattr(train, "load_yolo", lambda *a, **k: stub)
    return stub


@pytest.fixture(autouse=True)
def no_toolchain_search(monkeypatch):
    """The real one runs vswhere and vcvars64.bat and rewrites this process's PATH."""
    monkeypatch.setattr(train, "_prepare_kernel_toolchain", lambda: {})


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_a_run_lands_in_the_folder_it_was_named(project, yolo):
    result = train.run(FakeChannel(), job_for(project, name="bottles"))

    run_dir = project / "runs" / "bottles"
    assert result["run"] == str(run_dir)
    assert result["name"] == "bottles"
    assert result["weights"] == str(run_dir / "weights" / "best.pt")
    # exist_ok, because the app has already decided the name is free or is being
    # deliberately replaced. Without it Ultralytics would invent "bottles2".
    assert yolo.train_kwargs["exist_ok"] is True
    assert yolo.train_kwargs["name"] == "bottles"


def test_the_app_decides_model_data_device_and_size(project, yolo):
    train.run(FakeChannel(), job_for(project, args={"batch": 8, "patience": 20}))

    kwargs = yolo.train_kwargs
    assert kwargs["data"] == str(project / "dataset" / "data.yaml")
    assert kwargs["device"] == "cpu"
    assert kwargs["imgsz"] == 640
    assert kwargs["epochs"] == 3
    # The user's own settings survive alongside them.
    assert kwargs["batch"] == 8
    assert kwargs["patience"] == 20


def test_every_epoch_reports_its_metrics(project, yolo):
    channel = FakeChannel()

    train.run(channel, job_for(project))

    assert [epoch for epoch, _values, _best in channel.metric_calls] == [1, 2, 3]
    _epoch, values, best = channel.metric_calls[-1]
    assert best is True  # fitness climbs in the stub, so the last is the best
    assert values["metrics/mAP50-95(B)"] == pytest.approx(0.3)
    # Losses and learning rate ride along with the accuracy figures.
    assert "train/box_loss" in values
    assert "lr/pg0" in values


def test_the_best_epoch_is_the_one_ultralytics_says_it_is(project, monkeypatch):
    """Fitness climbs for two epochs and then falls; best must stay at two."""
    stub = StubYOLO(improving_epochs=2)
    monkeypatch.setattr(train, "load_yolo", lambda *a, **k: stub)
    channel = FakeChannel()

    result = train.run(channel, job_for(project, epochs=4))

    assert [best for _e, _v, best in channel.metric_calls] == [True, True, False, False]
    assert result["best_epoch"] == 2
    assert result["best_fitness"] == pytest.approx(0.2)


def test_progress_counts_batches_and_names_the_epoch(project, yolo):
    channel = FakeChannel()

    train.run(channel, job_for(project, epochs=3))  # 3 epochs x 4 batches

    training = [call for call in channel.progress_calls if call["phase"] == "train"]
    assert training[-1]["current"] == 12
    assert training[-1]["total"] == 12
    assert training[-1]["unit"] == "batches"
    # The specification asks for epochs trained out of total; the bar is finer
    # than that, so the epoch travels alongside it.
    assert "epoch 3 of 3" in training[-1]["detail"]


def test_an_out_of_memory_restart_recounts_the_batches(project, monkeypatch):
    """Half the batch is twice the batches, and the aborted ones do not count."""
    stub = StubYOLO(batches=4, oom_after=2)
    monkeypatch.setattr(train, "load_yolo", lambda *a, **k: stub)
    channel = FakeChannel()

    train.run(channel, job_for(project, epochs=3))  # restarts at 3 epochs x 8 batches

    training = [call for call in channel.progress_calls if call["phase"] == "train"]
    restart = next(
        i for i, call in enumerate(training) if call["total"] == 24
    )
    assert training[restart]["current"] == 0
    assert [call["current"] for call in training[restart:]] == sorted(
        call["current"] for call in training[restart:]
    )
    assert training[-1]["current"] == 24
    assert training[-1]["total"] == 24
    assert "batch size 8" in channel.text


def test_no_estimate_is_offered_until_an_epoch_has_been_timed(project, yolo):
    """The first epoch pays for the download and the warmup, so it is a liar."""
    channel = FakeChannel()

    train.run(channel, job_for(project, epochs=3))

    # Everything sent before the first epoch finished -- which is the only time
    # the app has nothing but a partial, download-inflated epoch to go on.
    before_first_epoch_ended = []
    for kind, payload in channel.events:
        if kind == "metrics":
            break
        before_first_epoch_ended.append(payload)

    assert before_first_epoch_ended
    assert all(call["eta_s"] is None for call in before_first_epoch_ended)
    assert any(call["eta_s"] is not None for call in channel.progress_calls)


def test_the_run_record_survives_for_step_6(project, yolo):
    train.run(FakeChannel(), job_for(project, name="bottles", imgsz_hw=[480, 640], qat=False))

    record = json.loads((project / "runs" / "bottles" / "umla.json").read_text())
    # The height/width pair is the thing Ultralytics never sees and Step 6 needs.
    assert record["imgsz_hw"] == [480, 640]
    assert record["imgsz"] == 640
    assert record["qat"] is False
    assert record["epochs_completed"] == 3
    assert record["finished_at"] > 0


def test_the_weights_are_announced_as_artifacts(project, yolo):
    channel = FakeChannel()

    train.run(channel, job_for(project, name="bottles"))

    kinds = {kind for kind, _path in channel.artifacts}
    paths = {Path(path).name for _kind, path in channel.artifacts}
    assert kinds == {"weights"}
    assert paths == {"best.pt", "last.pt"}


# --------------------------------------------------------------------------- #
# Refusals that happen before anything is downloaded
# --------------------------------------------------------------------------- #


def test_an_unprepared_dataset_is_refused(tmp_path, yolo):
    (tmp_path / "project").mkdir()

    with pytest.raises(DatasetError, match="prepared dataset"):
        train.run(FakeChannel(), job_for(tmp_path / "project"))


def test_a_dataset_with_no_labels_at_all_is_refused(tmp_path, yolo):
    root = build_dataset(tmp_path / "project")
    (root / "dataset" / "data.yaml").write_text("path: .\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="nothing for the model to learn"):
        train.run(FakeChannel(), job_for(root))


def test_an_image_size_off_the_grid_is_refused(project, yolo):
    with pytest.raises(DatasetError, match="multiple of 32"):
        train.run(FakeChannel(), job_for(project, imgsz=641))


def test_a_tiny_dataset_warns_but_still_trains(project, yolo):
    channel = FakeChannel()

    train.run(channel, job_for(project))

    assert any("mostly luck" in msg for _code, msg in channel.warnings)


def test_nothing_is_downloaded_before_the_dataset_is_checked(tmp_path, monkeypatch):
    """A refusal must cost seconds, not a checkpoint."""
    monkeypatch.setattr(
        train, "load_yolo", lambda *a, **k: pytest.fail("the model was loaded")
    )
    (tmp_path / "project").mkdir()

    with pytest.raises(DatasetError):
        train.run(FakeChannel(), job_for(tmp_path / "project"))


# --------------------------------------------------------------------------- #
# The run folder
# --------------------------------------------------------------------------- #


def test_a_previous_run_of_the_same_name_is_replaced(project, yolo):
    stale = project / "runs" / "train" / "weights"
    stale.mkdir(parents=True)
    (stale / "best.pt").write_bytes(b"old")
    (project / "runs" / "train" / "results.csv").write_text("old", encoding="utf-8")

    train.run(FakeChannel(), job_for(project))

    # Replaced, not merged: a leftover results.csv from another run would be
    # read as this run's history.
    assert not (project / "runs" / "train" / "results.csv").exists()
    assert (project / "runs" / "train" / "weights" / "best.pt").read_bytes() == b"weights"


def test_a_folder_that_is_not_a_run_is_never_deleted(project, yolo):
    intruder = project / "runs" / "train"
    intruder.mkdir(parents=True)
    (intruder / "holiday.jpg").write_bytes(b"not a training run")

    with pytest.raises(DatasetError, match="will not be deleted"):
        train.run(FakeChannel(), job_for(project))

    assert (intruder / "holiday.jpg").exists()


def test_an_existing_run_is_kept_when_overwriting_was_not_agreed(project, yolo):
    (project / "runs" / "train" / "weights").mkdir(parents=True)

    with pytest.raises(DatasetError, match="already exists"):
        train.run(FakeChannel(), job_for(project, overwrite=False))


def test_a_run_that_saved_nothing_says_so(project, monkeypatch):
    monkeypatch.setattr(
        train, "load_yolo", lambda *a, **k: StubYOLO(write_weights=())
    )
    channel = FakeChannel()

    result = train.run(channel, job_for(project))

    assert result["weights"] == ""
    assert any("nothing to export" in msg for _code, msg in channel.warnings)


def test_a_stopped_run_reports_the_checkpoint_it_did_leave(project, monkeypatch):
    """Ultralytics writes best.pt only once it has chosen a best epoch. A run
    that ended before that still left a loadable model behind."""
    monkeypatch.setattr(
        train, "load_yolo", lambda *a, **k: StubYOLO(write_weights=("last.pt",))
    )

    result = train.run(FakeChannel(), job_for(project, name="stopped"))

    assert Path(result["weights"]).name == "last.pt"


@pytest.mark.parametrize("given", [None, [], [640], "nonsense", [640, "x"]])
def test_a_malformed_size_pair_falls_back_rather_than_vanishing(project, yolo, given):
    """Step 6 reads the pair back out of this file, and Ultralytics never sees
    it -- so a broken one has to become something, not nothing."""
    train.run(FakeChannel(), job_for(project, name="pair", imgsz_hw=given))

    record = json.loads((project / "runs" / "pair" / "umla.json").read_text())
    assert record["imgsz_hw"] == [640, 640]


# --------------------------------------------------------------------------- #
# Stopping
# --------------------------------------------------------------------------- #


def test_cancelling_unwinds_through_the_batch_callback(project, yolo):
    """An epoch can be twenty minutes; the app starts signalling after five
    seconds. The per-batch check is the only one fast enough to matter."""
    channel = FakeChannel(cancel_after=6)

    with pytest.raises(Cancelled):
        train.run(channel, job_for(project, epochs=10))

    # It stopped part-way through, not at the end of the run.
    assert len(channel.metric_calls) < 10


def test_what_was_asked_for_is_on_disk_even_if_the_run_never_finishes(project, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    stub = StubYOLO()
    stub.train = explode
    monkeypatch.setattr(train, "load_yolo", lambda *a, **k: stub)

    with pytest.raises(RuntimeError):
        train.run(FakeChannel(), job_for(project, name="bottles"))

    record = json.loads((project / "runs" / "bottles" / "umla.json").read_text())
    assert record["model"] == "yolo11n.pt"
    assert "finished_at" not in record


# --------------------------------------------------------------------------- #
# INT8 QAT
# --------------------------------------------------------------------------- #


def test_qat_without_the_toolkit_fails_in_seconds_rather_than_pretending(project, monkeypatch):
    monkeypatch.setattr(
        train, "load_yolo", lambda *a, **k: pytest.fail("the model was loaded")
    )
    monkeypatch.setattr(train, "_require_qat_toolkit", _raise_qat_unavailable)

    with pytest.raises(QatUnavailable):
        train.run(FakeChannel(), job_for(project, qat=True))


def test_qat_is_asked_of_ultralytics_so_the_checkpoint_carries_it(project, yolo, monkeypatch):
    """Quantising the live model from a callback misses the EMA copy that is
    saved: best.pt came out float, with quantize=None, and export calibrated it
    again. Only Ultralytics' own ``quantize=8`` puts the quantisers in best.pt."""
    monkeypatch.setattr(train, "_require_qat_toolkit", lambda: None)

    train.run(FakeChannel(), job_for(project, qat=True))

    assert yolo.train_kwargs["quantize"] == 8


def test_qat_turns_mixed_precision_off(project, yolo, monkeypatch):
    """Fake quantisers calibrated on fp32 ranges, fed fp16 ones by autocast, is
    a quietly worse model rather than an error -- so it is not left to chance."""
    monkeypatch.setattr(train, "_require_qat_toolkit", lambda: None)

    train.run(FakeChannel(), job_for(project, qat=True))

    assert yolo.train_kwargs["amp"] is False


def test_without_qat_precision_is_left_to_ultralytics(project, yolo):
    train.run(FakeChannel(), job_for(project))

    assert "amp" not in yolo.train_kwargs
    assert "quantize" not in yolo.train_kwargs


def test_the_compilers_are_found_before_model_optimizer_is_imported(project, yolo, monkeypatch):
    """torch reads CUDA_HOME once, when Model Optimizer imports cpp_extension."""
    order = []
    monkeypatch.setattr(train, "_prepare_kernel_toolchain", lambda: order.append("toolchain") or {})
    monkeypatch.setattr(train, "_require_qat_toolkit", lambda: order.append("modelopt"))

    train.run(FakeChannel(), job_for(project, qat=True))

    assert order == ["toolchain", "modelopt"]


def test_without_qat_no_compiler_is_looked_for(project, yolo, monkeypatch):
    monkeypatch.setattr(
        train, "_prepare_kernel_toolchain", lambda: pytest.fail("looked for a compiler")
    )

    train.run(FakeChannel(), job_for(project))


def test_a_missing_microsoft_compiler_is_said_out_loud(project, yolo, monkeypatch):
    monkeypatch.setattr(train, "_require_qat_toolkit", lambda: None)
    monkeypatch.setattr(
        train,
        "_prepare_kernel_toolchain",
        lambda: {"torch_cuda": "13.2", "msvc": False, "cuda_toolkit": True},
    )
    channel = FakeChannel()

    train.run(channel, job_for(project, qat=True))

    messages = [msg for _code, msg in channel.warnings]
    assert any("Microsoft C++ compiler" in msg for msg in messages)
    assert not any("CUDA Toolkit" in msg for msg in messages)
    # Still a QAT run: the fallback is slower, not wrong.
    assert yolo.train_kwargs["quantize"] == 8


def test_a_missing_cuda_toolkit_names_the_version_torch_needs(project, yolo, monkeypatch):
    monkeypatch.setattr(train, "_require_qat_toolkit", lambda: None)
    monkeypatch.setattr(
        train,
        "_prepare_kernel_toolchain",
        lambda: {"torch_cuda": "12.8", "msvc": True, "cuda_toolkit": False},
    )
    channel = FakeChannel()

    train.run(channel, job_for(project, qat=True))

    messages = [msg for _code, msg in channel.warnings]
    assert any("CUDA Toolkit 12.x" in msg for msg in messages)
    assert not any("Microsoft C++ compiler" in msg for msg in messages)


def test_a_found_toolchain_says_nothing(project, yolo, monkeypatch):
    monkeypatch.setattr(train, "_require_qat_toolkit", lambda: None)
    monkeypatch.setattr(
        train,
        "_prepare_kernel_toolchain",
        lambda: {"torch_cuda": "13.2", "msvc": True, "cuda_toolkit": True},
    )
    monkeypatch.setattr(train.shutil, "which", lambda name, *a, **k: f"/bin/{name}")
    channel = FakeChannel()

    train.run(channel, job_for(project, qat=True))

    assert not any("cannot be compiled" in msg for _code, msg in channel.warnings)


def _raise_qat_unavailable():
    raise QatUnavailable("no toolkit here")
