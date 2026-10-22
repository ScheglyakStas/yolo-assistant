"""What changes when the app runs as an installed, frozen build.

A PyInstaller build is not something the suite can produce in a second, so
these tests imitate one -- ``sys.frozen`` and ``sys._MEIPASS`` are all the app
looks at -- and ``packaging/build.py`` covers the real thing by running the
frozen app's ``--self-check`` before it packages anything.
"""

import sys
from pathlib import Path

import pytest

from yolo_assistant.app import main as app_main
from yolo_assistant.app import self_check
from yolo_assistant.app.core import paths

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frozen(monkeypatch, tmp_path):
    """A fake bundle holding a small worker package, and a fresh data root."""
    bundle = tmp_path / "bundle"
    (bundle / "worker" / "commands").mkdir(parents=True)
    (bundle / "worker" / "__main__.py").write_text("print('worker')\n")
    (bundle / "worker" / "commands" / "train.py").write_text("STEP = 5\n")
    (bundle / "worker" / "__pycache__").mkdir()
    (bundle / "worker" / "__pycache__" / "stale.cpython-312.pyc").write_bytes(b"\0")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    paths._extracted_worker.cache_clear()
    yield bundle
    paths._extracted_worker.cache_clear()


def test_the_worker_is_copied_out_of_the_bundle(frozen):
    root = paths.worker_source_dir()

    assert root.parent == paths.workers_dir()
    assert (root / "worker" / "__main__.py").read_text() == "print('worker')\n"
    assert (root / "worker" / "commands" / "train.py").read_text() == "STEP = 5\n"
    # Bytecode compiled for the build machine's Python is not the source.
    assert not (root / "worker" / "__pycache__").exists()


def test_the_same_build_reuses_its_copy(frozen):
    first = paths.worker_source_dir()
    (first / "worker" / "marker").write_text("still here")
    paths._extracted_worker.cache_clear()

    assert paths.worker_source_dir() == first
    assert (first / "worker" / "marker").exists()


def test_an_upgrade_gets_a_directory_of_its_own(frozen):
    """A running job keeps importing the copy it started with."""
    old = paths.worker_source_dir()
    (frozen / "worker" / "commands" / "train.py").write_text("STEP = 6\n")
    paths._extracted_worker.cache_clear()

    new = paths.worker_source_dir()
    assert new != old
    assert (old / "worker" / "commands" / "train.py").read_text() == "STEP = 5\n"
    assert (new / "worker" / "commands" / "train.py").read_text() == "STEP = 6\n"


def test_a_damaged_copy_is_replaced(frozen):
    """Files removed from a finished copy -- antivirus quarantine, say."""
    target = paths.worker_source_dir()
    paths._extracted_worker.cache_clear()
    (target / "worker" / "__main__.py").unlink()

    assert paths.worker_source_dir() == target
    assert (target / "worker" / "__main__.py").read_text() == "print('worker')\n"
    leftovers = [p for p in paths.workers_dir().iterdir() if p != target]
    assert leftovers == [], "staging directories must not be left behind"


def test_a_source_checkout_runs_the_worker_in_place(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert paths.bundle_dir() is None
    assert paths.worker_source_dir() == ROOT


def test_the_simulated_run_is_refused_in_an_installed_build(frozen, capsys):
    with pytest.raises(SystemExit) as exit_info:
        app_main.main(["--fake-worker"])
    assert exit_info.value.code == 2
    assert "source checkout" in capsys.readouterr().err


def test_the_self_check_passes_from_source(qt_app, monkeypatch, tmp_path, capsys):
    """Every check that applies to a checkout -- the build runs the rest frozen."""
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.delattr(sys, "frozen", raising=False)

    assert self_check.run() == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "FAIL" not in out
    assert "all checks passed" in out
