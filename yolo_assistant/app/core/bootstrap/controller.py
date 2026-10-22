"""Run the environment bootstrap as an ordered sequence of stages.

Each stage is an ordinary job under :class:`~..job.JobRunner`, so the whole
sequence inherits the machinery M0 established: streamed output into the log
pane and a log file, a cancel ladder that escalates from a signal to killing the
process tree, and grandchild reaping -- which matters here more than anywhere
else, because uv spawns its own download and unpack children and a cancelled
install that leaves them running will corrupt the cache.

The stages, in order:

1. **uv** -- fetched over HTTPS and checksum-verified, unless one is already
   present. This is the only stage that is not a subprocess.
2. **python** -- ``uv python install``, which gives us an interpreter without
   touching the user's system Python or asking for administrator rights.
3. **venv** -- an empty environment at a fingerprinted path.
4. **packages** -- the CUDA-matched torch and everything above it. The long one:
   most of the 10-14 GB is here.
5. **verify** -- ``worker probe`` *inside* the new environment. Only if this
   reports that torch can see the GPU, and that every package the later steps
   import can be imported, is ``ready.json`` written.

Stage 5 is the point of the whole design. An install can succeed completely and
still produce an environment that cannot train, and the only way to find out is
to import torch and ask it. Doing that here means the user learns at the end of
Step 1 rather than three steps later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from ..i18n import N_, tr
from ..job import JobRunner, JobSpec, JobState
from ..paths import cache_dir, logs_dir, worker_source_dir
from ..protocol import ErrorCode, Phase
from . import uv_manager, venv_manager
from .planner import MANAGED_PYTHON
from .uv_output import UvOutputParser
from .venv_manager import EnvSpec, VenvStatus


@dataclass(frozen=True)
class Stage:
    key: str
    title: str


#: Sentinel the fetch thread reports instead of an error message.
CANCELLED = "\x00cancelled"

#: Titles are templates: ``{python}`` is the managed Python version.
STAGES: tuple[Stage, ...] = (
    Stage("uv", N_("Fetching the environment tool")),
    Stage("python", N_("Installing Python {python}")),
    Stage("venv", N_("Creating the environment")),
    Stage("packages", N_("Installing PyTorch and Ultralytics")),
    Stage("verify", N_("Checking the environment can use the GPU")),
)


class _UvFetchThread(QThread):
    """Downloads uv off the GUI thread.

    The one stage that is not a subprocess, because there is no subprocess to
    run yet -- this is the thing that runs everything else.
    """

    progressed = Signal(int, int)
    done = Signal(object, str)  # Path | None, error message ("" or CANCELLED)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            path = uv_manager.download_uv(
                progress=lambda done, total: self.progressed.emit(done, total),
                should_stop=lambda: self._stop,
            )
        except uv_manager.UvCancelled:
            self.done.emit(None, CANCELLED)
            return
        except Exception as exc:  # noqa: BLE001 - reported, never raised across threads
            self.done.emit(None, str(exc))
            return
        self.done.emit(path, "")


class BootstrapController(QObject):
    """Drives the stages above, one at a time."""

    stage_changed = Signal(str, str, int, int)   # key, title, index, count
    progress = Signal(dict)
    log_line = Signal(str, str)
    failed = Signal(str, str, str)               # code, message, hint
    finished = Signal(bool, object)              # ok, VenvStatus | None
    busy_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._runner = JobRunner(self)
        self._runner.log_line.connect(self._on_log)
        self._runner.finished.connect(self._on_stage_finished)
        self._runner.failed.connect(self._on_stage_failed)
        self._runner.state_changed.connect(self._on_state)

        self._spec: EnvSpec | None = None
        self._uv: Path | None = None
        self._stage_index = -1
        self._parser = UvOutputParser()
        self._label = ""
        self._fetch: _UvFetchThread | None = None
        self._last_progress: dict | None = None
        self._probe_payload: dict = {}
        self._failed = False
        self._active = False

        # uv prints nothing at all while a single large wheel downloads, and
        # torch is a three-gigabyte wheel: on a slow connection that is several
        # silent minutes. Re-emitting the last progress once a second keeps the
        # elapsed time moving, which is the whole difference between a window
        # that is working and a window that looks frozen.
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._on_tick)

    # -- public API -------------------------------------------------------- #

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self, spec: EnvSpec) -> None:
        if self._active:
            raise RuntimeError("a bootstrap is already running")
        # Before anything else, including announcing that we are busy: an
        # environment being rebuilt must not claim to be proven at any point an
        # observer could look, so a crash halfway leaves it "incomplete" rather
        # than "ready but wrong".
        venv_manager.clear_ready(spec)
        self._spec = spec
        self._stage_index = -1
        self._failed = False
        self._probe_payload = {}
        self._set_active(True)
        self._tick.start()
        self.log_line.emit(
            tr("Setting up {backend} in {path}", backend=tr(spec.backend.label), path=spec.path),
            "info",
        )
        self._advance()

    def cancel(self) -> None:
        if self._fetch is not None and self._fetch.isRunning():
            self._fetch.stop()
        if self._runner.is_active():
            self._runner.cancel()

    def shutdown(self) -> None:
        """Stop everything and wait, so nothing outlives the window.

        A QThread destroyed while still running aborts the process outright
        rather than raising, so this is not optional tidiness.
        """
        self.cancel()
        if self._fetch is not None:
            self._fetch.wait(5000)
            self._fetch = None

    # -- stage sequencing -------------------------------------------------- #

    def _advance(self) -> None:
        self._stage_index += 1
        if self._stage_index >= len(STAGES):
            self._complete()
            return
        stage = STAGES[self._stage_index]
        title = tr(stage.title, python=MANAGED_PYTHON)
        self.stage_changed.emit(stage.key, title, self._stage_index, len(STAGES))
        self._parser = UvOutputParser()
        getattr(self, f"_run_{stage.key}")()

    def _run_uv(self) -> None:
        existing = uv_manager.find_uv()
        if existing is not None:
            self._uv = existing
            self.log_line.emit(tr("Using uv at {path}", path=existing), "info")
            self._advance()
            return

        self.log_line.emit(tr("Downloading uv..."), "info")
        self._fetch = _UvFetchThread(self)
        self._fetch.progressed.connect(self._on_fetch_progress)
        self._fetch.done.connect(self._on_fetch_done)
        self._fetch.start()

    def _run_python(self) -> None:
        assert self._uv is not None
        self._start_uv(
            # --no-bin: we launch the interpreter by absolute path and never
            # want a shim on the user's PATH. It also avoids uv warning the
            # user about a PATH entry they did not ask for.
            uv_manager.python_install_argv(self._uv, MANAGED_PYTHON) + ["--no-bin"],
            error_code=ErrorCode.DEPENDENCY_INSTALL_FAILED.value,
            hint=tr(
                "Python could not be downloaded. Check the network connection or proxy settings."
            ),
            label=f"Python {MANAGED_PYTHON}",
        )

    def _run_venv(self) -> None:
        assert self._uv is not None and self._spec is not None
        spec = self._spec
        if spec.python.exists():
            # Resuming an interrupted setup. Keeping the existing environment is
            # the difference between re-downloading ten gigabytes and not.
            self.log_line.emit(
                tr("Reusing the existing environment at {path}", path=spec.path), "info"
            )
            self._advance()
            return
        if spec.path.exists():
            # A directory with no interpreter in it is the debris of a failed
            # attempt. uv refuses to create an environment over an existing
            # directory, and its own --clear will not remove one that was never
            # a virtual environment, so clear it here, where the intent is plain.
            self.log_line.emit(
                tr("Removing an unusable environment at {path}", path=spec.path), "warning"
            )
            venv_manager.remove(spec)
        self._start_uv(
            uv_manager.venv_argv(self._uv, spec.path, spec.python_version),
            error_code=ErrorCode.VENV_BROKEN.value,
            hint=tr("The environment could not be created at {path}.", path=spec.path),
            label=tr("Creating the environment"),
        )

    def _run_packages(self) -> None:
        assert self._uv is not None and self._spec is not None
        spec = self._spec
        # Written here rather than at the start of the sequence: it lives inside
        # the environment directory, and uv refuses to create an environment in
        # a directory that already exists.
        venv_manager.write_requirements(spec, venv_manager.render_requirements(spec.backend))
        extra: list[str] = []
        if venv_manager.has_cpu_torch(spec):
            # The CPU wheel already satisfies the version range, so without this
            # uv would check it, find nothing to do, and leave the GPU unusable.
            self.log_line.emit(
                tr("Replacing the CPU-only PyTorch in this environment with the GPU build."),
                "warning",
            )
            extra = ["--reinstall-package", "torch", "--reinstall-package", "torchvision"]
        argv = uv_manager.pip_install_argv(
            self._uv,
            spec.python,
            requirements=spec.requirements_file,
            torch_backend=spec.backend.key if spec.backend.is_cuda else None,
            extra_args=extra,
        )
        self.log_line.emit(
            tr(
                "Installing from {path}. This downloads roughly "
                "{estimate} the first time and can take 5-15 minutes.",
                path=spec.requirements_file,
                estimate=venv_manager.DOWNLOAD_ESTIMATE,
            ),
            "info",
        )
        self._start_uv(
            argv,
            error_code=ErrorCode.DEPENDENCY_INSTALL_FAILED.value,
            hint="",
        )

    def _run_verify(self) -> None:
        assert self._spec is not None
        self._runner.start(
            JobSpec(
                command="probe",
                # The probe is told what to import rather than keeping its own
                # copy of the list; one place decides what "installed" means.
                job={"packages": list(venv_manager.PACKAGE_NAMES)},
                python=str(self._spec.python),
                worker_dir=worker_source_dir(),
                cache_dir=cache_dir(),
                log_path=logs_dir() / "bootstrap.log",
            )
        )

    def _start_uv(self, argv: list[str], *, error_code: str, hint: str, label: str = "") -> None:
        # uv's package-install output is what the progress parser understands;
        # the shorter stages describe themselves rather than borrowing its words.
        self._label = label
        self._runner.start(
            JobSpec(
                command="uv",
                argv=argv,
                expect_result=False,
                error_code=error_code,
                error_hint=hint,
                worker_dir=worker_source_dir(),
                cache_dir=cache_dir(),
                log_path=logs_dir() / "bootstrap.log",
                env_extra=_uv_environment(),
            )
        )

    # -- slots ------------------------------------------------------------- #

    def _emit_progress(self, payload: dict) -> None:
        self._last_progress = payload
        self.progress.emit(payload)

    def _on_tick(self) -> None:
        """Repeat the last progress so the elapsed time keeps moving."""
        if self._active and self._last_progress is not None:
            self.progress.emit(self._last_progress)

    def _on_fetch_progress(self, done: int, total: int) -> None:
        self._emit_progress(
            {
                "phase": Phase.DOWNLOAD.value,
                "current": done // (1 << 20),
                "total": (total // (1 << 20)) or None,
                "unit": "MB",
                "detail": "uv",
            }
        )

    def _on_fetch_done(self, path, error: str) -> None:
        self._fetch = None
        if error == CANCELLED:
            self._cancelled()
            return
        if path is None:
            self._fail(
                ErrorCode.DOWNLOAD_FAILED.value,
                tr("uv could not be downloaded: {error}", error=error),
                tr(
                    "The application needs to download a small tool (about 20 MB) before it "
                    "can create the environment. Check the network connection or proxy "
                    "settings."
                ),
            )
            return
        self._uv = path
        self.log_line.emit(tr("Installed uv at {path}", path=path), "info")
        self._advance()

    def _on_log(self, text: str, level: str) -> None:
        self.log_line.emit(text, level)
        if level not in ("stdout", "stderr"):
            return
        if self._parser.feed(text):
            state = self._parser.state
            detail = self._label or state.describe()
            byte_text = state.describe_bytes()
            self._emit_progress(
                {
                    "phase": Phase.INSTALL.value,
                    "current": state.current,
                    "total": state.total,
                    "unit": "packages",
                    "detail": f"{detail} ({byte_text})" if byte_text else detail,
                }
            )

    def _on_state(self, state: JobState) -> None:
        if state is JobState.CANCELLING:
            self.log_line.emit(tr("Cancelling the environment setup..."), "warning")

    def _on_stage_failed(self, code: str, msg: str, hint: str, _traceback: str) -> None:
        self._failed = True
        self._fail(code, msg, hint or self._parser.failure_hint())

    def _on_stage_finished(self, ok: bool, payload: dict) -> None:
        if not self._active:
            return
        if payload.get("cancelled"):
            self._cancelled()
            return
        stage = STAGES[self._stage_index].key if 0 <= self._stage_index < len(STAGES) else ""

        if not ok:
            if self._failed:
                return  # the failure was already reported with its own code
            if stage == "verify":
                self._fail(
                    ErrorCode.VENV_BROKEN.value,
                    tr("The environment was installed but could not be started."),
                    tr(
                        "PyTorch failed to import. The log below has the details; removing "
                        "the environment and setting it up again usually resolves it."
                    ),
                )
            else:
                self._fail(
                    ErrorCode.DEPENDENCY_INSTALL_FAILED.value,
                    tr("The environment setup stopped unexpectedly."),
                    self._parser.failure_hint(),
                )
            return

        if stage == "verify":
            self._probe_payload = dict(payload)
        self._advance()

    # -- endings ----------------------------------------------------------- #

    def _complete(self) -> None:
        assert self._spec is not None
        info = venv_manager.write_ready(self._spec, self._probe_payload)
        status = venv_manager.status(self._spec)
        self._set_active(False)
        missing = venv_manager.missing_packages(info)
        if missing:
            # The install itself succeeded, so this is not a failure -- but it
            # must not be announced as a working environment either.
            self.log_line.emit(
                tr(
                    "The environment installed, but {packages} could not be "
                    "imported afterwards.",
                    packages=", ".join(missing),
                ),
                "warning",
            )
        elif not info.cuda_available and self._spec.backend.is_cuda:
            self.log_line.emit(
                tr("The environment installed, but PyTorch reports no usable GPU."), "warning"
            )
        else:
            self.log_line.emit(tr("Environment ready."), "info")
        self.finished.emit(True, status)

    def _cancelled(self) -> None:
        self._set_active(False)
        self.log_line.emit(
            tr(
                "Setup cancelled. Nothing downloaded so far is lost -- starting again resumes "
                "from the package cache."
            ),
            "info",
        )
        self.finished.emit(False, None)

    def _fail(self, code: str, msg: str, hint: str) -> None:
        self._set_active(False)
        self.failed.emit(code, msg, hint)
        self.finished.emit(False, None)

    def _set_active(self, active: bool) -> None:
        if not active:
            self._tick.stop()
            self._last_progress = None
        if active != self._active:
            self._active = active
            self.busy_changed.emit(active)


def _uv_environment() -> dict[str, str]:
    """Only the UV_* settings; the rest of the environment is built by job.py."""
    return {key: value for key, value in uv_manager.uv_env().items() if key.startswith("UV_")}


def status_for(spec: EnvSpec) -> VenvStatus:
    return venv_manager.status(spec)


__all__ = ["BootstrapController", "STAGES", "Stage", "status_for"]
