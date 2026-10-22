"""Update Ultralytics inside an existing environment, showing the plan first.

A plain ``pip install -U ultralytics`` is a trap here. Ultralytics depends on
torch with a wide floor, so a resolver is free to satisfy an upgrade by pulling
a *different* torch -- and if that torch comes from PyPI rather than the CUDA
index, the result is a working install that has silently lost the GPU. The user
would discover this at their next training run, with no obvious connection to
the button they pressed.

Hence two passes. ``preview`` runs the resolution with ``--dry-run`` and shows
exactly what would change; ``apply`` repeats it for real, keeping the same CUDA
index in play so torch cannot drift, and then re-runs ``worker probe`` before
the environment is allowed to call itself ready again.

Unlike a backend change, this mutates the environment in place. Rebuilding ten
gigabytes to move one package would make the update button unusable, so the
trade is deliberate: ``ready.json`` is removed first, so an interrupted update
leaves the environment marked unfinished rather than ready-but-wrong.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from ..i18n import tr
from ..job import JobRunner, JobSpec
from ..paths import cache_dir, logs_dir, worker_source_dir
from ..protocol import ErrorCode
from . import uv_manager, venv_manager
from .venv_manager import EnvSpec, VenvStatus

PREVIEW = "preview"
INSTALL = "install"
VERIFY = "verify"


class UpdateController(QObject):
    """Runs the preview, then -- only if asked again -- the update itself."""

    plan_ready = Signal(list)        # the dry-run output, verbatim
    log_line = Signal(str, str)
    failed = Signal(str, str, str)   # code, message, hint
    finished = Signal(bool, object)  # ok, VenvStatus | None
    busy_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._runner = JobRunner(self)
        self._runner.log_line.connect(self._on_log)
        self._runner.finished.connect(self._on_finished)
        self._runner.failed.connect(self._on_failed)

        self._spec: EnvSpec | None = None
        self._target = ""
        self._stage = ""
        self._plan: list[str] = []
        self._probe: dict = {}
        self._failed = False
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    # -- public API -------------------------------------------------------- #

    def preview(self, spec: EnvSpec, target: str = "ultralytics") -> None:
        """Resolve the update without changing anything."""
        self._begin(spec, target, PREVIEW)
        self._plan = []
        self._start_pip(["--dry-run", "-U", target])

    def apply(self, spec: EnvSpec, target: str = "ultralytics") -> None:
        """Install the update, then re-verify the environment."""
        if self._active:
            raise RuntimeError("an update is already running")
        # Before anything else, including announcing that we are busy: from here
        # on the environment is being changed, so it must not claim to be proven
        # at any point an observer could look.
        venv_manager.clear_ready(spec)
        self._begin(spec, target, INSTALL)
        self._start_pip(["-U", target])

    def cancel(self) -> None:
        if self._runner.is_active():
            self._runner.cancel()

    # -- internals --------------------------------------------------------- #

    def _begin(self, spec: EnvSpec, target: str, stage: str) -> None:
        if self._active:
            raise RuntimeError("an update is already running")
        self._spec = spec
        self._target = target
        self._stage = stage
        self._failed = False
        self._probe = {}
        self._set_active(True)

    def _start_pip(self, extra: list[str]) -> None:
        assert self._spec is not None
        uv = uv_manager.find_uv()
        if uv is None:
            self._fail(
                ErrorCode.VENV_BROKEN.value,
                tr("The environment tool is missing."),
                tr("Run environment setup first; updates apply to an environment this app built."),
            )
            return
        spec = self._spec
        argv = uv_manager.pip_install_argv(
            uv,
            spec.python,
            packages=[],
            # Keep torch on the CUDA index for the upgrade too. Without it an
            # Ultralytics bump can quietly replace the GPU torch with the CPU
            # build that happens to be newest on PyPI.
            torch_backend=spec.backend.key if spec.backend.is_cuda else None,
            extra_args=extra,
        )
        self._runner.start(
            JobSpec(
                command="uv",
                argv=argv,
                expect_result=False,
                error_code=ErrorCode.DEPENDENCY_INSTALL_FAILED.value,
                error_hint=tr(
                    "The update could not be installed. The environment was left as it was."
                ),
                worker_dir=worker_source_dir(),
                cache_dir=cache_dir(),
                log_path=logs_dir() / "update.log",
                env_extra={k: v for k, v in uv_manager.uv_env().items() if k.startswith("UV_")},
            )
        )

    def _start_probe(self) -> None:
        assert self._spec is not None
        self._stage = VERIFY
        self._runner.start(
            JobSpec(
                command="probe",
                job={"packages": list(venv_manager.PACKAGE_NAMES)},
                python=str(self._spec.python),
                worker_dir=worker_source_dir(),
                cache_dir=cache_dir(),
                log_path=logs_dir() / "update.log",
            )
        )

    # -- slots ------------------------------------------------------------- #

    def _on_log(self, text: str, level: str) -> None:
        self.log_line.emit(text, level)
        if self._stage == PREVIEW and level in ("stdout", "stderr") and text.strip():
            self._plan.append(text)

    def _on_failed(self, code: str, msg: str, hint: str, _traceback: str) -> None:
        self._failed = True
        self._fail(code, msg, hint)

    def _on_finished(self, ok: bool, payload: dict) -> None:
        if not self._active:
            return
        if payload.get("cancelled"):
            self._set_active(False)
            self.log_line.emit(tr("Update cancelled."), "info")
            self.finished.emit(False, None)
            return
        if not ok:
            if not self._failed:
                self._fail(
                    ErrorCode.DEPENDENCY_INSTALL_FAILED.value,
                    tr("The update stopped unexpectedly."),
                    "",
                )
            return

        if self._stage == PREVIEW:
            self._set_active(False)
            self.plan_ready.emit(list(self._plan))
            self.finished.emit(True, None)
            return

        if self._stage == INSTALL:
            self._start_probe()
            return

        # VERIFY
        assert self._spec is not None
        self._probe = dict(payload)
        venv_manager.write_ready(self._spec, self._probe)
        status: VenvStatus = venv_manager.status(self._spec)
        self._set_active(False)
        self.log_line.emit(tr("Updated. {message}", message=status.message), "info")
        self.finished.emit(True, status)

    def _fail(self, code: str, msg: str, hint: str) -> None:
        self._set_active(False)
        self.failed.emit(code, msg, hint)
        self.finished.emit(False, None)

    def _set_active(self, active: bool) -> None:
        if active != self._active:
            self._active = active
            self.busy_changed.emit(active)
