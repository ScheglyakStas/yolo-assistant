"""Step 1: system requirements, hardware, and building the managed environment.

This is the screen that decides whether anything else in the app can work, and
the only one most users will ever see fail. Three ideas shape it.

**Every check states its own remedy.** A red cross that says "CUDA not found"
teaches a beginner nothing. Each row carries a sentence in plain language and,
where the app can act, the button that acts.

**Nothing claims to be ready until something has proved it.** The CUDA row stays
deliberately unknown until ``worker probe`` has imported torch inside the built
environment and reported that it can see the GPU. Reporting success from the
absence of an error is how people end up training on a CPU for two days without
noticing.

**The long operation is honest about being long.** Setup downloads 10-14 GB. The
progress line names the package being downloaded and the gigabytes so far, the
log pane shows the raw output, and Cancel works at any point -- the alternative
is a window that looks frozen for a quarter of an hour.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core import env_probe, updates
from ...core.bootstrap import venv_manager
from ...core.bootstrap.controller import BootstrapController
from ...core.bootstrap.planner import BACKENDS, BY_KEY, BackendPlan, plan_backend
from ...core.bootstrap.updater import UpdateController
from ...core.bootstrap.venv_manager import EnvSpec, VenvStatus
from ...core.i18n import N_, tr
from ..common.log_console import LogConsole
from ..common.param_form import ParamForm, apply_help
from ..common.progress_header import ProgressHeader
from ..common.status_row import BAD, BUSY, OK, UNKNOWN, WARN, StatusRow
from .base import StepPage

DRIVER_HELP_URL = "https://www.nvidia.com/en-us/drivers/"

AUTO_BACKEND = ""

#: The hardware facts the specification asks for, in display order.
FACTS: tuple[tuple[str, str], ...] = (
    ("cpu", N_("Processor")),
    ("ram", N_("Memory")),
    ("gpu", N_("Graphics")),
    ("driver", N_("Driver")),
    ("os", N_("Operating system")),
    ("ultralytics", "Ultralytics"),
)


def _is_yes(answer: object) -> bool:
    """Did the user press Yes?

    ``QMessageBox.question`` returns a plain ``int``, not a ``StandardButton``
    member, so an identity test against the enum is silently always false and
    the dialog becomes a pair of buttons that do nothing. Compare by value.
    """
    return int(answer) == int(QMessageBox.StandardButton.Yes)


class _UpdateCheckThread(QThread):
    """Asks PyPI for the newest Ultralytics without blocking the window."""

    checked = Signal(object)  # updates.UpdateInfo

    def __init__(self, installed: str | None, parent=None) -> None:
        super().__init__(parent)
        self._installed = installed

    def run(self) -> None:  # noqa: D102 - QThread entry point
        self.checked.emit(updates.check(self._installed))


class SystemStep(StepPage):
    title = N_("System requirements")
    subtitle = N_(
        "Checks that this computer can train a model, and sets up the private Python "
        "environment the app uses. Nothing is installed system-wide."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._report = env_probe.SystemReport()
        self._plan: BackendPlan | None = None
        self._spec: EnvSpec | None = None
        self._status: VenvStatus | None = None
        self._update_info: updates.UpdateInfo | None = None
        self._update_check: _UpdateCheckThread | None = None
        #: Whether the check in flight is the one made at launch rather than a
        #: press of the button. Only the button is owed an answer either way.
        self._update_check_automatic = False
        self._pending_update = ""

        self._bootstrap = BootstrapController(self)
        self._updater = UpdateController(self)

        self._build_ui()
        self._wire()
        self.refresh()

    # -- construction ------------------------------------------------------ #

    def _build_ui(self) -> None:
        heading = QLabel(f"<h2>{tr(self.title)}</h2>")
        subtitle = QLabel(tr(self.subtitle))
        subtitle.setWordWrap(True)

        requirements = QGroupBox(tr("Requirements"))
        self._driver_row = StatusRow(tr("NVIDIA driver"))
        self._cuda_row = StatusRow(tr("GPU support in PyTorch"))
        self._env_row = StatusRow(tr("Python environment"))
        requirements_layout = QVBoxLayout(requirements)
        for row in (self._driver_row, self._cuda_row, self._env_row):
            requirements_layout.addWidget(row)

        self._driver_row.action.clicked.connect(self._open_driver_help)
        self._env_row.action.clicked.connect(self._start_setup)

        hardware = QGroupBox(tr("This computer"))
        self._facts = QFormLayout(hardware)
        # AllNonFixedFieldsGrow, plus an expanding policy on each value below:
        # a word-wrapped QLabel in a QFormLayout reports a minimal width hint,
        # and with the default policy the layout believes it and clips the text
        # instead of wrapping it.
        self._facts.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._fact_labels: dict[str, QLabel] = {}
        self._fact_titles: dict[str, str] = {}
        for key, label in FACTS:
            label = tr(label)
            self._fact_titles[key] = label
            value = QLabel("--")
            value.setWordWrap(True)
            value.setSizePolicy(
                QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred
            )
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._fact_labels[key] = value
            self._facts.addRow(f"{label}:", value)

        environment = QGroupBox(tr("Environment"))
        form = ParamForm(environment)
        self._backend = QComboBox()
        self._backend.addItem(tr("Automatic"), AUTO_BACKEND)
        for backend in BACKENDS:
            self._backend.addItem(tr(backend.label), backend.key)
        form.add_row("system.backend", tr("PyTorch build"), self._backend)

        self._setup_button = QPushButton(tr("Set up environment"))
        apply_help(self._setup_button, "system.setup")
        self._cancel_button = QPushButton(tr("Cancel"))
        self._cancel_button.setEnabled(False)
        self._updates_button = QPushButton(tr("Check for updates"))
        apply_help(self._updates_button, "system.check_updates")
        self._rebuild_button = QPushButton(tr("Remove and rebuild"))
        apply_help(self._rebuild_button, "system.rebuild")

        buttons = QHBoxLayout()
        buttons.addWidget(self._setup_button)
        buttons.addWidget(self._cancel_button)
        buttons.addStretch(1)
        buttons.addWidget(self._updates_button)
        buttons.addWidget(self._rebuild_button)

        self._progress = ProgressHeader()
        self._console = LogConsole()

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(requirements)
        top_layout.addWidget(hardware)
        top_layout.addWidget(environment)
        top_layout.addLayout(buttons)
        top_layout.addWidget(self._progress)
        top_layout.addStretch(1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(self._console)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(subtitle)
        layout.addWidget(splitter, 1)

    def _wire(self) -> None:
        self._backend.currentIndexChanged.connect(self._on_backend_changed)
        self._setup_button.clicked.connect(self._start_setup)
        self._cancel_button.clicked.connect(self._cancel)
        self._updates_button.clicked.connect(lambda: self._check_updates())
        self._rebuild_button.clicked.connect(self._rebuild)

        self._bootstrap.log_line.connect(self._console.append_line)
        self._bootstrap.progress.connect(self._progress.update_progress)
        self._bootstrap.stage_changed.connect(self._on_stage)
        self._bootstrap.failed.connect(self._on_failed)
        self._bootstrap.finished.connect(self._on_bootstrap_finished)
        self._bootstrap.busy_changed.connect(self._on_busy)

        self._updater.log_line.connect(self._console.append_line)
        self._updater.plan_ready.connect(self._on_update_plan)
        self._updater.failed.connect(self._on_failed)
        self._updater.finished.connect(self._on_update_finished)
        self._updater.busy_changed.connect(self._on_busy)

    # -- state ------------------------------------------------------------- #

    def refresh(self) -> None:
        """Re-read the machine and the environment, and repaint everything."""
        self._report = env_probe.probe_system()
        preferred = self._backend.currentData() or None
        # No availability probe here: this runs on the GUI thread on every page
        # entry, and a network round trip would stall the window. The bootstrap
        # verifies the index before it commits to a download.
        self._plan = plan_backend(self._report, preferred=preferred)
        self._spec = EnvSpec(backend=self._plan.backend)
        self._status = venv_manager.status(self._spec)

        self._paint_facts()
        self._paint_rows()
        self._paint_buttons()
        self.gates_changed.emit()

    def _paint_facts(self) -> None:
        report = self._report
        self._fact_labels["cpu"].setText(report.cpu.describe())
        self._fact_labels["ram"].setText(report.memory.describe())
        if report.gpus:
            self._fact_labels["gpu"].setText("\n".join(gpu.describe() for gpu in report.gpus))
        else:
            self._fact_labels["gpu"].setText(tr("No NVIDIA GPU detected"))
        if report.driver.present:
            self._fact_labels["driver"].setText(
                tr(
                    "{version} (supports CUDA up to {cuda})",
                    version=report.driver.version,
                    cuda=report.driver.cuda_version or tr("unknown"),
                )
            )
        else:
            self._fact_labels["driver"].setText(tr("Not installed"))
        self._fact_labels["os"].setText(report.os_description)

        ready = self._status.ready if self._status else None
        if ready and ready.ultralytics:
            self._fact_labels["ultralytics"].setText(
                tr(
                    "{ultralytics} (PyTorch {torch}, {build})",
                    ultralytics=ready.ultralytics,
                    torch=ready.torch,
                    build=ready.cuda_build or tr("CPU build"),
                )
            )
        else:
            self._fact_labels["ultralytics"].setText(tr("Not installed yet"))

    def _paint_rows(self) -> None:
        driver = self._report.driver
        if driver.present:
            self._driver_row.set_state(
                OK,
                tr(
                    "Driver {version} is installed and supports CUDA {cuda}.",
                    version=driver.version,
                    cuda=driver.cuda_version,
                ),
            )
            self._driver_row.action.setVisible(False)
        else:
            self._driver_row.set_state(
                BAD,
                tr(
                    "{reason} Training needs an NVIDIA GPU with a current driver. "
                    'Install one from <a href="{url}">nvidia.com</a>, restart, '
                    "then come back to this screen.",
                    reason=driver.reason,
                    url=DRIVER_HELP_URL,
                ),
            )
            self._driver_row.set_action(tr("Open driver downloads"))

        status = self._status
        plan = self._plan
        ready = status.ready if status else None

        unsupported = venv_manager.missing_kernels(ready)
        if unsupported:
            self._cuda_row.set_state(BAD, unsupported)
        elif ready and ready.cuda_available:
            if ready.devices:
                detail = tr(
                    "PyTorch {torch} ({build}) can use {device}.",
                    torch=ready.torch,
                    build=ready.cuda_build,
                    device=ready.devices[0]["name"],
                )
            else:
                detail = tr(
                    "PyTorch {torch} ({build}) can use the GPU.",
                    torch=ready.torch,
                    build=ready.cuda_build,
                )
            self._cuda_row.set_state(OK, detail)
        elif ready is not None:
            self._cuda_row.set_state(
                BAD,
                tr(
                    "PyTorch is installed but cannot use the GPU. Training would fall back to "
                    "the CPU and take days rather than minutes. Rebuilding with a different "
                    "PyTorch build usually fixes this."
                ),
            )
        elif plan is not None and not plan.backend.is_cuda:
            self._cuda_row.set_state(BAD, plan.reason)
        elif plan is not None:
            self._cuda_row.set_state(
                UNKNOWN,
                tr(
                    "Will be confirmed once the environment is set up. Planned: the "
                    "{backend} build of PyTorch.",
                    backend=tr(plan.backend.label),
                ),
            )

        if status is None:
            return
        state = {
            venv_manager.READY: OK,
            venv_manager.DEGRADED: WARN,
            venv_manager.MISSING: BAD,
            venv_manager.INCOMPLETE: WARN,
            venv_manager.STALE: WARN,
        }.get(status.state, UNKNOWN)
        self._env_row.set_state(state, status.message)
        if status.state == venv_manager.READY:
            self._env_row.action.setVisible(False)
        else:
            self._env_row.set_action(
                tr("Set up now") if status.state == venv_manager.MISSING else tr("Repair"),
                enabled=not self._busy,
            )

    def _paint_buttons(self) -> None:
        busy = self._busy
        usable = bool(self._status and self._status.is_usable)
        self._setup_button.setEnabled(not busy)
        self._setup_button.setText(
            tr("Rebuild environment") if usable else tr("Set up environment")
        )
        self._cancel_button.setEnabled(busy)
        self._updates_button.setEnabled(not busy and usable)
        self._rebuild_button.setEnabled(not busy and bool(self._spec and self._spec.path.exists()))
        self._backend.setEnabled(not busy)

    @property
    def _busy(self) -> bool:
        return self._bootstrap.is_active or self._updater.is_active

    # -- actions ----------------------------------------------------------- #

    def _on_backend_changed(self, _index: int) -> None:
        if not self._busy:
            self.refresh()

    def _start_setup(self) -> None:
        if self._busy or self._spec is None or self._plan is None:
            return
        if self._report.long_paths is False:
            self._console.append_line(tr(env_probe.LONG_PATHS_WARNING), "warning")
        for warning in self._plan.warnings:
            self._console.append_line(warning, "warning")
        self._console.append_line(self._plan.reason, "info")
        self._progress.start()
        self._bootstrap.start(self._spec)

    def _rebuild(self) -> None:
        if self._busy or self._spec is None:
            return
        answer = QMessageBox.question(
            self,
            tr("Remove and rebuild"),
            tr(
                "This deletes the environment at\n{path}\nand builds it again "
                "(about {estimate}, though anything already downloaded is "
                "reused).\n\nYour datasets and trained models are not touched.\n\nContinue?",
                path=self._spec.path,
                estimate=venv_manager.DOWNLOAD_ESTIMATE,
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if not _is_yes(answer):
            return
        venv_manager.remove(self._spec)
        self._console.append_line(tr("Removed {path}", path=self._spec.path), "info")
        self.refresh()
        self._start_setup()

    def _cancel(self) -> None:
        self._bootstrap.cancel()
        self._updater.cancel()

    def _check_updates(self, *, automatic: bool = False) -> None:
        if self._busy or self._status is None or self._status.ready is None:
            return
        if self._update_check is not None and self._update_check.isRunning():
            return
        self._update_check_automatic = automatic
        self._updates_button.setEnabled(False)
        self._console.append_line(tr("Checking pypi.org for a newer Ultralytics..."), "info")
        self._update_check = _UpdateCheckThread(self._status.ready.ultralytics, self)
        self._update_check.checked.connect(self._on_update_checked)
        self._update_check.start()

    def _on_update_checked(self, info: updates.UpdateInfo) -> None:
        self._update_info = info
        self._paint_buttons()
        self._console.append_line(info.summary(), "warning" if info.error else "info")
        if self._busy:
            # Setup or a rebuild started while PyPI was answering; offering to
            # change the environment under it would only collide with it.
            return
        if not info.available or self._spec is None:
            # A launch with no news, or with no network, is not worth a dialog:
            # the console line above is enough. The button is owed an answer.
            if not self._update_check_automatic:
                QMessageBox.information(self, tr("Check for updates"), info.summary())
            return

        answer = QMessageBox.question(
            self,
            tr("Update available"),
            tr(
                "{summary}\n\nShow exactly what would change before installing anything?",
                summary=info.summary(),
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if not _is_yes(answer):
            self._console.append_line(tr("Update declined; nothing was changed."), "info")
            return
        self._pending_update = f"ultralytics=={info.latest}"
        self._updater.preview(self._spec, self._pending_update)

    def _on_update_plan(self, lines: list[str]) -> None:
        if self._spec is None:
            return
        box = QMessageBox(self)
        box.setWindowTitle(tr("Update plan"))
        box.setText(
            tr(
                "This is what the update would change. Check that PyTorch is not being "
                "replaced: a different PyTorch build can remove GPU support."
            )
        )
        # The package list goes in the visible body, not behind "Show Details".
        # Asking someone to check that torch is untouched and then hiding the
        # only place that says so would make the warning decorative.
        changes = [line.strip() for line in lines if line.strip()[:2] in ("- ", "+ ")]
        box.setInformativeText("\n".join(changes[:40]) or tr("Nothing would change."))
        box.setDetailedText("\n".join(lines) or tr("(no output)"))
        box.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        box.button(QMessageBox.StandardButton.Ok).setText(tr("Install"))
        if box.exec() != QMessageBox.StandardButton.Ok:
            self._console.append_line(tr("Update cancelled; nothing was changed."), "info")
            return
        self._progress.start()
        self._updater.apply(self._spec, self._pending_update)

    def details_text(self) -> str:
        """Everything on this page as plain text."""
        lines = [
            f"{self._fact_titles[key]}: {label.text()}".replace("\n", "; ")
            for key, label in self._fact_labels.items()
        ]
        if self._report.long_paths is not None:
            if self._report.long_paths:
                lines.append(tr("Windows long paths: enabled"))
            else:
                lines.append(tr("Windows long paths: DISABLED"))
        if self._plan is not None:
            lines.append(
                tr(
                    "Planned backend: {backend} - {reason}",
                    backend=self._plan.backend.key,
                    reason=self._plan.reason,
                )
            )
        if self._spec is not None:
            lines.append(tr("Environment path: {path}", path=self._spec.path))
        if self._status is not None:
            lines.append(
                tr(
                    "Environment state: {state} - {message}",
                    state=self._status.state,
                    message=self._status.message,
                )
            )
        return "\n".join(lines)

    def _open_driver_help(self) -> None:
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl

        QDesktopServices.openUrl(QUrl(DRIVER_HELP_URL))

    # -- slots ------------------------------------------------------------- #

    def _on_stage(self, key: str, title: str, index: int, count: int) -> None:
        self._console.append_line(f"[{index + 1}/{count}] {title}", "info")
        self._env_row.set_state(BUSY, title)

    def _on_busy(self, busy: bool) -> None:
        self._paint_buttons()
        self._env_row.action.setEnabled(not busy)

    def _on_failed(self, code: str, msg: str, hint: str) -> None:
        self._console.append_line(f"[{code}] {msg}", "error")
        if hint:
            self._console.append_line(hint, "warning")
        self._progress.stop(tr("Failed"))
        QMessageBox.warning(self, tr("Setup failed"), f"{msg}\n\n{hint}".strip())

    def _on_bootstrap_finished(self, ok: bool, status) -> None:
        self._progress.stop(tr("Environment ready") if ok else tr("Stopped"))
        self.refresh()

    def _on_update_finished(self, ok: bool, status) -> None:
        if status is not None:
            self._progress.stop(tr("Updated") if ok else tr("Stopped"))
        self.refresh()

    # -- StepPage ---------------------------------------------------------- #

    def can_enter(self) -> bool:
        return True

    def is_complete(self) -> bool:
        return bool(
            self._report.driver.present and self._status is not None and self._status.is_usable
        )

    def blocked_reason(self) -> str:
        return tr("Finish setting up the environment in Step 1 first.")

    def on_enter(self) -> None:
        if not self._busy:
            self.refresh()

    def on_app_start(self) -> None:
        """Check for a newer Ultralytics once per launch, in the background.

        Quiet unless there is something to offer: an offline machine or an
        up-to-date environment gets a console line, not a dialog. Nothing is
        installed without the same two confirmations the button asks for.
        """
        self._check_updates(automatic=True)

    def is_busy(self) -> bool:
        return self._busy

    def shutdown(self) -> None:
        self._bootstrap.shutdown()
        self._updater.cancel()
        if self._update_check is not None and self._update_check.isRunning():
            self._update_check.wait(5000)


#: Convenience for tests and diagnostics.
def backend_keys() -> list[str]:
    return list(BY_KEY)
