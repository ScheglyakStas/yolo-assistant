"""Step 7: point the exported model at something and watch what it does.

This is the step that answers the question the other six only promise. A
training run that reached mAP50-95 of 0.82 and an export that wrote a file
without complaining are both claims about a process; a box drawn around the
right object, at eight milliseconds a frame, is the thing the user came for.

Three things follow for this page.

**The picture is the output, so the picture gets the room.** The controls are a
narrow column and everything else is the frame. That is also why the last frame
is never cleared when a run ends: it is the answer, and blanking it at the
moment the job finishes would hide the result at exactly the wrong time.

**The numbers next to it have to be honest about what they measure.** Getting a
model ready takes seconds and every frame afterwards takes milliseconds, so the
two are reported separately and never averaged together. The rate shown is the
end-to-end one, decoding included, because that is what the user is watching --
and because inference time alone invites "8 ms, so 125 frames a second", which
is never true.

**A video is watched afterwards, not during.** On a fast card the frames come
out many times faster than they were filmed, and showing them as they arrive is
the video on fast-forward. So while a video runs the picture says it is being
worked on, and when it finishes the annotated copy the worker wrote plays at its
own frame rate, in a player that can be paused and scrubbed.

**The annotated copy is temporary unless it is saved.** A photo or video run
leaves a copy with the boxes drawn on in the project's ``temp/`` folder, and a
video's can be gigabytes. It lasts exactly as long as it is the result on
screen: choosing another file, starting another run, switching project or
closing the app deletes it. Save As copies it somewhere of the user's choosing,
and nothing outside ``temp/`` is ever deleted.

**Leaving the page stops the run.** Every other step keeps working when the
user navigates away, because every other step is producing something that would
be lost. This one produces nothing but a picture nobody is looking at any more,
and a camera left running holds the device, the GPU and the indicator light for
no reason at all.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...core import export as export_model
from ...core import inference as inference_model
from ...core import project as project_model
from ...core import runtime
from ...core.i18n import N_, tr, trn
from ...core.inference import CAMERA, DEVICES, MODES, PHOTO, VIDEO, TestPlan
from ...core.job import JobRunner, JobState
from ...core.project import Project
from ..common import video_player
from ..common.log_console import LogConsole
from ..common.param_form import ParamForm, apply_help
from ..common.preview_canvas import PreviewCanvas, image_from_payload
from ..common.progress_header import ProgressHeader
from ..common.speed_panel import SpeedPanel
from ..common.telemetry_panel import TelemetryPanel
from .base import StepPage

#: The controls column. Wide enough for a file name, narrow enough that the
#: frame gets the rest of the window.
CONTROLS_WIDTH = 430


class InferenceStep(StepPage):
    title = N_("Testing")
    subtitle = N_(
        "Run the exported model on a photo, a video or a camera, and see what it "
        "finds and how long it takes per frame."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._runner = JobRunner(self)
        #: A second runner, because looking for cameras is its own short job and
        #: has to be possible while nothing else is running.
        self._scanner = JobRunner(self)

        self._project: Project | None = None
        self._exports: tuple = ()
        self._cameras: tuple = ()
        self._scanned = False
        self._plan = TestPlan()
        self._settings: dict = {}
        self._adopted: str | None = None
        self._ran = False
        self._failure = ""
        #: Warnings the last run raised that belong beside the picture rather
        #: than only in the log, and what that run was, so they are not shown
        #: against settings they no longer describe.
        self._run_warnings: list[str] = []
        self._run_key: tuple = ()
        #: The last file chosen for each of the two file modes, so that moving
        #: between them does not lose either.
        self._sources: dict[str, str] = {PHOTO: "", VIDEO: ""}
        #: The annotated copy the last run left in ``temp/``, and what it was
        #: made from. Deleted as soon as it stops being the result on screen.
        self._output: Path | None = None
        self._output_source = ""
        #: Where the last Save As went, so the next one starts there.
        self._save_dir = ""

        self._build_ui()
        self._wire()
        self._adopt_project()

    # -- construction ------------------------------------------------------ #

    def _build_ui(self) -> None:
        heading = QLabel(f"<h2>{tr(self.title)}</h2>")
        subtitle = QLabel(tr(self.subtitle))
        subtitle.setWordWrap(True)

        settings_box = QGroupBox(tr("What to test"))
        form = ParamForm(settings_box)

        self._model = QComboBox()
        _fill_width(self._model)
        form.add_row("test.model", tr("Model"), self._model)

        self._device = QComboBox()
        for device in DEVICES:
            self._device.addItem(device.caption, device.key)
            self._device.setItemData(
                self._device.count() - 1, tr(device.note), Qt.ItemDataRole.ToolTipRole
            )
        form.add_row("test.device", tr("Run on"), self._device)

        self._mode = QComboBox()
        for mode in MODES:
            self._mode.addItem(mode.caption, mode.key)
            self._mode.setItemData(
                self._mode.count() - 1, tr(mode.note), Qt.ItemDataRole.ToolTipRole
            )
        form.add_row("test.mode", tr("Test with"), self._mode)

        form.add_row("test.source", tr("Run against"), self._build_source())

        # Its own row, shown only for a camera: a size and a rate offered beside
        # a photo would read as settings for the photo.
        self._form = form
        self._camera_mode_row = form.add_row(
            "test.camera_mode", tr("Camera mode"), self._build_camera_mode()
        )

        self._confidence = QDoubleSpinBox()
        self._confidence.setRange(
            inference_model.MIN_CONFIDENCE, inference_model.MAX_CONFIDENCE
        )
        self._confidence.setSingleStep(0.05)
        self._confidence.setDecimals(2)
        self._confidence.setValue(inference_model.DEFAULT_CONFIDENCE)
        form.add_row("test.confidence", tr("Confidence needed"), self._confidence)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        # Escaped but usually tagless, which auto-detection would show as-is.
        self._summary.setTextFormat(Qt.TextFormat.RichText)
        self._problems = QLabel("")
        self._problems.setWordWrap(True)
        self._problems.setVisible(False)
        self._warnings = QLabel("")
        self._warnings.setWordWrap(True)
        self._warnings.setVisible(False)

        self._start_button = QPushButton(tr("Start"))
        apply_help(self._start_button, "test.start")
        self._stop_button = QPushButton(tr("Stop"))
        self._stop_button.setEnabled(False)
        self._save_button = QPushButton(tr("Save As..."))
        apply_help(self._save_button, "test.save")
        self._save_button.setEnabled(False)
        self._saved = QLabel("")
        self._saved.setWordWrap(True)
        self._saved.setVisible(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self._start_button)
        buttons.addWidget(self._stop_button)
        buttons.addStretch(1)
        buttons.addWidget(self._save_button)

        self._progress = ProgressHeader()
        self._speed = SpeedPanel()
        self._telemetry = TelemetryPanel()

        controls = QWidget()
        controls.setMaximumWidth(CONTROLS_WIDTH)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(self._summary)
        controls_layout.addWidget(settings_box)
        controls_layout.addWidget(self._problems)
        controls_layout.addWidget(self._warnings)
        controls_layout.addLayout(buttons)
        controls_layout.addWidget(self._saved)
        controls_layout.addWidget(self._progress)
        controls_layout.addWidget(self._speed)
        controls_layout.addWidget(self._telemetry)
        controls_layout.addStretch(1)

        self._canvas = PreviewCanvas()
        self._player = video_player.VideoPlayer() if video_player.AVAILABLE else None
        self._picture = QStackedWidget()
        self._picture.addWidget(self._canvas)
        if self._player is not None:
            self._picture.addWidget(self._player)
        self._found = QLabel("")
        self._found.setWordWrap(True)
        self._console = LogConsole()

        viewer = QWidget()
        viewer_layout = QVBoxLayout(viewer)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.addWidget(self._picture, 1)
        viewer_layout.addWidget(self._found)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(viewer)
        right.addWidget(self._console)
        right.setStretchFactor(0, 4)
        right.setStretchFactor(1, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(subtitle)
        layout.addWidget(splitter, 1)

    def _build_source(self) -> QWidget:
        """One row that is a file picker or a camera picker, never both.

        A stack rather than two rows that appear and disappear: the form stays
        the same height whichever mode is chosen, so switching modes does not
        move every control below it.
        """
        self._source_stack = QStackedWidget()

        file_row = QWidget()
        file_layout = QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self._source = QLineEdit()
        self._source.setReadOnly(True)
        self._source.setPlaceholderText(tr("Choose a photo..."))
        apply_help(self._source, "test.source")
        self._browse_button = QPushButton(tr("Choose..."))
        apply_help(self._browse_button, "test.source")
        file_layout.addWidget(self._source, 1)
        file_layout.addWidget(self._browse_button)

        camera_row = QWidget()
        camera_layout = QHBoxLayout(camera_row)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        self._camera = QComboBox()
        _fill_width(self._camera)
        apply_help(self._camera, "test.camera")
        self._refresh_button = QPushButton(tr("Refresh"))
        apply_help(self._refresh_button, "test.refresh")
        camera_layout.addWidget(self._camera, 1)
        camera_layout.addWidget(self._refresh_button)

        self._source_stack.addWidget(file_row)
        self._source_stack.addWidget(camera_row)
        return self._source_stack

    def _build_camera_mode(self) -> QWidget:
        """The size and the frame rate a camera is asked for, side by side."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        self._camera_size = QComboBox()
        # Keyed by the "WxH" text: Qt compares a string reliably, and a list
        # handed to ``findData`` is not one.
        for size in inference_model.CAMERA_SIZES:
            caption = inference_model.size_caption(size)
            self._camera_size.addItem(caption, caption)
        self._set_data(
            self._camera_size, inference_model.size_caption(inference_model.DEFAULT_CAMERA_SIZE)
        )
        apply_help(self._camera_size, "test.camera_mode")

        self._camera_fps = QComboBox()
        for fps in inference_model.CAMERA_RATES:
            self._camera_fps.addItem(inference_model.rate_caption(fps), fps)
        self._set_data(self._camera_fps, inference_model.DEFAULT_CAMERA_FPS)
        apply_help(self._camera_fps, "test.camera_mode")

        layout.addWidget(self._camera_size, 1)
        layout.addWidget(self._camera_fps, 1)
        return row

    def _wire(self) -> None:
        self._start_button.clicked.connect(self._start)
        self._stop_button.clicked.connect(self._runner.cancel)
        self._save_button.clicked.connect(self._save_output)
        self._browse_button.clicked.connect(self._choose_source)
        self._refresh_button.clicked.connect(self._scan_cameras)

        self._model.currentIndexChanged.connect(lambda _index: self.refresh())
        self._device.currentIndexChanged.connect(lambda _index: self.refresh())
        self._mode.currentIndexChanged.connect(self._on_mode_changed)
        self._camera.currentIndexChanged.connect(lambda _index: self.refresh())
        self._camera_size.currentIndexChanged.connect(lambda _index: self.refresh())
        self._camera_fps.currentIndexChanged.connect(lambda _index: self.refresh())
        self._confidence.valueChanged.connect(lambda _value: self.refresh())

        self._runner.state_changed.connect(self._on_state)
        self._runner.progress.connect(self._progress.update_progress)
        self._runner.preview.connect(self._on_preview)
        self._runner.log_line.connect(self._console.append_line)
        self._runner.warned.connect(self._on_warned)
        self._runner.failed.connect(self._on_failed)
        self._runner.finished.connect(self._on_finished)
        self._runner.stalled.connect(
            lambda quiet: self._console.append_line(
                tr(
                    "No output for {quiet:.0f}s. A camera that has been unplugged can "
                    "leave the read waiting; press Stop.",
                    quiet=quiet,
                ),
                "warning",
            )
        )

        self._scanner.log_line.connect(self._console.append_line)
        self._scanner.finished.connect(self._on_cameras_found)
        self._scanner.failed.connect(
            lambda code, msg, hint, tb: self._console.append_line(f"[{code}] {msg}", "error")
        )

    # -- state ------------------------------------------------------------- #

    def _adopt_project(self) -> None:
        project = project_model.current()

        root = str(project.root) if project else None
        if root == self._adopted:
            self._project = project
            self._rescan()
            return
        # The copy belongs to the project being left, so it goes with it.
        self._discard_output()
        self._project = project
        self._adopted = root
        # Copies a crashed or killed session never got to delete -- or one a
        # run still finishing when the app closed wrote after it had gone.
        inference_model.sweep_temp(self._project)
        self._failure = ""
        # A different project means a different camera is no more or less
        # likely, but the files certainly are: forget the last frame.
        self._show_canvas()
        self._canvas.set_message(tr("Nothing has been run yet."))
        self._speed.clear()
        self._found.setText("")

        self._settings = inference_model.read_settings(self._project)
        self._sources = {
            PHOTO: self._settings["photo"],
            VIDEO: self._settings["video"],
        }
        self._set_data(self._device, self._settings["device"])
        self._set_data(self._mode, self._settings["mode"])
        self._set_data(
            self._camera_size, inference_model.size_caption(self._settings["camera_size"])
        )
        self._set_data(self._camera_fps, self._settings["camera_fps"])
        self._paint_source_row()
        self._confidence.blockSignals(True)
        self._confidence.setValue(self._settings["conf"])
        self._confidence.blockSignals(False)

        self._rescan()
        # Not only when the mode is *changed*: a page that comes back up
        # already in camera mode -- because that is what it was left in
        # yesterday -- changes nothing, fires no signal, and would otherwise
        # sit in front of an empty list saying no camera was found.
        self._maybe_scan_cameras()

    def _rescan(self) -> None:
        """Re-read the exports folder. Files appear and disappear between visits."""
        self._exports = export_model.list_exports(self._project)
        self._repopulate_models()
        self.refresh()
        self.gates_changed.emit()

    def _repopulate_models(self) -> None:
        wanted = str(self._model.currentData() or self._settings.get("model") or "")
        available = inference_model.targets(self._project, self._exports)

        self._model.blockSignals(True)
        self._model.clear()
        for target in available:
            self._model.addItem(target.caption(), target.key)
            self._model.setItemData(
                self._model.count() - 1,
                tr(target.runtime.note) if target.runtime.note else "",
                Qt.ItemDataRole.ToolTipRole,
            )
        index = self._model.findData(wanted)
        self._model.setCurrentIndex(max(0, index))
        self._model.blockSignals(False)

    def _repopulate_cameras(self) -> None:
        wanted = self._camera.currentData()
        if wanted is None:
            wanted = self._settings.get("camera", -1)

        self._camera.blockSignals(True)
        self._camera.clear()
        for camera in self._cameras:
            self._camera.addItem(camera.caption(), camera.index)
        index = self._camera.findData(wanted)
        self._camera.setCurrentIndex(max(0, index))
        self._camera.blockSignals(False)

    def refresh(self) -> None:
        self._plan = inference_model.build_plan(
            project=self._project,
            exports=self._exports,
            target_key=str(self._model.currentData() or ""),
            device=str(self._device.currentData() or inference_model.DEFAULT_DEVICE),
            mode=self._current_mode(),
            source=self._sources.get(self._current_mode(), ""),
            camera_index=_as_int(self._camera.currentData(), -1),
            cameras=self._cameras,
            camera_size=_as_size(self._camera_size.currentData()),
            camera_fps=_as_int(self._camera_fps.currentData(), 0),
            confidence=self._confidence.value(),
        )
        self._paint_notes()
        self._paint_buttons()

    def _current_mode(self) -> str:
        return str(self._mode.currentData() or inference_model.DEFAULT_MODE)

    def _paint_notes(self) -> None:
        self._summary.setText(_escape(self._plan.summary()))
        _paint_lines(self._problems, self._plan.problems(), "#c0392b", "x")
        warnings = self._plan.warnings()
        if self._run_warnings and _run_key(self._plan) == self._run_key:
            warnings += self._run_warnings
        _paint_lines(self._warnings, warnings, "#b9770e", "!")

    def _paint_buttons(self) -> None:
        busy = self._busy
        self._start_button.setEnabled(not busy and self._plan.is_runnable)
        self._start_button.setText(tr("Running...") if busy else tr("Start"))
        self._stop_button.setEnabled(busy)
        self._save_button.setEnabled(
            not busy and self._output is not None and self._output.is_file()
        )
        self._refresh_button.setEnabled(not busy and not self._scanner.is_active())
        for widget in (
            self._model,
            self._device,
            self._mode,
            self._camera,
            self._camera_size,
            self._camera_fps,
            self._browse_button,
            self._confidence,
        ):
            widget.setEnabled(not busy)

    def _paint_source_row(self) -> None:
        mode = self._current_mode()
        self._source_stack.setCurrentIndex(1 if mode == CAMERA else 0)
        self._form.setRowVisible(self._camera_mode_row, mode == CAMERA)
        self._source.setPlaceholderText(
            tr("Choose a video...") if mode == VIDEO else tr("Choose a photo...")
        )
        self._source.setText(self._sources.get(mode, ""))

    @property
    def _busy(self) -> bool:
        return self._runner.is_active()

    # -- actions ----------------------------------------------------------- #

    def _on_mode_changed(self, _index: int) -> None:
        self._paint_source_row()
        self._maybe_scan_cameras()
        self.refresh()

    def _maybe_scan_cameras(self) -> None:
        """Look for cameras the first time this session needs a list of them."""
        if self._current_mode() == CAMERA and not self._scanned:
            self._scan_cameras()

    def _choose_source(self) -> None:
        mode = self._current_mode()
        previous = self._sources.get(mode, "")
        start_in = str(Path(previous).parent) if previous else ""
        chosen, _selected = QFileDialog.getOpenFileName(
            self,
            tr("Choose a video") if mode == VIDEO else tr("Choose a photo"),
            start_in,
            inference_model.file_filter(mode),
        )
        if not chosen:
            return
        # The same file chosen again is not a different source: the result stays.
        if not _same_file(chosen, self._output_source) and self._discard_output():
            self._canvas.set_message(tr("Press Start to run the model on the new file."))
            self._found.setText("")
        self._sources[mode] = chosen
        self._source.setText(chosen)
        self.refresh()

    def _scan_cameras(self) -> None:
        """Ask the worker what cameras answer. Never while a test is running.

        A camera this app is already reading from cannot be opened a second
        time, so scanning mid-run would report the camera in use as absent and
        empty the list the user is watching.
        """
        if self._busy or self._scanner.is_active():
            return
        try:
            spec = runtime.job_spec(
                "cameras", {}, project=self._project, log_name="cameras"
            )
        except runtime.EnvironmentMissing as exc:
            self._console.append_line(str(exc), "error")
            return
        self._console.append_line(tr("Looking for cameras..."), "info")
        self._scanned = True
        self._scanner.start(spec)
        self._paint_buttons()

    def _start(self) -> None:
        if self._busy or self._project is None or not self._plan.is_runnable:
            return

        self._console.clear()
        self._console.append_line(runtime.describe_python(), "info")
        self._failure = ""
        self._run_warnings = []
        self._run_key = _run_key(self._plan)
        self._paint_notes()
        self._speed.clear()
        self._found.setText("")
        # Stops the player before the file under it is deleted, which on
        # Windows is the difference between deleting it and not.
        self._show_canvas()
        self._discard_output()
        self._canvas.set_message(tr("Getting the model ready..."))
        self._progress.start()

        try:
            spec = runtime.job_spec(
                "infer",
                self._plan.job(self._project),
                project=self._project,
                log_name="test",
            )
        except runtime.EnvironmentMissing as exc:
            self._progress.stop(tr("Not started"))
            QMessageBox.warning(self, tr("Environment not ready"), str(exc))
            return

        inference_model.write_settings(self._project, self._plan, previous=self._settings)
        self._settings = inference_model.read_settings(self._project)
        self._runner.start(spec)
        self._paint_buttons()

    # -- slots ------------------------------------------------------------- #

    def _on_state(self, state: JobState) -> None:
        self._paint_buttons()
        if state is JobState.CANCELLING:
            self._progress.stop(tr("Stopping..."))

    def _on_preview(self, data: dict) -> None:
        if self._plan.mode == VIDEO:
            # Not drawn: see the module docstring. The numbers still are.
            self._canvas.set_message(
                tr(
                    "Running the model on every frame. The video, with its boxes, "
                    "plays here at its own speed once that is done."
                )
            )
        else:
            image = image_from_payload(data)
            if image is not None:
                self._canvas.set_frame(image)
        self._speed.update_stats(data.get("stats") or {}, data.get("timings") or {})
        self._found.setText(_describe(data.get("detections") or []))

    def _on_warned(self, code: str, msg: str, hint: str) -> None:
        self._console.append_line(msg, "warning")
        if hint:
            self._console.append_line(hint, "warning")
        # The one warning that changes what the numbers mean rather than adding
        # to them: see ``_check_onnx_provider`` in the worker.
        if code == inference_model.ONNX_ON_CPU_WARNING:
            self._speed.set_footnote(
                tr("These are processor timings: onnxruntime has no CUDA provider here.")
            )
        # The one that is about the export rather than this machine, and whose
        # remedy is a step back -- so it stays on screen, not only in the log.
        if code == inference_model.ASPECT_MISMATCH_WARNING and msg not in self._run_warnings:
            self._run_warnings.append(msg)
            self._paint_notes()

    def _on_failed(self, code: str, msg: str, hint: str, traceback: str) -> None:
        self._console.append_line(f"[{code}] {msg}", "error")
        if hint:
            self._console.append_line(hint, "warning")
        self._failure = f"{msg} {hint}".strip()
        self._canvas.set_message(msg)
        title = {
            "camera_unavailable": tr("That camera cannot be opened"),
            "source_unreadable": tr("That file cannot be read"),
            "no_cuda_runtime": tr("No GPU available"),
        }.get(code) or tr("The test could not run")
        QMessageBox.warning(self, title, f"{msg}\n\n{hint}".strip())

    def _on_finished(self, ok: bool, payload: dict) -> None:
        if ok:
            self._ran = True
            self._progress.stop(_ending(payload))
            self._speed.set_footnote(_startup_note(payload))
        elif payload.get("cancelled"):
            self._progress.stop(tr("Stopped"))
        else:
            self._progress.stop(tr("Failed"))
        if ok:
            self._keep_output(payload)
        if ok and payload.get("mode") == VIDEO:
            self._play_recording(str(payload.get("output") or ""))
        # The last frame stays where it is: it is the result.
        self._canvas.set_idle_message(self._failure or tr("Nothing has been run yet."))
        self._paint_buttons()
        self.gates_changed.emit()

    def _play_recording(self, output: str) -> None:
        path = Path(output) if output else None
        if path is None or not path.is_file():
            self._canvas.set_message(tr("No frames were read, so there is nothing to play."))
            return
        self._found.setText(
            tr(
                "This copy is temporary: it is deleted when you choose another file or "
                "close the app. Press Save As... to keep it."
            )
        )
        if self._player is None:
            self._canvas.set_message(
                tr(
                    "The video with its boxes was written to {path}. This build of Qt "
                    "cannot play it here; press Save As... to keep it and open it in any "
                    "video player.",
                    path=path,
                )
            )
            return
        self._picture.setCurrentWidget(self._player)
        self._player.play_file(path)

    def _keep_output(self, payload: dict) -> None:
        """Remember the copy a finished run wrote, so it can be saved or deleted."""
        raw = str(payload.get("output") or "").strip()
        path = Path(raw) if raw else None
        self._output = path if path is not None and path.is_file() else None
        self._output_source = str(payload.get("source") or "")

    def _discard_output(self) -> bool:
        """Delete the last run's temporary copy. Returns whether there was one.

        Only ever a file inside ``temp/`` -- :func:`inference_model.discard`
        checks -- so a copy the user saved elsewhere is safe whatever happens
        here.
        """
        output = self._output
        if output is None:
            return False
        if self._player is not None and self._player.path is not None:
            self._show_canvas()
        inference_model.discard(output, self._project)
        self._output = None
        self._output_source = ""
        self._saved.setVisible(False)
        self._paint_buttons()
        return True

    def _save_output(self) -> None:
        """Copy the annotated result somewhere the app will never delete it."""
        output = self._output
        if output is None or not output.is_file():
            return
        start_in = self._save_dir or (
            str(Path(self._output_source).parent) if self._output_source else ""
        )
        suggested = str(Path(start_in) / output.name) if start_in else output.name
        chosen, _selected = QFileDialog.getSaveFileName(
            self, tr("Save the result as"), suggested, inference_model.save_filter(output)
        )
        if not chosen:
            return
        if inference_model.is_temporary(chosen, self._project):
            QMessageBox.warning(
                self,
                tr("Choose another folder"),
                tr(
                    "{folder} is the app's temporary folder, and everything in it is "
                    "deleted automatically. Save the file somewhere else to keep it.",
                    folder=inference_model.temp_dir(self._project),
                ),
            )
            return
        try:
            saved = inference_model.save_copy(output, chosen)
        except OSError as exc:
            QMessageBox.warning(self, tr("The file could not be saved"), str(exc))
            return
        self._save_dir = str(saved.parent)
        self._console.append_line(tr("Saved to {path}.", path=saved), "info")
        self._saved.setText(tr("Saved to {path}", path=_escape(str(saved))))
        self._saved.setVisible(True)

    def _show_canvas(self) -> None:
        """Back to the frame view, letting go of any video that was playing."""
        if self._player is not None:
            self._player.stop()
        self._picture.setCurrentWidget(self._canvas)

    def _on_cameras_found(self, ok: bool, payload: dict) -> None:
        self._cameras = inference_model.cameras_from(payload) if ok else ()
        self._repopulate_cameras()
        if ok:
            self._console.append_line(
                trn("{n} camera found.", "{n} cameras found.", len(self._cameras))
                if self._cameras
                else tr("No camera answered."),
                "info",
            )
        self.refresh()

    # -- StepPage ---------------------------------------------------------- #

    def can_enter(self) -> bool:
        if runtime.worker_python() is None:
            return False
        project = project_model.current()
        if project is None or not project.prepared_file.is_file():
            return False
        return export_model.has_export(project)

    def is_complete(self) -> bool:
        """Whether a test has been run in this session.

        The one place the app's "derive state from the filesystem" rule cannot
        apply, because testing deliberately writes nothing: there is no file
        whose existence would mean "this model has been tried". Nothing is
        gated on the answer -- this is the last step -- so a session fact is
        the honest one to show.
        """
        return self._ran

    def blocked_reason(self) -> str:
        if runtime.worker_python() is None:
            return tr(
                "Step 1 has to finish setting up the Python environment before a "
                "model can be run."
            )
        project = project_model.current()
        if project is None or not project.prepared_file.is_file():
            return tr("Prepare your photos in Step 2 first; there is nothing to test yet.")
        return tr(
            "There is no exported model to test. Export one in Step 6, or copy an "
            "exported file into the project's exports folder."
        )

    def on_enter(self) -> None:
        if self._busy:
            return
        self._adopt_project()

    def on_leave(self) -> None:
        """Stop the run. See the module docstring for why this step is different."""
        if self._player is not None:
            self._player.pause()
        if self._runner.is_active():
            self._console.append_line(
                tr("Stopped: nothing is tested while this page is not on screen."), "info"
            )
            self._runner.cancel()

    def is_busy(self) -> bool:
        return self._busy

    def shutdown(self) -> None:
        for runner in (self._runner, self._scanner):
            if runner.is_active():
                runner.cancel()
        if self._player is not None:
            self._player.stop()
        self._discard_output()
        self._telemetry.shutdown()

    # -- internals --------------------------------------------------------- #

    @staticmethod
    def _set_data(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)


# --------------------------------------------------------------------------- #
# Wording
# --------------------------------------------------------------------------- #


def _describe(detections: list) -> str:
    """The boxes as one line, strongest first.

    The frame already shows them. This says how many there are without counting
    rectangles, and names the ones the model was least sure about, which is
    where a threshold worth changing usually shows itself.
    """
    if not detections:
        return tr("Nothing found in this frame.")
    named = ", ".join(
        f"{item.get('name', '?')} {float(item.get('conf', 0.0)):.2f}"
        for item in detections[:8]
    )
    more = len(detections) - 8
    if more > 0:
        named = tr("{names}, and {more} more", names=named, more=more)
    total = trn("{n} object", "{n} objects", len(detections))
    return f"{total}: {named}"


def _ending(payload: dict) -> str:
    frames = int(payload.get("frames") or 0)
    mean = float(payload.get("inference_ms_mean") or 0.0)
    if payload.get("mode") == PHOTO:
        return tr("Done - {mean:.1f} ms", mean=mean) if mean else tr("Done")
    # Both English forms are the same on purpose; Polish needs its own three.
    if payload.get("stopped"):
        return trn(
            "Stopped after {n:,} frames, {mean:.1f} ms each",
            "Stopped after {n:,} frames, {mean:.1f} ms each",
            frames,
            mean=mean,
        )
    return trn(
        "Finished - {n:,} frames, {mean:.1f} ms each",
        "Finished - {n:,} frames, {mean:.1f} ms each",
        frames,
        mean=mean,
    )


def _startup_note(payload: dict) -> str:
    startup = float(payload.get("startup_ms") or 0.0)
    if startup <= 0:
        return ""
    return tr(
        "Getting the model ready took {seconds:.1f} s. That is paid once, "
        "not per frame, and is not counted above.",
        seconds=startup / 1000,
    )


def _paint_lines(label: QLabel, lines: list[str], colour: str, marker: str) -> None:
    if not lines:
        label.setVisible(False)
        return
    joined = f"<br>{marker} ".join(_escape(line) for line in lines)
    label.setText(f"<span style='color:{colour}'>{marker} {joined}</span>")
    label.setVisible(True)


def _run_key(plan) -> tuple:
    """What decides the shapes a run compares: which model, against what input."""
    target = plan.target
    return (
        target.key if target is not None else "",
        plan.mode,
        str(plan.source),
        plan.camera_index,
        tuple(plan.camera_size),
    )


def _fill_width(combo: QComboBox) -> None:
    """Let a combo take the width of its row rather than of its contents.

    Both of these are filled in after the page is first shown -- one from the
    exports folder, the other from a job that goes and looks for cameras -- and
    a combo left to size itself to its contents therefore sizes itself to being
    empty, and stays that width with a file name elided down to "live_bes".
    """
    combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)
    combo.setMinimumContentsLength(18)


def _same_file(a: str | Path, b: str | Path) -> bool:
    if not str(a).strip() or not str(b).strip():
        return False
    try:
        return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
    except OSError:
        return False


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_size(value) -> tuple[int, int]:
    """``"1280x720"`` as ``(1280, 720)``; the default for anything else."""
    try:
        width, height = (int(part) for part in str(value).split("x"))
    except (TypeError, ValueError):
        return inference_model.DEFAULT_CAMERA_SIZE
    return (width, height)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
