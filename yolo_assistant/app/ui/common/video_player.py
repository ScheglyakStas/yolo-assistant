"""The annotated copy of a video, played back at the speed it was filmed.

Step 7 runs a video as fast as the model allows, which on a fast GPU is ten
times the speed it was recorded at -- so the frames are not shown as they are
produced. The worker writes them to a file at the source's own frame rate, and
this widget plays that file once the run is over. Decoding it is Qt's job, not
ours: the GUI process has no OpenCV, and a player is the one place it does not
need one.

Qt Multimedia ships with PySide6 but is imported defensively all the same. A
build without it should cost the playback, not the page: :data:`AVAILABLE` says
which, and the page says where the file is instead.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QLoggingCategory, Qt, QUrl
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from ...core.i18n import tr

# Decoded in software. The only thing this player opens is the worker's
# MPEG-4 Part 2 copy, which a CPU decodes with ease, and the GPU decoders cap
# that codec at 2048 pixels a side: on a 4K video Qt would try NVDEC first and
# FFmpeg would print two errors about it before falling back anyway. An empty
# list of device types is Qt's own switch for "no hardware decoding"; a value
# already in the environment is someone asking for it, and is kept. Qt then
# says, at info level, that it found no hardware decoder -- which is the point.
os.environ.setdefault("QT_FFMPEG_DECODING_HW_DEVICE_TYPES", "")
QLoggingCategory.setFilterRules("qt.multimedia.playbackengine.codec.info=false")

try:
    from PySide6.QtMultimedia import QMediaPlayer, QVideoSink
    from PySide6.QtMultimediaWidgets import QVideoWidget
except ImportError:  # pragma: no cover - PySide6-Essentials, or a broken install
    QMediaPlayer = QVideoSink = QVideoWidget = None

AVAILABLE = QMediaPlayer is not None


class VideoPlayer(QWidget):
    """A video, a play button and a position slider. Nothing else."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path: Path | None = None
        self._duration = 0

        self._video = QVideoWidget()
        self._player = QMediaPlayer(self)
        # Not straight into the widget. At the end of the file Qt sends one
        # empty frame, and the widget has nothing to draw from it: on X11 the
        # picture goes black, on Wayland it cycles through the last few frames
        # still in its buffers. Passing on only real frames keeps the last one
        # on screen, which is what a finished video should look like.
        self._sink = QVideoSink(self)
        self._sink.videoFrameChanged.connect(self._on_frame)
        self._player.setVideoSink(self._sink)

        self._play_button = QPushButton(tr("Pause"))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._time = QLabel(_clock(0, 0))

        controls = QHBoxLayout()
        controls.addWidget(self._play_button)
        controls.addWidget(self._slider, 1)
        controls.addWidget(self._time)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._video, 1)
        layout.addLayout(controls)

        self._play_button.clicked.connect(self._toggle)
        # Moved by the user, not by playback: ``sliderMoved`` only fires for a
        # drag, so the position updates below cannot feed back into a seek.
        self._slider.sliderMoved.connect(self._player.setPosition)
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._paint_button)

    # -- contents ---------------------------------------------------------- #

    def play_file(self, path: Path) -> None:
        self._path = Path(path)
        self._player.setSource(QUrl.fromLocalFile(str(self._path)))
        self._player.play()

    def pause(self) -> None:
        self._player.pause()

    def stop(self) -> None:
        """Stop and let go of the file, so the next run can overwrite it."""
        self._player.stop()
        self._player.setSource(QUrl())
        self._path = None

    @property
    def path(self) -> Path | None:
        return self._path

    # -- slots ------------------------------------------------------------- #

    def _toggle(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
            return
        # Pressing play on a finished video means "again", not "nothing".
        if self._duration and self._player.position() >= self._duration:
            self._player.setPosition(0)
        self._player.play()

    def _on_frame(self, frame) -> None:
        if frame.isValid():
            self._video.videoSink().setVideoFrame(frame)

    def _on_position(self, position: int) -> None:
        if not self._slider.isSliderDown():
            self._slider.setValue(position)
        self._time.setText(_clock(position, self._duration))

    def _on_duration(self, duration: int) -> None:
        self._duration = int(duration)
        self._slider.setRange(0, self._duration)
        self._time.setText(_clock(self._player.position(), self._duration))

    def _paint_button(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_button.setText(tr("Pause") if playing else tr("Play"))


def _clock(position_ms: int, duration_ms: int) -> str:
    return f"{_mmss(position_ms)} / {_mmss(duration_ms)}"


def _mmss(ms: int) -> str:
    seconds = max(0, int(ms) // 1000)
    return f"{seconds // 60}:{seconds % 60:02d}"
