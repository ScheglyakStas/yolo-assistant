"""The frame the worker last sent, drawn to fit.

Deliberately much less than ``image_canvas.py``, which is next door and does
know about boxes, selection and the mouse. Here the boxes are already painted
on -- by Ultralytics, in the worker, on a frame the GUI process has no way to
decode for itself -- so there is nothing to interact with and nothing to
correct. What is left is fitting an image into a rectangle without distorting
it, and saying something useful when there is no image yet.

The one behaviour worth naming is that a frame is never cleared by the end of a
run. When a video finishes or a camera stops, the last frame stays on screen:
it is the answer the user was waiting for, and blanking it at the moment the
job ends would hide the result at exactly the wrong moment.
"""

from __future__ import annotations

import base64

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ...core.i18n import tr

BACKGROUND = QColor("#1f2124")
TEXT_COLOUR = QColor("#ecf0f1")


class PreviewCanvas(QWidget):
    """Shows one frame at a time, letterboxed."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._message = tr("Nothing has been run yet.")
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(False)

    # -- contents ---------------------------------------------------------- #

    def set_frame(self, image: QImage) -> None:
        if image is None or image.isNull():
            return
        self._image = image
        self.update()

    def set_message(self, text: str) -> None:
        """Replace whatever is on screen with a sentence."""
        self._image = None
        self._message = text
        self.update()

    def set_idle_message(self, text: str) -> None:
        """Change what is shown *if* there is no frame, keeping any there is."""
        self._message = text
        if self._image is None:
            self.update()

    @property
    def has_frame(self) -> bool:
        return self._image is not None and not self._image.isNull()

    @property
    def frame(self) -> QImage | None:
        return self._image

    # -- painting ---------------------------------------------------------- #

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        if not self.has_frame:
            painter.setPen(QPen(TEXT_COLOUR))
            painter.drawText(
                QRectF(self.rect()).adjusted(24, 24, -24, -24),
                int(Qt.AlignmentFlag.AlignCenter) | int(Qt.TextFlag.TextWordWrap),
                self._message,
            )
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(self._target_rect(), self._image)

    def _target_rect(self) -> QRectF:
        area = QRectF(self.rect())
        size = self._image.size()
        if size.width() <= 0 or size.height() <= 0:
            return area
        scale = min(area.width() / size.width(), area.height() / size.height())
        width, height = size.width() * scale, size.height() * scale
        return QRectF(
            area.x() + (area.width() - width) / 2,
            area.y() + (area.height() - height) / 2,
            width,
            height,
        )


def image_from_payload(data: dict) -> QImage | None:
    """Decode one ``preview`` message into an image, or ``None``.

    Returns ``None`` rather than raising on anything malformed. This runs once
    per displayed frame off a socket, and a single corrupt frame should cost one
    dropped picture rather than the run.
    """
    encoded = data.get("image")
    if not isinstance(encoded, str) or not encoded:
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return None
    image = QImage()
    if not image.loadFromData(raw):
        return None
    return image
