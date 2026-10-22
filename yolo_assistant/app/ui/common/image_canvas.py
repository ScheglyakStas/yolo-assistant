"""A photo with its boxes drawn on it, and the three ways to change them.

The widget owns the geometry and nothing else. It knows how to fit a photo into
whatever space it has been given, how to turn a point on screen into a point on
the photo, and how to hand back a box the user drew; it does not know what a
class is, where labels live, or when anything should be saved. That belongs to
the page, which is also the only thing that can say what a change means.

The boxes it is given and the boxes it emits are in the photo's own fractional
coordinates rather than in pixels, because the photo on screen is a scaled copy
-- often decoded at a quarter size deliberately -- and a box measured in screen
pixels would be wrong the moment the window is resized.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ...core.i18n import tr
from ...core.review import Box

#: Wheel notches are reported in eighths of a degree; one notch is 120.
WHEEL_NOTCH = 120

BACKGROUND = QColor("#1f2124")
BOX_COLOUR = QColor("#27ae60")
SELECTED_COLOUR = QColor("#e67e22")
DRAFT_COLOUR = QColor("#f1c40f")
TEXT_COLOUR = QColor("#ecf0f1")

#: A drag shorter than this in both directions is a click that wobbled.
CLICK_SLOP = 4


class ImageCanvas(QWidget):
    """Draws one photo and its boxes, and lets the user correct them."""

    #: The user asked for the next (+1) or previous (-1) photo.
    stepped = Signal(int)
    #: The boxes changed. Carries the full new list, never a delta: the page
    #: writes the whole label file anyway, and a delta is a second description
    #: of the same thing that can disagree with the first.
    boxes_changed = Signal(list)
    #: Which box is selected, or -1. The page turns this into a Delete button.
    selection_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._boxes: list[Box] = []
        self._label = ""
        self._note = ""
        self._selected = -1
        self._draft: tuple[QPointF, QPointF] | None = None
        self._wheel = 0
        self._class_index = 0

        self.setMinimumSize(420, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(False)
        self.setAutoFillBackground(False)

    # -- contents ---------------------------------------------------------- #

    def set_photo(
        self,
        image: QImage | None,
        boxes=(),
        *,
        label: str = "",
        note: str = "",
        class_index: int = 0,
    ) -> None:
        self._image = image
        self._boxes = list(boxes)
        self._label = label
        self._note = note
        self._class_index = class_index
        self._draft = None
        self._select(-1)
        self.update()

    def clear(self, note: str = "") -> None:
        self.set_photo(None, (), note=note)

    @property
    def boxes(self) -> list[Box]:
        return list(self._boxes)

    @property
    def selected(self) -> int:
        return self._selected

    @property
    def has_photo(self) -> bool:
        return self._image is not None and not self._image.isNull()

    def delete_selected(self) -> bool:
        """Remove the selected box. Returns whether anything was removed."""
        if not 0 <= self._selected < len(self._boxes):
            return False
        del self._boxes[self._selected]
        self._select(min(self._selected, len(self._boxes) - 1))
        self.update()
        self.boxes_changed.emit(self.boxes)
        return True

    def _select(self, index: int) -> None:
        index = index if 0 <= index < len(self._boxes) else -1
        if index != self._selected:
            self._selected = index
            self.selection_changed.emit(index)

    # -- geometry ---------------------------------------------------------- #

    def photo_rect(self) -> QRectF:
        """Where the photo is drawn, letterboxed inside the widget."""
        area = QRectF(self.rect())
        if not self.has_photo:
            return area
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

    def _to_photo(self, point: QPointF) -> QPointF:
        """A widget point as a fraction of the photo, clamped to its edges."""
        rect = self.photo_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return QPointF(0.0, 0.0)
        x = (point.x() - rect.x()) / rect.width()
        y = (point.y() - rect.y()) / rect.height()
        return QPointF(min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0))

    def _box_at(self, point: QPointF) -> int:
        """The box under a point: the smallest one, so nesting stays reachable."""
        candidates = [
            (box.area, index)
            for index, box in enumerate(self._boxes)
            if box.contains(point.x(), point.y())
        ]
        return min(candidates)[1] if candidates else -1

    # -- painting ---------------------------------------------------------- #

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        if not self.has_photo:
            self._paint_message(painter, self._note or tr("No photo to show."))
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self.photo_rect()
        painter.drawImage(rect, self._image)

        for index, box in enumerate(self._boxes):
            self._paint_box(painter, rect, box, selected=index == self._selected)
        if self._draft is not None:
            self._paint_draft(painter)
        if self._note:
            self._paint_note(painter, self._note)

    def _paint_box(self, painter: QPainter, rect: QRectF, box: Box, *, selected: bool) -> None:
        x0, y0, x1, y1 = box.to_pixels(rect.width(), rect.height())
        drawn = QRectF(rect.x() + x0, rect.y() + y0, x1 - x0, y1 - y0)
        colour = SELECTED_COLOUR if selected else BOX_COLOUR

        painter.setPen(QPen(colour, 3 if selected else 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(drawn)

        if selected:
            painter.setBrush(QBrush(colour))
            for corner in (drawn.topLeft(), drawn.topRight(), drawn.bottomLeft(), drawn.bottomRight()):
                painter.drawRect(QRectF(corner.x() - 3, corner.y() - 3, 6, 6))
            painter.setBrush(Qt.BrushStyle.NoBrush)

        if not self._label:
            return
        font = QFont(self.font())
        font.setPointSizeF(max(8.0, font.pointSizeF()))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text = self._label
        width = metrics.horizontalAdvance(text) + 10
        height = metrics.height() + 2
        # Above the box, unless that would fall off the top of the photo.
        top = drawn.top() - height
        if top < rect.top():
            top = drawn.top()
        chip = QRectF(drawn.left(), top, width, height)
        painter.fillRect(chip, colour)
        painter.setPen(QPen(TEXT_COLOUR))
        painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_draft(self, painter: QPainter) -> None:
        start, end = self._draft
        painter.setPen(QPen(DRAFT_COLOUR, 2, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(start, end).normalized())

    def _paint_message(self, painter: QPainter, text: str) -> None:
        painter.setPen(QPen(TEXT_COLOUR))
        painter.drawText(
            QRectF(self.rect()).adjusted(24, 24, -24, -24),
            int(Qt.AlignmentFlag.AlignCenter) | int(Qt.TextFlag.TextWordWrap),
            text,
        )

    def _paint_note(self, painter: QPainter, text: str) -> None:
        """A line along the bottom, for what the photo itself cannot show."""
        metrics = painter.fontMetrics()
        height = metrics.height() + 10
        strip = QRectF(0, self.height() - height, self.width(), height)
        painter.fillRect(strip, QColor(0, 0, 0, 160))
        painter.setPen(QPen(TEXT_COLOUR))
        painter.drawText(strip, Qt.AlignmentFlag.AlignCenter, text)

    # -- input ------------------------------------------------------------- #

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() != Qt.MouseButton.LeftButton or not self.has_photo:
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        point = event.position()
        hit = self._box_at(self._to_photo(point))
        if hit >= 0:
            self._select(hit)
            self._draft = None
        else:
            self._select(-1)
            self._draft = (point, point)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._draft is None:
            return
        self._draft = (self._draft[0], event.position())
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._draft is None or event.button() != Qt.MouseButton.LeftButton:
            return
        start, _ = self._draft
        end = event.position()
        self._draft = None
        self.update()
        if abs(end.x() - start.x()) < CLICK_SLOP and abs(end.y() - start.y()) < CLICK_SLOP:
            return  # a click on the background, not a box
        self.add_box(self._to_photo(start), self._to_photo(end))

    def add_box(self, start: QPointF, end: QPointF) -> bool:
        """Add a box from two points already in the photo's coordinates."""
        box = Box.from_corners(
            start.x(), start.y(), end.x(), end.y(), index=self._class_index
        )
        if box is None:
            return False
        self._boxes.append(box)
        self._select(len(self._boxes) - 1)
        self.update()
        self.boxes_changed.emit(self.boxes)
        return True

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.scroll_by(event.angleDelta().y())
        event.accept()

    def scroll_by(self, delta: int) -> None:
        """Turn wheel movement into photo steps, keeping the remainder.

        Trackpads and fine-resolution wheels send a stream of small deltas; one
        photo per event would make a single flick jump through twenty of them.
        """
        self._wheel += int(delta)
        while self._wheel <= -WHEEL_NOTCH:
            self._wheel += WHEEL_NOTCH
            self.stepped.emit(1)  # scrolling down moves forward
        while self._wheel >= WHEEL_NOTCH:
            self._wheel -= WHEEL_NOTCH
            self.stepped.emit(-1)
