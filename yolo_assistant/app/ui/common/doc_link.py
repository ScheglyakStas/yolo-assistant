"""A link to the Ultralytics documentation page behind a step.

The app explains every setting it shows, but it deliberately shows only a few
of them; the additional-settings box accepts dozens more. Their reference is
Ultralytics' own documentation, so the steps that hand settings to Ultralytics
point there rather than paraphrasing it.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget

from ...core.i18n import tr


def doc_link(url: str, caption: str, parent: QWidget | None = None) -> QLabel:
    """A label holding one link that opens ``url`` in the system browser."""
    label = QLabel(f'<a href="{url}">{caption}</a>', parent)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    label.setOpenExternalLinks(True)
    label.setToolTip(
        tr(
            "Opens {url} in your web browser: every setting Ultralytics accepts on this "
            "step, including the ones you can type into Additional settings.",
            url=url,
        )
    )
    return label
