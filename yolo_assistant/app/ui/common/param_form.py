"""Parameter widgets that cannot exist without an explanation.

The spec's third requirement is that every parameter control explains itself on
hover. Enforcing that by code review would mean catching it every time someone
adds a field; enforcing it structurally means it cannot be got wrong.

:meth:`ParamForm.add_row` looks the explanation up by key and raises
:class:`MissingHelpText` if there isn't one. A control with no tooltip therefore
fails at construction, in the first test that builds the page -- and
``tests/test_tooltips.py`` walks every declared key independently.

Translations live beside the English as ``tooltips.<language>.toml``, with the
same keys, and are laid over it. The tests require every key in every language,
so the English showing through is only ever a fallback for a damaged install.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

from PySide6.QtWidgets import QFormLayout, QLabel, QWidget

from ...core import i18n

HELP_FILE = Path(__file__).resolve().parents[2] / "resources" / "tooltips.toml"


class MissingHelpText(KeyError):
    """Raised when a parameter has no hover explanation."""


def translated_help_file(code: str) -> Path:
    """Where the explanations in language ``code`` are kept."""
    return HELP_FILE.with_name(f"{HELP_FILE.stem}.{code}{HELP_FILE.suffix}")


class HelpTexts:
    """Flattened ``section.key`` -> explanation."""

    def __init__(self, path: Path | None = None, *, language: str | None = None) -> None:
        self.path = path or HELP_FILE
        self._texts = _read(self.path)
        code = i18n.language() if language is None else language
        if path is None and code != i18n.SOURCE_LANGUAGE:
            translated = translated_help_file(code)
            if translated.exists():
                self._texts.update(
                    {key: text for key, text in _read(translated).items() if text}
                )

    def __contains__(self, key: object) -> bool:
        return key in self._texts

    def __getitem__(self, key: str) -> str:
        try:
            text = self._texts[key]
        except KeyError:
            raise MissingHelpText(
                f"No hover explanation for {key!r}. Add it to {self.path.name} -- "
                "every parameter in this app must explain itself."
            ) from None
        if not text:
            raise MissingHelpText(f"The explanation for {key!r} is empty.")
        return text

    def keys(self):
        return self._texts.keys()

    def items(self):
        return self._texts.items()


def _read(path: Path) -> dict[str, str]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    texts: dict[str, str] = {}
    for section, entries in raw.items():
        if not isinstance(entries, dict):
            continue
        for key, value in entries.items():
            if isinstance(value, dict):
                texts[f"{section}.{key}"] = str(value.get("help", "")).strip()
            else:
                texts[f"{section}.{key}"] = str(value).strip()
    return texts


@lru_cache(maxsize=1)
def help_texts() -> HelpTexts:
    return HelpTexts()


def apply_help(widget: QWidget, key: str, *, texts: HelpTexts | None = None) -> str:
    """Attach the explanation for ``key`` to ``widget``. Returns the text."""
    text = (texts or help_texts())[key]
    widget.setToolTip(text)
    # Qt shows the tooltip on the label too, which is the larger hover target
    # and the thing a beginner actually points at.
    widget.setWhatsThis(text)
    return text


class ParamForm(QFormLayout):
    """A form whose every row carries a hover explanation."""

    def __init__(self, parent: QWidget | None = None, *, texts: HelpTexts | None = None) -> None:
        super().__init__(parent)
        self._texts = texts or help_texts()
        self.keys: list[str] = []
        self.setLabelAlignment(self.labelAlignment())
        self.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

    def add_row(self, key: str, label: str, widget: QWidget) -> QWidget:
        text = apply_help(widget, key, texts=self._texts)
        label_widget = QLabel(label)
        label_widget.setToolTip(text)
        label_widget.setBuddy(widget)
        super().addRow(label_widget, widget)
        self.keys.append(key)
        return widget
