"""Translations: :func:`tr`, :func:`trn`, :func:`N_` and the catalogue behind them.

The English text in the source *is* the message id. A string is made
translatable by wrapping it where it is shown::

    tr("Looking through {folder}...", folder=source)
    trn("{n} photo", "{n} photos", count)

and its Polish wording goes in ``resources/i18n/pl.po`` -- an ordinary gettext
catalogue, so any translation tool can edit it. There is no compile step: the
``.po`` file is read as it is, because a build step that someone forgets is how
a catalogue silently stops matching the code. ``python
packaging/i18n_messages.py`` lists what is missing, and the test suite fails if
any string is.

**Nothing may be translated at import time.** The language is chosen in
``main()``, after most modules have been imported, so a ``tr()`` in a class body
or a module constant would freeze the English. Text defined up front is marked
with :func:`N_` instead -- which does nothing but let the extractor find it --
and passed through :func:`tr` where it is displayed.
``tests/test_i18n.py`` enforces this.

The worker runs under another interpreter and cannot import the app, so
``worker/i18n.py`` is a byte-identical copy of this file, kept in step by
``packaging/sync_protocol.py``. The GUI hands its choice to the worker through
the environment; see :func:`environment`.
"""

from __future__ import annotations

import os
import string
from pathlib import Path
from typing import Any

#: The languages on offer, each named in itself, in menu order.
LANGUAGES: dict[str, str] = {
    "en": "English",
    "de": "Deutsch",
    "pl": "Polski",
    "ru": "Русский",
    "be": "Беларуская",
}

#: The language the source is written in. It needs no catalogue.
SOURCE_LANGUAGE = "en"

#: How a worker learns which language the GUI is in, and where the catalogues are.
LANGUAGE_ENV = "UMLA_LANGUAGE"
CATALOG_DIR_ENV = "UMLA_CATALOG_DIR"

#: Thousands and decimal separators, for languages that do not use ``,`` and ``.``.
#: German groups with a point, the others with a (non-breaking) space; all use a
#: decimal comma.
_NUMBER_SEPARATORS: dict[str, tuple[str, str]] = {
    "de": (".", ","),
    "pl": ("\u00a0", ","),
    "ru": ("\u00a0", ","),
    "be": ("\u00a0", ","),
}

#: Languages with three plural forms where the last digit decides: "1 файл",
#: "21 файл", "2 файла", "5 файлов" -- unlike Polish, where 21 takes the third.
_EAST_SLAVIC = ("ru", "be")

_language = SOURCE_LANGUAGE
_catalog_dir: Path | None = None
_catalog: dict[str, str | tuple[str, ...]] = {}


# -- choosing a language ----------------------------------------------------- #


def set_language(code: str, catalog_dir: str | os.PathLike[str] | None = None) -> None:
    """Switch to ``code``, reading its catalogue from ``catalog_dir``.

    An unknown code, or a catalogue that cannot be read, leaves the app in
    English rather than failing: a missing translation must never be the reason
    the program does not start.
    """
    global _language, _catalog_dir, _catalog
    if catalog_dir is not None:
        _catalog_dir = Path(catalog_dir)
    if code not in LANGUAGES:
        code = SOURCE_LANGUAGE
    catalog: dict[str, str | tuple[str, ...]] = {}
    if code != SOURCE_LANGUAGE and _catalog_dir is not None:
        try:
            catalog = load_po(_catalog_dir / f"{code}.po")
        except (OSError, ValueError):
            code, catalog = SOURCE_LANGUAGE, {}
    _language, _catalog = code, catalog


def language() -> str:
    """The code of the language currently in use."""
    return _language


def environment() -> dict[str, str]:
    """The variables that put a worker subprocess in the same language."""
    env = {LANGUAGE_ENV: _language}
    if _catalog_dir is not None:
        env[CATALOG_DIR_ENV] = str(_catalog_dir)
    return env


def configure_from_environment(env: dict[str, str] | None = None) -> None:
    """Adopt the language the GUI passed down. Called once by the worker."""
    env = os.environ if env is None else env
    code = env.get(LANGUAGE_ENV)
    if code:
        set_language(code, env.get(CATALOG_DIR_ENV) or None)


# -- translating --------------------------------------------------------------- #


def N_(text: str) -> str:  # noqa: N802 - the gettext convention
    """Mark ``text`` for translation without translating it yet."""
    return text


def tr(text: str, /, **values: Any) -> str:
    """``text`` in the current language, with ``{placeholders}`` filled in.

    Placeholders are filled only when values are given, so a string with no
    values may contain literal braces.
    """
    found = _catalog.get(text)
    translated = found if isinstance(found, str) and found else text
    return _fill(translated, text, values)


def trn(singular: str, plural: str, n: int, /, **values: Any) -> str:
    """The form of a message that agrees with the number ``n``.

    ``n`` is available to the message as ``{n}``. English has two forms; Polish
    has three ("1 zdjęcie", "2 zdjęcia", "5 zdjęć"), which is why this cannot be
    done by choosing between two English strings at the call site.
    """
    values.setdefault("n", n)
    english = singular if n == 1 else plural
    forms = _catalog.get(singular)
    translated = english
    if isinstance(forms, tuple):
        index = plural_index(_language, n)
        if index < len(forms) and forms[index]:
            translated = forms[index]
    return _fill(translated, english, values)


def plural_index(code: str, n: int) -> int:
    """Which of a language's plural forms agrees with ``n``."""
    n = abs(int(n))
    if code in _EAST_SLAVIC:
        if n % 10 == 1 and n % 100 != 11:
            return 0
        if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
            return 1
        return 2
    if code == "pl":
        if n == 1:
            return 0
        if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
            return 1
        return 2
    return 0 if n == 1 else 1


def plural_count(code: str) -> int:
    """How many plural forms a language's catalogue entries carry."""
    return 3 if code == "pl" or code in _EAST_SLAVIC else 2


class _Formatter(string.Formatter):
    """``str.format`` with the current language's number separators."""

    def format_field(self, value: Any, format_spec: str) -> str:
        text = super().format_field(value, format_spec)
        separators = _NUMBER_SEPARATORS.get(_language)
        # Only numbers formatted on purpose ("{n:,}", "{x:.1f}"). A bare "{x}"
        # may be a version number that happens to be a float.
        if (
            separators
            and format_spec
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            group, decimal = separators
            text = text.translate({ord(","): "\0", ord("."): decimal}).replace("\0", group)
        return text


_FORMATTER = _Formatter()


def _fill(template: str, fallback: str, values: dict[str, Any]) -> str:
    if not values:
        return template
    try:
        return _FORMATTER.vformat(template, (), values)
    except (KeyError, IndexError, ValueError):
        # A mistyped placeholder in a translation must not crash the app.
        return _FORMATTER.vformat(fallback, (), values)


# -- reading catalogues -------------------------------------------------------- #


def load_po(path: str | os.PathLike[str]) -> dict[str, str | tuple[str, ...]]:
    """Read a gettext ``.po`` file into ``msgid -> msgstr``.

    Plural entries map their singular ``msgid`` to a tuple of forms. Untranslated
    and fuzzy entries are left out, so they fall back to English.
    """
    return {
        entry["msgid"]: entry["forms"] if entry["plural"] is not None else entry["msgstr"]
        for entry in parse_po(Path(path).read_text(encoding="utf-8"))
        if entry["msgid"] and not entry["fuzzy"] and _is_translated(entry)
    }


def parse_po(text: str) -> list[dict[str, Any]]:
    """Every entry of a ``.po`` file, translated or not, in file order.

    Each is a dict with ``msgid``, ``plural`` (the ``msgid_plural`` or ``None``),
    ``msgstr``, ``forms`` (the ``msgstr[i]`` values) and ``fuzzy``.
    """
    entries: list[dict[str, Any]] = []
    entry: dict[str, Any] | None = None
    field = "msgid"
    fuzzy = False  # a "#, fuzzy" flag belongs to the entry that follows it

    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            fuzzy = fuzzy or (line.startswith("#,") and "fuzzy" in line)
            continue
        if line.startswith('"'):
            if entry is None:
                raise ValueError(f"line {number}: text outside an entry")
            _append(entry, field, _unquote(line, number))
            continue

        keyword, _, rest = line.partition(" ")
        if keyword == "msgid":
            entry = _new_entry()
            entry["fuzzy"], fuzzy = fuzzy, False
            entries.append(entry)
            field = "msgid"
        elif entry is None:
            raise ValueError(f"line {number}: {keyword} before msgid")
        elif keyword == "msgid_plural":
            entry["plural"] = ""
            field = "plural"
        elif keyword == "msgstr":
            field = "msgstr"
        elif keyword.startswith("msgstr[") and keyword.endswith("]"):
            index = int(keyword[7:-1])
            while len(entry["forms"]) <= index:
                entry["forms"].append("")
            field = f"form:{index}"
        else:
            raise ValueError(f"line {number}: cannot read {keyword!r}")
        _append(entry, field, _unquote(rest.strip(), number))

    for entry in entries:
        entry["forms"] = tuple(entry["forms"])
    return entries


def _new_entry() -> dict[str, Any]:
    return {"msgid": "", "plural": None, "msgstr": "", "forms": [], "fuzzy": False}


def _append(entry: dict[str, Any], field: str, text: str) -> None:
    if field.startswith("form:"):
        index = int(field[5:])
        entry["forms"][index] += text
    else:
        entry[field] += text


def _is_translated(entry: dict[str, Any]) -> bool:
    if entry["plural"] is not None:
        return bool(entry["forms"]) and all(entry["forms"])
    return bool(entry["msgstr"])


_ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}


def _unquote(token: str, number: int) -> str:
    if len(token) < 2 or token[0] != '"' or token[-1] != '"':
        raise ValueError(f"line {number}: expected a quoted string")
    body, out, i = token[1:-1], [], 0
    while i < len(body):
        char = body[i]
        if char == "\\" and i + 1 < len(body):
            out.append(_ESCAPES.get(body[i + 1], body[i + 1]))
            i += 2
            continue
        out.append(char)
        i += 1
    return "".join(out)


def quote_po(text: str) -> str:
    """``text`` as one or more ``.po`` string lines, split after newlines."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\t", "\\t")
    pieces = escaped.split("\n")
    lines = [piece + "\\n" for piece in pieces[:-1]] + [pieces[-1]]
    lines = [line for line in lines if line]
    if len(lines) <= 1:
        return f'"{lines[0] if lines else ""}"'
    return '""\n' + "\n".join(f'"{line}"' for line in lines)
