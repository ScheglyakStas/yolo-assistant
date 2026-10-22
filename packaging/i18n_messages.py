#!/usr/bin/env python3
"""Find every translatable string and keep the catalogues in step with them.

A string is translatable when it is the literal first argument of ``tr()``,
``N_()`` or -- with its plural -- ``trn()``, anywhere in the app or the worker.

    python packaging/i18n_messages.py            # report what needs doing
    python packaging/i18n_messages.py --update   # add new strings, drop stale ones

``--update`` rewrites each catalogue in ``resources/i18n/``: translations that
still match a string are kept, strings new to the code are added untranslated
(and so fall back to English), and strings the code no longer has are removed.
It never invents a translation. ``tests/test_i18n.py`` runs the same report and
fails while anything is left.

Run with the plain system Python; it imports nothing but the standard library
and the app's ``i18n`` module.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (ROOT / "yolo_assistant", ROOT / "worker")
CATALOG_DIR = ROOT / "yolo_assistant" / "app" / "resources" / "i18n"

_spec = importlib.util.spec_from_file_location(
    "_umla_i18n", ROOT / "yolo_assistant" / "app" / "core" / "i18n.py"
)
i18n = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i18n)

#: Call names that mark their first argument as a message.
SINGULAR_CALLS = {"tr", "N_"}
PLURAL_CALLS = {"trn"}


@dataclass
class Message:
    msgid: str
    plural: str | None = None
    places: list[str] = field(default_factory=list)


@dataclass
class Problem:
    where: str
    text: str

    def __str__(self) -> str:
        return f"{self.where}: {self.text}"


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in SOURCES:
        files.extend(
            path for path in sorted(root.rglob("*.py")) if "__pycache__" not in path.parts
        )
    # The worker's copy of i18n.py is the app's, and neither translates anything.
    return [path for path in files if path.name != "i18n.py"]


def extract(files: list[Path] | None = None) -> tuple[dict[str, Message], list[Problem]]:
    """Every message in the source, and every call that cannot be extracted."""
    messages: dict[str, Message] = {}
    problems: list[Problem] = []
    for path in files if files is not None else _python_files():
        path = path.resolve()
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name not in SINGULAR_CALLS and name not in PLURAL_CALLS:
                continue
            where = f"{relative}:{node.lineno}"
            literal_count = 2 if name in PLURAL_CALLS else 1
            args = node.args[:literal_count]
            if any(isinstance(arg, ast.JoinedStr) for arg in args):
                problems.append(Problem(where, f"{name}() of an f-string; use {{placeholders}}"))
                continue
            literals = [
                arg.value
                for arg in args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
            if len(literals) != literal_count:
                if name == "N_" or name in PLURAL_CALLS:
                    problems.append(Problem(where, f"{name}() needs string literals"))
                # tr(variable) translates something marked with N_() elsewhere.
                continue
            msgid = literals[0]
            plural = literals[1] if name in PLURAL_CALLS else None
            message = messages.setdefault(msgid, Message(msgid, plural))
            if plural is not None and message.plural not in (None, plural):
                problems.append(Problem(where, f"{msgid!r} has two different plurals"))
            message.plural = message.plural or plural
            message.places.append(where)
    return messages, problems


def placeholders(text: str) -> set[str] | None:
    """The named fields of a format string, or ``None`` if it is not one."""
    try:
        return {
            name.split(".")[0].split("[")[0]
            for _, name, _, _ in string.Formatter().parse(text)
            if name
        }
    except ValueError:
        return None


def catalogs() -> dict[str, Path]:
    return {
        code: CATALOG_DIR / f"{code}.po"
        for code in i18n.LANGUAGES
        if code != i18n.SOURCE_LANGUAGE
    }


def check(
    code: str, messages: dict[str, Message], path: Path | None = None, *, stale: bool = True
) -> list[Problem]:
    """What is wrong with one language's catalogue."""
    path = path or catalogs()[code]
    if not path.exists():
        return [Problem(path.name, "the catalogue does not exist")]
    problems: list[Problem] = []
    entries = {e["msgid"]: e for e in i18n.parse_po(path.read_text(encoding="utf-8")) if e["msgid"]}
    forms_needed = i18n.plural_count(code)

    for msgid, message in messages.items():
        where = message.places[0]
        entry = entries.get(msgid)
        if entry is None:
            problems.append(Problem(where, f"not in {path.name}: {msgid!r}"))
            continue
        if entry["fuzzy"]:
            problems.append(Problem(where, f"marked fuzzy in {path.name}: {msgid!r}"))
        source_fields = placeholders(msgid) or set()
        if message.plural is None:
            if entry["plural"] is not None:
                problems.append(Problem(where, f"{path.name} gives plurals for {msgid!r}"))
                continue
            if not entry["msgstr"]:
                problems.append(Problem(where, f"untranslated: {msgid!r}"))
                continue
            translations = [entry["msgstr"]]
            required = source_fields
        else:
            source_fields |= placeholders(message.plural) or set()
            if len(entry["forms"]) != forms_needed or not all(entry["forms"]):
                problems.append(
                    Problem(where, f"{msgid!r} needs {forms_needed} plural forms in {path.name}")
                )
                continue
            translations = list(entry["forms"])
            # A form may spell the number out ("jedno zdjęcie") rather than use {n}.
            required = source_fields - {"n"}
            source_fields |= {"n"}
        for translation in translations:
            fields = placeholders(translation)
            if fields is None:
                problems.append(Problem(where, f"unbalanced braces in the translation of {msgid!r}"))
            elif not fields <= source_fields:
                problems.append(
                    Problem(where, f"translation of {msgid!r} uses {sorted(fields - source_fields)}, "
                                   "which the English does not supply")
                )
            elif not required <= fields:
                problems.append(
                    Problem(where, f"translation of {msgid!r} drops {sorted(required - fields)}")
                )

    for msgid in entries.keys() - messages.keys() if stale else ():
        problems.append(Problem(path.name, f"no longer in the code: {msgid!r}"))
    return problems


def update(code: str, messages: dict[str, Message]) -> tuple[int, int]:
    """Rewrite one catalogue to match the code. Returns (added, removed)."""
    path = catalogs()[code]
    existing = (
        {e["msgid"]: e for e in i18n.parse_po(path.read_text(encoding="utf-8")) if e["msgid"]}
        if path.exists()
        else {}
    )
    forms_needed = i18n.plural_count(code)
    lines = [
        f"# {i18n.LANGUAGES[code]} translation of YOLO Assistant.",
        "# Maintained with packaging/i18n_messages.py.",
        'msgid ""',
        'msgstr ""',
        '"Content-Type: text/plain; charset=UTF-8\\n"',
        f'"Language: {code}\\n"',
        "",
    ]
    ordered = sorted(messages.values(), key=lambda m: _sort_key(m.places[0]))
    for message in ordered:
        old = existing.get(message.msgid, {})
        lines.append("#: " + " ".join(message.places[:4]))
        if old.get("fuzzy"):
            lines.append("#, fuzzy")
        lines.append("msgid " + i18n.quote_po(message.msgid))
        if message.plural is None:
            text = old.get("msgstr", "") if old.get("plural") is None else ""
            lines.append("msgstr " + i18n.quote_po(text))
        else:
            lines.append("msgid_plural " + i18n.quote_po(message.plural))
            forms = list(old.get("forms", ())) if old.get("plural") is not None else []
            forms = (forms + [""] * forms_needed)[:forms_needed]
            for index, form in enumerate(forms):
                lines.append(f"msgstr[{index}] " + i18n.quote_po(form))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    added = len(messages.keys() - existing.keys())
    removed = len(existing.keys() - messages.keys())
    return added, removed


def _sort_key(place: str) -> tuple[str, int]:
    file, _, line = place.rpartition(":")
    return file, int(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true", help="rewrite the catalogues to match the code")
    parser.add_argument(
        "--part", nargs="+", metavar="FILE", type=Path,
        help="check only these source files against --catalogue (for translating piecemeal)",
    )
    parser.add_argument("--catalogue", type=Path, help="the .po file --part is checked against")
    parser.add_argument(
        "--language", default="pl", choices=sorted(catalogs()),
        help="the language --catalogue is in (default: pl)",
    )
    args = parser.parse_args()

    if args.part:
        messages, problems = extract(args.part)
        problems += check(args.language, messages, args.catalogue, stale=False) if args.catalogue else []
        for problem in problems:
            print(problem)
        print(f"{len(messages)} messages, {len(problems)} problems")
        return 1 if problems else 0

    messages, problems = extract()
    for problem in problems:
        print(problem)
    status = 1 if problems else 0
    for code in catalogs():
        if args.update:
            added, removed = update(code, messages)
            print(f"{code}.po: {added} added, {removed} removed")
        found = check(code, messages)
        for problem in found:
            print(f"[{code}] {problem}")
        status = status or (1 if found else 0)
        if not found:
            print(f"{code}.po: all {len(messages)} messages translated")
    return status


if __name__ == "__main__":
    sys.exit(main())
