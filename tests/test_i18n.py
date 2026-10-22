"""Translations: the mechanism, the catalogues, and the rules that keep them whole.

The catalogue checks are the ones that matter day to day. A string added to the
code without a wording in every language, or a translation that lost a
placeholder, fails here -- the same way a parameter without a hover explanation fails in
``test_tooltips.py``.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from yolo_assistant.app.core import i18n, language
from yolo_assistant.app.core.i18n import N_, tr, trn
from yolo_assistant.app.core.worker_env import build_worker_env
from yolo_assistant.app.ui.common.param_form import (
    HELP_FILE,
    HelpTexts,
    translated_help_file,
)

ROOT = Path(__file__).resolve().parents[1]

#: Every language that needs a catalogue and a tooltip file.
TRANSLATED = sorted(code for code in i18n.LANGUAGES if code != i18n.SOURCE_LANGUAGE)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "_i18n_messages", ROOT / "packaging" / "i18n_messages.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Registered first, as an import would: its dataclasses look themselves up.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


@pytest.fixture
def polish():
    """The app in Polish for one test, and back in English afterwards."""
    language.apply("pl")
    yield
    language.apply("en")


@pytest.fixture
def german():
    """The app in German for one test, and back in English afterwards."""
    language.apply("de")
    yield
    language.apply("en")


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("UMLA_DATA_DIR", str(tmp_path))
    return tmp_path


# -- the mechanism ------------------------------------------------------------- #


def test_english_is_the_source_and_needs_no_catalogue():
    assert i18n.language() == "en"
    assert tr("Back") == "Back"
    assert tr("Copied {n} photos", n=3) == "Copied 3 photos"


def test_a_translation_is_used_when_the_language_has_one(polish):
    assert tr("Back") == "Wstecz"


def test_german_is_offered_too(german):
    assert i18n.language() == "de"
    assert tr("Back") == "Zurück"


def test_an_unknown_string_falls_back_to_english(polish):
    assert tr("No catalogue has this sentence in it.") == "No catalogue has this sentence in it."


def test_polish_plurals_have_three_forms():
    assert [i18n.plural_index("pl", n) for n in (1, 2, 4, 5, 11, 12, 14, 21, 22, 25, 102, 112)] == [
        0, 1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2,
    ]
    assert [i18n.plural_index("en", n) for n in (0, 1, 2)] == [1, 0, 1]
    assert [i18n.plural_index("de", n) for n in (0, 1, 2, 5, 21)] == [1, 0, 1, 1, 1]
    assert i18n.plural_count("de") == 2


@pytest.mark.parametrize("code", ["ru", "be"])
def test_east_slavic_plurals_follow_the_last_digit(code):
    # 1 файл, 2 файла, 5 файлов -- and, unlike Polish, 21 файл and 22 файла.
    numbers = (0, 1, 2, 4, 5, 11, 12, 14, 21, 22, 25, 101, 102, 111, 112)
    assert [i18n.plural_index(code, n) for n in numbers] == [
        2, 0, 1, 1, 2, 2, 2, 2, 0, 1, 2, 0, 1, 2, 2,
    ]
    assert i18n.plural_count(code) == 3


def test_trn_picks_the_english_form_and_supplies_n():
    assert trn("{n} photo", "{n} photos", 1) == "1 photo"
    assert trn("{n} photo", "{n} photos", 5) == "5 photos"


def test_trn_picks_the_polish_form(tmp_path):
    (tmp_path / "pl.po").write_text(
        'msgid "{n} photo"\nmsgid_plural "{n} photos"\n'
        'msgstr[0] "{n} zdjęcie"\nmsgstr[1] "{n} zdjęcia"\nmsgstr[2] "{n} zdjęć"\n',
        encoding="utf-8",
    )
    i18n.set_language("pl", tmp_path)
    try:
        assert [trn("{n} photo", "{n} photos", n) for n in (1, 3, 5, 22)] == [
            "1 zdjęcie", "3 zdjęcia", "5 zdjęć", "22 zdjęcia",
        ]
    finally:
        language.apply("en")


def test_numbers_formatted_on_purpose_follow_the_language(polish):
    assert tr("No catalogue has {n:,} of these", n=12345) == "No catalogue has 12 345 of these"
    assert tr("No catalogue has {x:.2f} of these", x=1.5) == "No catalogue has 1,50 of these"
    # A bare field may be a version number; it is left alone.
    assert tr("No catalogue has CUDA {v}", v=12.4) == "No catalogue has CUDA 12.4"


@pytest.mark.parametrize("code", ["ru", "be"])
def test_russian_and_belarusian_group_with_a_space_and_use_a_decimal_comma(code):
    language.apply(code)
    try:
        assert tr("No catalogue has {n:,} of these", n=12345) == "No catalogue has 12\u00a0345 of these"
        assert tr("No catalogue has {x:.2f} of these", x=1.5) == "No catalogue has 1,50 of these"
    finally:
        language.apply("en")


def test_german_groups_thousands_with_a_point(german):
    assert tr("No catalogue has {n:,} of these", n=12345) == "No catalogue has 12.345 of these"
    assert tr("No catalogue has {x:,.2f} of these", x=1234.5) == "No catalogue has 1.234,50 of these"


def test_a_broken_placeholder_in_a_translation_falls_back_to_english(tmp_path):
    (tmp_path / "pl.po").write_text('msgid "Hello {name}"\nmsgstr "Cześć {imie}"\n', encoding="utf-8")
    i18n.set_language("pl", tmp_path)
    try:
        assert tr("Hello {name}", name="Ala") == "Hello Ala"
    finally:
        language.apply("en")


def test_a_missing_catalogue_leaves_the_app_in_english(tmp_path):
    i18n.set_language("pl", tmp_path)
    try:
        assert i18n.language() == "en"
    finally:
        language.apply("en")


def test_po_files_survive_a_round_trip():
    tricky = 'Line one\nLine "two"\twith a tab and a back\\slash\n'
    text = "msgid " + i18n.quote_po(tricky) + "\nmsgstr " + i18n.quote_po("x") + "\n"
    (entry,) = i18n.parse_po(text)
    assert entry["msgid"] == tricky and entry["msgstr"] == "x"


def test_fuzzy_and_empty_entries_are_not_used():
    text = (
        '#, fuzzy\nmsgid "A"\nmsgstr "Guessed"\n\n'
        'msgid "B"\nmsgstr ""\n\n'
        'msgid "C"\nmsgstr "Ce"\n'
    )
    entries = i18n.parse_po(text)
    assert [e["fuzzy"] for e in entries] == [True, False, False]


def test_the_worker_copy_is_byte_identical():
    app_copy = (ROOT / "yolo_assistant/app/core/i18n.py").read_bytes()
    worker_copy = (ROOT / "worker/i18n.py").read_bytes()
    assert hashlib.sha256(app_copy).digest() == hashlib.sha256(worker_copy).digest(), (
        "worker/i18n.py has drifted; run python packaging/sync_protocol.py"
    )


def test_a_worker_is_started_in_the_gui_language(polish, tmp_path):
    env = build_worker_env(tmp_path, base={})
    assert env[i18n.LANGUAGE_ENV] == "pl"
    assert env[i18n.CATALOG_DIR_ENV] == str(language.CATALOG_DIR)


def test_a_worker_adopts_the_language_it_was_given():
    i18n.configure_from_environment(
        {i18n.LANGUAGE_ENV: "pl", i18n.CATALOG_DIR_ENV: str(language.CATALOG_DIR)}
    )
    try:
        assert i18n.language() == "pl" and tr("Back") == "Wstecz"
    finally:
        language.apply("en")


# -- choosing and remembering ---------------------------------------------------- #


def test_the_first_launch_follows_the_system(data_dir):
    assert language.choose("pl_PL") == "pl"
    assert language.choose("de_AT") == "de"
    assert language.choose("ru_RU") == "ru"
    assert language.choose("be_BY") == "be"
    assert language.choose("fr_FR") == "en"
    assert language.choose(None) == "en"


def test_a_saved_choice_beats_the_system_and_an_override_beats_both(data_dir):
    language.save("en")
    assert language.choose("pl_PL") == "en"
    assert language.choose("pl_PL", override="pl") == "pl"


def test_saving_keeps_other_settings(data_dir):
    (data_dir / "settings.json").write_text(json.dumps({"other": 1}), encoding="utf-8")
    language.save("pl")
    assert json.loads((data_dir / "settings.json").read_text()) == {"other": 1, "language": "pl"}
    assert language.saved() == "pl"


def test_a_damaged_settings_file_is_ignored(data_dir):
    (data_dir / "settings.json").write_text("{not json", encoding="utf-8")
    assert language.saved() is None
    language.save("pl")
    assert language.saved() == "pl"


def test_restarting_drops_a_one_off_language():
    from yolo_assistant.app.ui.wizard import restart_command

    _, args = restart_command(["app", "--fake-worker", "--language", "pl", "--language=en"])
    assert args[-1] == "--fake-worker"
    assert not any("language" in arg for arg in args)


# -- the catalogues ------------------------------------------------------------ #


def test_every_string_can_be_extracted(tool):
    _, problems = tool.extract()
    assert not problems, "\n".join(map(str, problems))


@pytest.mark.parametrize("code", TRANSLATED)
def test_every_string_has_a_translation(tool, code):
    messages, _ = tool.extract()
    problems = tool.check(code, messages)
    assert not problems, (
        f"{len(problems)} catalogue problem(s); run python packaging/i18n_messages.py\n"
        + "\n".join(map(str, problems[:40]))
    )


@pytest.mark.parametrize("code", TRANSLATED)
def test_every_tooltip_has_a_translation(code):
    english = HelpTexts(language="en")
    assert translated_help_file(code).exists()
    translated = HelpTexts(HELP_FILE.with_name(translated_help_file(code).name))
    assert set(translated.keys()) == set(english.keys())
    untranslated = [key for key in english.keys() if translated[key] == english[key]]
    assert not untranslated, f"these tooltips are still in English: {untranslated}"


def test_tooltips_follow_the_language(polish):
    assert HelpTexts()["system.setup"] == HelpTexts(
        HELP_FILE.with_name(translated_help_file("pl").name)
    )["system.setup"]


def _translated_at_import(tree: ast.AST) -> list[int]:
    """Lines where tr() or trn() runs before main() has chosen a language."""
    found: list[int] = []

    def visit(node: ast.AST, deferred: bool) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # Decorators and defaults run when the function is defined.
            for child in [*getattr(node, "decorator_list", []), *node.args.defaults,
                          *[d for d in node.args.kw_defaults if d is not None]]:
                visit(child, deferred)
            body = node.body if isinstance(node.body, list) else [node.body]
            for child in body:
                visit(child, True)
            return
        if (
            not deferred
            and isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", None)) in ("tr", "trn")
        ):
            found.append(node.lineno)
        for child in ast.iter_child_nodes(node):
            visit(child, deferred)

    visit(tree, False)
    return found


def test_nothing_is_translated_at_import_time(tool):
    offenders = []
    for path in tool._python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders += [f"{path.relative_to(ROOT)}:{line}" for line in _translated_at_import(tree)]
    assert not offenders, (
        "tr() here runs before the language is chosen; mark the text with N_() and "
        "translate it where it is shown:\n" + "\n".join(offenders)
    )


def test_n_marks_without_translating():
    assert N_("Back") == "Back"


# -- the whole window ------------------------------------------------------------ #


@pytest.fixture
def window(qapp, data_dir, monkeypatch):
    import sys

    from yolo_assistant.app.core import runtime
    from yolo_assistant.app.main import build_pages
    from yolo_assistant.app.ui.wizard import WizardWindow

    monkeypatch.setenv(runtime.PYTHON_ENV, sys.executable)

    def build():
        pages = build_pages(fake_worker=True)
        built = WizardWindow(pages)
        windows.append(built)
        return built

    windows = []
    yield build
    for built in windows:
        built.close()


def test_every_page_builds_in_polish(polish, window):
    built = window()
    rail = [built._rail.item(row).text() for row in range(built._rail.count())]
    assert "Przygotowanie zbioru danych" in " ".join(rail)
    assert built._back.text() == "Wstecz"
    menus = [action.text() for action in built.menuBar().actions()]
    assert menus == ["&Język", "&Pomoc"]


def test_every_page_builds_in_german(german, window):
    built = window()
    assert built._back.text() == "Zurück"
    menus = [action.text() for action in built.menuBar().actions()]
    assert menus == ["&Sprache", "&Hilfe"]
    languages = [action.text() for action in built._language_group.actions()]
    assert languages == ["English", "Deutsch", "Polski", "Русский", "Беларуская"]


@pytest.mark.parametrize(
    ("code", "back", "menus", "step"),
    [
        ("ru", "Назад", ["&Язык", "&Справка"], "Подготовка набора данных"),
        ("be", "Назад", ["&Мова", "&Даведка"], "Падрыхтоўка набору даных"),
    ],
)
def test_every_page_builds_in_russian_and_belarusian(window, code, back, menus, step):
    language.apply(code)
    try:
        built = window()
        rail = [built._rail.item(row).text() for row in range(built._rail.count())]
        assert step in " ".join(rail)
        assert built._back.text() == back
        assert [action.text() for action in built.menuBar().actions()] == menus
    finally:
        language.apply("en")


def test_the_language_menu_saves_the_choice_and_offers_a_restart(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    asked = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: asked.append(box.text()) or 0)
    built = window()
    restarts = []
    monkeypatch.setattr(built, "restart", lambda: restarts.append(True))

    built.choose_language("pl")
    assert language.saved() == "pl"
    # Asked in the language just chosen, and the app itself is still in English.
    assert asked and "uruchomieniu" in asked[0]
    assert i18n.language() == "en"
    assert restarts == [], "Later was the default, so nothing restarted"

    asked.clear()
    built.choose_language("en")
    assert language.saved() == "en" and not asked, "back to the language on screen"
