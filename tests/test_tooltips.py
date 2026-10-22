"""The spec requires every parameter to explain itself on hover.

``ParamForm.add_row`` enforces that at construction. These tests enforce the
other half: that the catalogue itself is complete and readable.
"""

import pytest

from yolo_assistant.app.ui.common.param_form import (
    HelpTexts,
    MissingHelpText,
)


@pytest.fixture(scope="module")
def texts():
    return HelpTexts()


def test_every_entry_has_text(texts):
    empty = [key for key, value in texts.items() if not value.strip()]
    assert not empty, f"these parameters have no explanation: {empty}"


def test_explanations_are_written_for_a_beginner(texts):
    """Short enough to read on hover, long enough to actually say something."""
    too_short = [k for k, v in texts.items() if len(v) < 40]
    assert not too_short, f"these explanations say too little: {too_short}"
    too_long = [k for k, v in texts.items() if len(v) > 600]
    assert not too_long, f"these explanations are too long for a tooltip: {too_long}"


def test_missing_key_raises_a_helpful_error(texts):
    with pytest.raises(MissingHelpText, match="must explain itself"):
        texts["train.no_such_parameter"]


def test_keys_are_namespaced_by_step(texts):
    assert all("." in key for key in texts.keys())
    sections = {key.split(".", 1)[0] for key in texts.keys()}
    # Every workflow step that has parameters must be represented.
    assert {"system", "dataset", "autolabel", "train", "export", "test"} <= sections
