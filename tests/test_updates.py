"""The update button has to answer two questions, not one.

"Is there something newer" is easy. "Is it something this app has been tested
against" is the one that matters: the 8.4 line renamed export arguments
mid-series, so an unconditional upgrade can break Step 6 for someone who has no
idea the two are connected.
"""

import json

from yolo_assistant.app.core import updates
from yolo_assistant.app.core.updates import UpdateInfo


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def opener_for(version):
    return lambda *_a, **_k: FakeResponse({"info": {"version": version}})


def test_a_newer_release_is_offered():
    info = updates.check("8.4.100", opener=opener_for("8.4.154"))
    assert info.available
    assert "8.4.154 is available" in info.summary()


def test_the_current_release_is_not_offered():
    info = updates.check("8.4.154", opener=opener_for("8.4.154"))
    assert not info.available
    assert "newest release" in info.summary()


def test_an_older_pypi_answer_is_not_an_update():
    assert not UpdateInfo("8.4.200", "8.4.154").available


def test_versions_compare_numerically_not_alphabetically():
    """8.4.9 vs 8.4.100 is where string comparison quietly gets it wrong."""
    assert UpdateInfo("8.4.9", "8.4.100").available
    assert not UpdateInfo("8.4.100", "8.4.9").available


def test_a_release_outside_the_tested_range_is_flagged_but_still_offered():
    info = UpdateInfo("8.4.154", "8.5.1")
    assert info.available
    assert not info.latest_is_tested
    assert "outside the range" in info.summary()
    assert "may break" in info.summary()


def test_the_pinned_version_is_inside_its_own_tested_range():
    """Guards against the pin and the range drifting apart when either is bumped."""
    assert UpdateInfo(installed="8.4.154").installed_is_tested


def test_an_unreachable_index_reports_itself_rather_than_failing():
    def opener(*_a, **_k):
        raise OSError("name resolution failed")

    info = updates.check("8.4.154", opener=opener)
    assert info.error and not info.available
    assert "Could not check for updates" in info.summary()


def test_no_environment_means_nothing_to_compare():
    info = updates.check(None, opener=opener_for("8.4.154"))
    assert not info.available
    assert "Set up the environment" in info.summary()


def test_malformed_payload_is_survivable():
    info = updates.check("8.4.154", opener=lambda *_a, **_k: FakeResponse({}))
    assert info.latest is None and not info.available


def test_unparseable_versions_fall_back_to_string_inequality():
    """PyPI has never served a non-PEP-440 version for this package, but the
    comparison must not raise if it ever does -- so "different" stands in for
    "newer", and the user gets a dialog rather than a traceback."""
    assert UpdateInfo("not-a-version", "also-not").available is True
    assert UpdateInfo("nightly", "nightly").available is False
    assert UpdateInfo("8.4.154", "nightly").available is True
