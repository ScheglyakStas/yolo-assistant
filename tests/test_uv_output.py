"""Parsing uv's output -- the one place in this app where that is allowed.

The transcripts below are real output captured from uv 0.12.15 with its stdout
piped, which is how the app always runs it. If a future uv changes this format
these tests fail here, in a second, rather than showing a frozen progress bar
during a ten-minute download.
"""

from yolo_assistant.app.core.bootstrap.uv_output import UvOutputParser

PIP_INSTALL = """Using Python 3.12.14 environment at: uvtest-venv
Resolved 10 packages in 648ms
Downloading sympy (6.0MiB)
Downloading numpy (15.9MiB)
Downloading onnx (18.2MiB)
 Downloaded sympy
 Downloaded numpy
 Downloaded onnx
Prepared 10 packages in 3.95s
Installed 10 packages in 69ms
 + numpy==2.5.3
 + onnx==1.22.0
"""

PYTHON_INSTALL = """Downloading cpython-3.12.14-linux-x86_64-gnu (download) (32.6MiB)
 Downloaded cpython-3.12.14-linux-x86_64-gnu (download)
Installed Python 3.12.14 in 3.16s
 + cpython-3.12.14-linux-x86_64-gnu (python3.12)
"""


def parse(transcript):
    parser = UvOutputParser()
    seen = []
    for line in transcript.splitlines():
        parser.feed(line)
        seen.append((parser.state.current, parser.state.total))
    return parser, seen


def test_the_resolved_count_becomes_the_total():
    parser, _ = parse(PIP_INSTALL)
    assert parser.state.total == 10


def test_progress_never_goes_backwards():
    """The reason the announced-download count is not used as the denominator."""
    _parser, seen = parse(PIP_INSTALL)
    currents = [current for current, _total in seen]
    assert currents == sorted(currents)


def test_it_reaches_the_end_even_though_some_packages_came_from_cache():
    parser, _ = parse(PIP_INSTALL)
    assert parser.state.current == parser.state.total == 10
    assert parser.state.prepared


def test_the_package_being_downloaded_is_named():
    parser = UvOutputParser()
    parser.feed("Resolved 137 packages in 1s")
    parser.feed("Downloading torch (821.2MiB)")
    parser.feed("Downloading filelock (0.1MiB)")
    assert parser.state.describe() == "Downloading torch"  # the biggest one, not the newest


def test_byte_totals_scale_to_the_download():
    parser = UvOutputParser()
    parser.feed("Downloading torch (3000.0MiB)")
    parser.feed("Downloading nvidia-cudnn (2000.0MiB)")
    parser.feed(" Downloaded torch")
    assert parser.state.describe_bytes() == "2.9 / 4.9 GB"


def test_small_installs_are_reported_in_megabytes():
    parser = UvOutputParser()
    parser.feed("Downloading numpy (16.0MiB)")
    assert parser.state.describe_bytes() == "0 / 16 MB"


def test_installed_versions_are_recorded():
    parser, _ = parse(PIP_INSTALL)
    assert parser.state.installed == {"numpy": "2.5.3", "onnx": "1.22.0"}


def test_the_size_is_found_even_when_it_is_not_the_first_bracket():
    """uv's Python downloads read "... (download) (32.6MiB)"; a regex anchored
    on the first parenthesis silently reports a zero-byte download."""
    parser = UvOutputParser()
    parser.feed("Downloading cpython-3.12.14-linux-x86_64-gnu (download) (32.6MiB)")
    assert parser.state.announced_bytes == 32.6 * 1024**2


def test_python_install_transcript():
    parser, _ = parse(PYTHON_INSTALL)
    assert parser.state.python_version == "3.12.14"


def test_blank_and_unrecognised_lines_are_ignored():
    parser = UvOutputParser()
    assert not parser.feed("")
    assert not parser.feed("   ")
    assert not parser.feed("some unrelated chatter")


def test_resolution_failures_become_an_actionable_hint():
    parser = UvOutputParser()
    parser.feed("error: No solution found when resolving dependencies:")
    assert "next backend down" in parser.failure_hint()


def test_network_failures_point_at_the_proxy():
    parser = UvOutputParser()
    parser.feed("error: Failed to connect to pypi.org: timed out")
    assert "proxy" in parser.failure_hint()


def test_disk_and_permission_failures_are_distinguished():
    disk = UvOutputParser()
    disk.feed("error: No space left on device (os error 28)")
    assert "disk space" in disk.failure_hint()

    locked = UvOutputParser()
    locked.feed("error: Permission denied (os error 5)")
    assert "Antivirus" in locked.failure_hint()


def test_no_errors_means_no_hint():
    parser, _ = parse(PIP_INSTALL)
    assert parser.failure_hint() == ""
