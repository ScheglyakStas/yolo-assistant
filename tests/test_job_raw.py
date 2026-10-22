"""Jobs that are not our worker.

The environment bootstrap runs ``uv``, which knows nothing about the protocol:
it reports success with an exit code and speaks no frames at all. That mode has
to carry the same guarantees as a worker job -- streamed output, a coded failure
rather than silence, and a cancel that actually reaches the process tree --
because it is the only mode available on a machine where nothing is installed
yet.
"""

import sys
import time

import psutil
import pytest

from yolo_assistant.app.core.job import JobRunner, JobSpec, JobState, kill_tree


@pytest.fixture
def runner(app):
    r = JobRunner()
    yield r
    proc = r._proc
    if proc is not None and proc.poll() is None:
        kill_tree(proc.pid)


def pump(app, predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


def raw(code: str, **kwargs) -> JobSpec:
    return JobSpec(
        command="uv",
        argv=[sys.executable, "-u", "-c", code],
        expect_result=False,
        **kwargs,
    )


def collect(runner):
    box = {"finished": None, "logs": [], "errors": [], "states": []}
    runner.finished.connect(lambda ok, payload: box.__setitem__("finished", (ok, payload)))
    runner.log_line.connect(lambda text, level: box["logs"].append((level, text)))
    runner.failed.connect(lambda c, m, h, tb: box["errors"].append((c, m, h)))
    runner.state_changed.connect(box["states"].append)
    return box


def test_exit_zero_is_success_without_any_protocol_frame(app, runner):
    box = collect(runner)
    runner.start(raw("print('done')"))

    assert pump(app, lambda: box["finished"] is not None)
    ok, payload = box["finished"]
    assert ok is True
    assert runner.state is JobState.SUCCEEDED
    assert payload["exit_code"] == 0
    assert not box["errors"], "a clean exit must not report an error"


def test_output_is_streamed_to_the_log(app, runner):
    box = collect(runner)
    runner.start(raw("import sys; print('hello'); print('oops', file=sys.stderr)"))
    assert pump(app, lambda: box["finished"] is not None)

    assert ("stdout", "hello") in box["logs"]
    assert ("stderr", "oops") in box["logs"]


def test_a_nonzero_exit_reports_the_caller_s_code_and_hint(app, runner):
    box = collect(runner)
    runner.start(
        raw(
            "raise SystemExit(3)",
            error_code="dependency_install_failed",
            error_hint="Try the next backend down.",
        )
    )
    assert pump(app, lambda: box["finished"] is not None)

    ok, payload = box["finished"]
    assert ok is False and payload["exit_code"] == 3
    assert runner.state is JobState.FAILED
    assert len(box["errors"]) == 1
    code, message, hint = box["errors"][0]
    assert code == "dependency_install_failed"
    assert "code 3" in message
    assert hint == "Try the next backend down."


def test_no_control_socket_is_opened_for_a_raw_process(app, runner):
    """It would never be connected to, and the connect timeout would fire a
    spurious warning in the middle of a long install."""
    box = collect(runner)
    runner.start(raw("print('x')"))
    assert runner._server is None
    assert pump(app, lambda: box["finished"] is not None)
    assert not any("did not connect" in text for _level, text in box["logs"])


def test_it_is_running_immediately_rather_than_waiting_for_a_hello(app, runner):
    box = collect(runner)
    runner.start(raw("import time; time.sleep(2)"))
    assert JobState.RUNNING in box["states"]
    runner.cancel()
    assert pump(app, lambda: box["finished"] is not None)


def test_cancel_reaches_a_process_that_ignores_the_signal(app, runner):
    """uv spawns children, and a raw process has no cooperative cancel path --
    so this has to escalate all the way to killing the tree."""
    box = collect(runner)
    runner.start(
        raw(
            "import signal, subprocess, sys, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "kids = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            "        for _ in range(2)]\n"
            "print('CHILDREN ' + ','.join(str(k.pid) for k in kids), flush=True)\n"
            "time.sleep(120)\n"
        )
    )
    assert pump(app, lambda: any("CHILDREN" in text for _lvl, text in box["logs"]), timeout=20)
    children = [
        int(pid)
        for _lvl, text in box["logs"]
        if text.startswith("CHILDREN ")
        for pid in text.split(" ", 1)[1].split(",")
    ]
    parent = runner._proc.pid

    runner.cancel()
    assert pump(app, lambda: box["finished"] is not None, timeout=40), "cancel never completed"
    assert runner.state is JobState.CANCELLED
    assert box["finished"] == (False, {"cancelled": True})

    assert pump(app, lambda: not any(_alive(pid) for pid in [parent, *children]), timeout=10), (
        "the process tree survived cancellation"
    )


def _alive(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False
    return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
