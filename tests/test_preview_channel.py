"""The frame path, from the worker's channel to the GUI's decoder.

A preview frame is two orders of magnitude larger than every other message this
protocol carries, and it is the only one that is allowed to be dropped. Both of
those are worth testing against the real ``Channel`` and the real
``FrameDecoder`` rather than against the builders alone: a 100 KB line arriving
in 4 KB pieces is exactly the case a framing bug survives everywhere else.
"""

from __future__ import annotations

import base64
import socket
import threading
import time

import pytest

from yolo_assistant.app.core import protocol as p
from yolo_assistant.app.core.protocol import FrameDecoder, MsgType
from worker.channel import Channel


def a_frame(kilobytes: int = 100) -> str:
    return base64.b64encode(b"\xff\xd8" + b"x" * (kilobytes * 1024)).decode("ascii")


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk", [1, 7, 4096, 1024 * 1024])
def test_a_large_frame_survives_any_chunking(chunk):
    image = a_frame()
    raw = p.encode(p.preview("job", 3, image, width=960, height=540))
    decoder = FrameDecoder()

    messages = []
    for start in range(0, len(raw), chunk):
        messages.extend(decoder.feed(raw[start : start + chunk]))

    assert len(messages) == 1
    assert messages[0].data["image"] == image
    assert not decoder.errors


def test_a_frame_between_ordinary_output_is_still_found():
    """The stdout transport genuinely carries a model's own logging around it."""
    decoder = FrameDecoder()
    stream = (
        b"Loading weights...\n"
        + p.encode(p.preview("job", 0, a_frame(8), width=64, height=64))
        + b"Ultralytics 8.4.157 \xf0\x9f\x9a\x80\n"
    )

    messages = list(decoder.feed(stream))

    assert [m.type for m in messages] == [MsgType.PREVIEW]


# --------------------------------------------------------------------------- #
# The channel
# --------------------------------------------------------------------------- #


@pytest.fixture
def listener():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    accepted: list[socket.socket] = []

    def accept():
        connection, _address = server.accept()
        accepted.append(connection)

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield server, accepted
    for connection in accepted:
        connection.close()
    server.close()


def drain(connection: socket.socket, *, seconds: float = 1.0) -> list:
    """Everything the worker sent, decoded."""
    connection.settimeout(0.1)
    decoder = FrameDecoder()
    messages: list = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            chunk = connection.recv(65536)
        except socket.timeout:
            continue
        if not chunk:
            break
        messages.extend(decoder.feed(chunk))
    return messages


def test_a_frame_reaches_the_other_end(listener):
    server, accepted = listener
    channel = Channel("job", port=server.getsockname()[1])
    try:
        channel.preview(1, a_frame(), width=960, height=540, stats={"frames": 1})
        channel.result(True)
        previews = [m for m in drain(accepted[0]) if m.type is MsgType.PREVIEW]
    finally:
        channel.close()

    assert len(previews) == 1
    assert previews[0].data["stats"]["frames"] == 1


def test_only_the_newest_frame_survives(listener):
    """A frame that is no longer the newest is a picture of the past."""
    server, accepted = listener
    channel = Channel("job", port=server.getsockname()[1])
    try:
        for index in range(8):
            channel.preview(index, a_frame(4), width=64, height=64)
        channel.result(True)
        previews = [m for m in drain(accepted[0]) if m.type is MsgType.PREVIEW]
    finally:
        channel.close()

    indices = [m.data["index"] for m in previews]
    assert indices, "no frame arrived at all"
    assert indices[-1] == 7, "the newest frame must always arrive"
    assert len(indices) < 8, "frames offered faster than they can be sent must coalesce"


def test_handing_over_a_frame_never_waits_on_the_reader(listener):
    """A busy GUI must not stall the inference this step is measuring.

    The reader here never reads, so the socket fills and a write would block.
    Offering frames still has to return promptly, because that call happens on
    the thread between two inferences.
    """
    server, _accepted = listener
    channel = Channel("job", port=server.getsockname()[1])
    try:
        started = time.monotonic()
        for index in range(60):
            channel.preview(index, a_frame(64), width=960, height=540)
        elapsed = time.monotonic() - started
    finally:
        channel.close()

    assert elapsed < 1.0, f"handing over 60 frames took {elapsed:.1f}s"


def test_the_rate_is_the_preview_rate_not_the_producers(listener):
    """A producer offering as fast as it can is told when a frame is due."""
    server, accepted = listener
    channel = Channel("job", port=server.getsockname()[1])
    try:
        due = 0
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if channel.preview_due():
                due += 1
                channel.preview(due, a_frame(4), width=64, height=64)
            time.sleep(0.001)
        channel.result(True)
        drain(accepted[0], seconds=0.3)
    finally:
        channel.close()

    # Thirty a second, give or take the scheduler. The bug this replaces
    # offered twelve and delivered ten.
    assert 20 <= due <= 40, f"{due} frames were due in a second"


def test_a_source_at_the_preview_rate_loses_nothing(listener):
    """The rounding trap: a 30 fps camera against a 30 Hz deadline.

    Comparing "has a whole period passed" against a loop of almost exactly one
    period misses by a hair every time, waits another period, and halves the
    rate. This is the regression test for the camera that displayed at ten.
    """
    server, accepted = listener
    channel = Channel("job", port=server.getsockname()[1])
    try:
        offered = shown = 0
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            time.sleep(1 / 30)  # a camera delivering thirty frames a second
            offered += 1
            if channel.preview_due():
                shown += 1
                channel.preview(shown, a_frame(4), width=64, height=64)
        channel.result(True)
        drain(accepted[0], seconds=0.3)
    finally:
        channel.close()

    assert shown >= offered - 1, f"only {shown} of {offered} frames were shown"


def test_the_stdout_transport_refuses_frames_out_loud(capsysbinary):
    """Frames on stdout would fill the job log with base64 nobody can see."""
    channel = Channel("job", port=0)
    try:
        for index in range(3):
            channel.preview(index, a_frame(), width=960, height=540)
    finally:
        channel.close()

    decoder = FrameDecoder()
    messages = list(decoder.feed(capsysbinary.readouterr().out))
    logs = [m for m in messages if m.type is MsgType.LOG]

    assert not [m for m in messages if m.type is MsgType.PREVIEW]
    assert len([m for m in logs if "loopback" in m.data["msg"]]) == 1
