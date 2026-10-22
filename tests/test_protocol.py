"""Protocol framing: the assumptions everything else rests on."""

import hashlib
from pathlib import Path

import pytest

from yolo_assistant.app.core import protocol as p
from yolo_assistant.app.core.protocol import (
    FrameDecoder,
    Message,
    MsgType,
    Phase,
    ProtocolError,
)

ROOT = Path(__file__).resolve().parents[1]


def test_worker_copy_is_byte_identical():
    """The worker imports its own copy under a different interpreter."""
    app_copy = (ROOT / "yolo_assistant/app/core/protocol.py").read_bytes()
    worker_copy = (ROOT / "worker/protocol.py").read_bytes()
    assert hashlib.sha256(app_copy).hexdigest() == hashlib.sha256(worker_copy).hexdigest(), (
        "worker/protocol.py has drifted; copy app/core/protocol.py over it"
    )


def test_round_trip_preserves_payload():
    msg = p.metrics("job1", 4, {"metrics/mAP50(B)": 0.81}, best=True)
    decoded = p.decode(p.encode(msg).decode())
    assert decoded.type is MsgType.METRICS
    assert decoded.data["values"]["metrics/mAP50(B)"] == pytest.approx(0.81)
    assert decoded.data["best"] is True


def test_a_preview_frame_survives_the_wire():
    """The one message that carries an image rather than a number."""
    msg = p.preview(
        "job1",
        7,
        "QkFTRTY0",
        width=960,
        height=540,
        detections=[{"name": "cat", "conf": 0.91}],
        timings={"inference_ms": 8.2},
        stats={"frames": 7, "fps": 24.0},
    )

    decoded = p.decode(p.encode(msg).decode())

    assert decoded.type is MsgType.PREVIEW
    assert decoded.data["index"] == 7
    assert decoded.data["image"] == "QkFTRTY0"
    assert decoded.data["detections"][0]["name"] == "cat"
    assert decoded.data["stats"]["frames"] == 7


def test_record_separator_is_stripped():
    framed = p.encode(p.heartbeat("j"), record_sep=True).decode()
    assert framed.startswith(p.RECORD_SEP)
    assert p.decode(framed).type is MsgType.HEARTBEAT


def test_version_mismatch_is_refused():
    """Guards against a stale extracted worker directory after an app upgrade."""
    obj = p.heartbeat("j").to_obj()
    obj["v"] = p.PROTOCOL_VERSION + 1
    with pytest.raises(ProtocolError, match="out of date"):
        Message.from_obj(obj)


@pytest.mark.parametrize(
    "bad",
    ['{"v":1}', "not json at all", "[]", '{"v":1,"type":"nope","job_id":"j"}'],
)
def test_malformed_frames_raise(bad):
    with pytest.raises(ProtocolError):
        p.decode(bad)


def test_decoder_skips_ordinary_output():
    """stdout carries tqdm bars and CUDA warnings alongside frames."""
    decoder = FrameDecoder()
    stream = (
        b"Ultralytics 8.4.154 starting\n"
        + p.encode(p.progress("j", Phase.TRAIN, 1, 10))
        + b"WARNING: something from a C library\n"
        + p.encode(p.result("j", True))
    )
    msgs = list(decoder.feed(stream))
    assert [m.type for m in msgs] == [MsgType.PROGRESS, MsgType.RESULT]
    assert decoder.errors == []


def test_decoder_survives_tqdm_carriage_returns():
    """A bare \\r bar on the same line as a frame must not eat the frame."""
    decoder = FrameDecoder()
    frame = p.encode(p.progress("j", Phase.TRAIN, 5, 10), record_sep=True)
    stream = b"  30%|###       | 3/10 [00:01<00:02]\r  40%|####      | 4/10\r" + frame
    msgs = list(decoder.feed(stream))
    assert [m.type for m in msgs] == [MsgType.PROGRESS]
    assert msgs[0].data["current"] == 5


def test_decoder_handles_split_chunks():
    decoder = FrameDecoder()
    frame = p.encode(p.ready("j", epochs=3))
    assert list(decoder.feed(frame[:12])) == []
    msgs = list(decoder.feed(frame[12:]))
    assert len(msgs) == 1 and msgs[0].data["epochs"] == 3


def test_decoder_require_record_sep_ignores_plain_json():
    """On the stdout transport, only RS-prefixed lines are ours."""
    decoder = FrameDecoder(require_record_sep=True)
    stream = b'{"v":1,"type":"log","job_id":"x","data":{}}\n' + p.encode(
        p.heartbeat("j"), record_sep=True
    )
    assert [m.type for m in decoder.feed(stream)] == [MsgType.HEARTBEAT]


def test_decoder_records_errors_without_raising():
    decoder = FrameDecoder()
    list(decoder.feed(b'{"v":1,"type":"bogus","job_id":"j"}\n'))
    assert decoder.errors and "unknown message type" in decoder.errors[0]


def test_buffer_overflow_is_bounded():
    decoder = FrameDecoder()
    list(decoder.feed(b"x" * (FrameDecoder.MAX_BUFFER_BYTES + 1024)))
    assert decoder.errors and "overflow" in decoder.errors[0]
    # And it still works afterwards.
    assert list(decoder.feed(b"\n" + p.encode(p.heartbeat("j"))))
