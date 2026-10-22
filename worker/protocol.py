"""Wire protocol shared by the GUI process and the worker process.

Two hard constraints govern this module:

1. **Standard library only.** The worker runs under a different interpreter, in
   the managed venv, and imports its own copy of this file. It must be able to
   do so before torch or anything else is installed.
2. **Byte-identical copies.** ``worker/protocol.py`` is a copy of this file and
   CI asserts their hashes match. Never edit one without the other; run
   ``python packaging/sync_protocol.py`` to resync.

Framing
-------
Each message is one JSON object followed by ``\\n``. The primary transport is a
loopback TCP socket. A fallback transport multiplexes frames onto the worker's
stdout, where they are prefixed with ASCII RS (``\\x1e``) so they can be picked
out of the surrounding noise -- tqdm, TensorRT's C++ logger and CUDA warnings
all write to the same file descriptors, and tqdm in particular emits bare
carriage returns that would otherwise destroy line framing.

``FrameDecoder`` therefore tolerates arbitrary junk between frames by design.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterator

PROTOCOL_VERSION = 1

#: Prefix marking a protocol frame on the stdout fallback transport. Chosen
#: because no logging library emits it, so it cannot collide with real output.
RECORD_SEP = "\x1e"

#: A worker that has not spoken within this many seconds, and has produced no
#: stdout either, is considered stalled.
HEARTBEAT_INTERVAL_S = 2.0
STALL_TIMEOUT_S = 60.0

#: How long the GUI waits for the worker to connect back before giving up.
CONNECT_TIMEOUT_S = 30.0


class MsgType(str, Enum):
    """Message kinds. The value is what goes on the wire."""

    # worker -> GUI
    HELLO = "hello"
    READY = "ready"
    PROGRESS = "progress"
    METRICS = "metrics"
    LOG = "log"
    ARTIFACT = "artifact"
    PREVIEW = "preview"
    WARNING = "warning"
    ERROR = "error"
    RESULT = "result"
    HEARTBEAT = "heartbeat"
    # GUI -> worker
    CANCEL = "cancel"


class Phase(str, Enum):
    """Progress phases.

    A single 0-100% bar lies about a training run, which spends time in setup,
    warmup, per-epoch training, per-epoch validation and a final validation pass
    with wildly different per-unit costs. Every ``progress`` message names its
    phase so the UI can show what is actually happening.
    """

    SETUP = "setup"
    DOWNLOAD = "download"
    WARMUP = "warmup"
    TRAIN = "train"
    VAL = "val"
    FINAL_VAL = "final_val"
    EXPORT = "export"
    INFER = "infer"
    PREPROCESS = "preprocess"
    LABEL = "label"
    INSTALL = "install"


class ErrorCode(str, Enum):
    """Coded failures.

    Codes exist so the GUI can offer a concrete remedy instead of dumping a
    traceback at a beginner. Every code should have an entry in the UI's error
    catalogue with a human-readable hint and, where possible, a fix action.
    """

    CANCELLED = "cancelled"
    NO_NVIDIA_DRIVER = "no_nvidia_driver"
    NO_CUDA_RUNTIME = "no_cuda_runtime"
    VENV_BROKEN = "venv_broken"
    DEPENDENCY_INSTALL_FAILED = "dependency_install_failed"
    DOWNLOAD_FAILED = "download_failed"
    DATASET_INVALID = "dataset_invalid"
    QAT_UNAVAILABLE = "qat_unavailable"
    QAT_FORMAT_MISMATCH = "qat_format_mismatch"
    EXPORT_FAILED = "export_failed"
    OUT_OF_MEMORY = "out_of_memory"
    CAMERA_UNAVAILABLE = "camera_unavailable"
    SOURCE_UNREADABLE = "source_unreadable"
    INFERENCE_FAILED = "inference_failed"
    UNKNOWN = "unknown"


class ProtocolError(Exception):
    """Raised when a frame cannot be understood or has the wrong version."""


@dataclass(frozen=True)
class Message:
    """One protocol frame."""

    type: MsgType
    job_id: str
    data: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    ts: float = field(default_factory=time.time)
    v: int = PROTOCOL_VERSION

    def to_obj(self) -> dict[str, Any]:
        return {
            "v": self.v,
            "type": self.type.value,
            "job_id": self.job_id,
            "seq": self.seq,
            "ts": round(self.ts, 6),
            "data": self.data,
        }

    @classmethod
    def from_obj(cls, obj: Any) -> "Message":
        if not isinstance(obj, dict):
            raise ProtocolError(f"frame is {type(obj).__name__}, expected object")
        try:
            version = int(obj["v"])
            raw_type = obj["type"]
            job_id = str(obj["job_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"malformed envelope: {exc}") from exc

        # Refuse a mismatched version outright. In practice this catches a stale
        # extracted worker directory left behind by a previous app build, which
        # would otherwise fail in far more confusing ways.
        if version != PROTOCOL_VERSION:
            raise ProtocolError(
                f"protocol version {version} != {PROTOCOL_VERSION}; "
                "the worker copy is out of date"
            )
        try:
            msg_type = MsgType(raw_type)
        except ValueError as exc:
            raise ProtocolError(f"unknown message type {raw_type!r}") from exc

        data = obj.get("data") or {}
        if not isinstance(data, dict):
            raise ProtocolError("data must be an object")
        return cls(
            type=msg_type,
            job_id=job_id,
            data=data,
            seq=int(obj.get("seq", 0)),
            ts=float(obj.get("ts", 0.0)),
            v=version,
        )

    def with_seq(self, seq: int) -> "Message":
        return replace(self, seq=seq)


def encode(msg: Message, *, record_sep: bool = False) -> bytes:
    """Serialise one message to a newline-terminated frame.

    ``record_sep`` prefixes the frame for the stdout fallback transport.
    """
    line = json.dumps(msg.to_obj(), separators=(",", ":"), ensure_ascii=False)
    if record_sep:
        line = RECORD_SEP + line
    return (line + "\n").encode("utf-8")


def decode(line: str) -> Message:
    """Parse a single frame. Raises :class:`ProtocolError` on anything invalid."""
    text = line.strip()
    if text.startswith(RECORD_SEP):
        text = text[len(RECORD_SEP):]
    if not text:
        raise ProtocolError("empty frame")
    try:
        obj = json.loads(text)
    except ValueError as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc
    return Message.from_obj(obj)


class FrameDecoder:
    """Incremental decoder over a byte stream.

    Feed it whatever arrives; it yields complete messages. Lines that are not
    valid frames are skipped rather than raising, because on the stdout
    transport the stream legitimately contains other output. Decode failures are
    recorded in :attr:`errors` so a caller that expects a clean stream (the TCP
    transport) can still notice them.
    """

    #: Guard against a runaway producer that never emits a newline.
    MAX_BUFFER_BYTES = 8 * 1024 * 1024

    def __init__(self, *, require_record_sep: bool = False) -> None:
        self._buf = bytearray()
        self._require_record_sep = require_record_sep
        self.errors: list[str] = []

    def feed(self, chunk: bytes) -> Iterator[Message]:
        self._buf.extend(chunk)
        if len(self._buf) > self.MAX_BUFFER_BYTES:
            # Drop everything up to the last newline; if there is none, drop it
            # all. Better to lose frames than to grow without bound.
            cut = self._buf.rfind(b"\n")
            del self._buf[: cut + 1 if cut >= 0 else len(self._buf)]
            self.errors.append("buffer overflow; discarded partial data")

        while True:
            idx = self._buf.find(b"\n")
            if idx < 0:
                return
            raw = bytes(self._buf[:idx])
            del self._buf[: idx + 1]
            # tqdm rewrites its bar with bare carriage returns; keep only the
            # final segment of the line so a frame appended after one is seen.
            if b"\r" in raw:
                raw = raw.rsplit(b"\r", 1)[-1]
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception as exc:  # pragma: no cover - decode never raises
                self.errors.append(str(exc))
                continue
            if self._require_record_sep and RECORD_SEP not in text:
                continue
            stripped = text.strip()
            if not stripped:
                continue
            if not self._require_record_sep and not stripped.startswith(("{", RECORD_SEP)):
                continue  # ordinary log output on the stdout transport
            try:
                yield decode(text)
            except ProtocolError as exc:
                self.errors.append(f"{exc}: {stripped[:200]}")


# --------------------------------------------------------------------------- #
# Builders. The worker uses these; keeping them here means the field names are
# defined exactly once for both sides.
# --------------------------------------------------------------------------- #


def hello(job_id: str, *, token: str, pid: int, versions: dict[str, str]) -> Message:
    return Message(MsgType.HELLO, job_id, {"token": token, "pid": pid, "versions": versions})


def ready(job_id: str, **info: Any) -> Message:
    return Message(MsgType.READY, job_id, dict(info))


def progress(
    job_id: str,
    phase: Phase | str,
    current: int,
    total: int | None = None,
    *,
    unit: str = "items",
    eta_s: float | None = None,
    detail: str = "",
) -> Message:
    """One progress sample.

    ``detail`` is free text for a producer whose phase alone is too coarse to be
    useful -- "Training" against "Training - epoch 12 of 100". The UI appends it
    to the phase name, so it should be a fragment rather than a sentence.
    """
    return Message(
        MsgType.PROGRESS,
        job_id,
        {
            "phase": phase.value if isinstance(phase, Phase) else str(phase),
            "current": current,
            "total": total,
            "unit": unit,
            "eta_s": eta_s,
            "detail": detail,
        },
    )


def metrics(job_id: str, epoch: int, values: dict[str, float], *, best: bool = False) -> Message:
    return Message(MsgType.METRICS, job_id, {"epoch": epoch, "values": values, "best": best})


def log(job_id: str, msg: str, *, level: str = "info") -> Message:
    return Message(MsgType.LOG, job_id, {"level": level, "msg": msg})


def artifact(job_id: str, kind: str, path: str, **meta: Any) -> Message:
    return Message(MsgType.ARTIFACT, job_id, {"kind": kind, "path": path, "meta": meta})


def preview(
    job_id: str,
    index: int,
    image: str,
    *,
    width: int,
    height: int,
    mime: str = "image/jpeg",
    detections: list[dict[str, Any]] | None = None,
    timings: dict[str, float] | None = None,
    stats: dict[str, Any] | None = None,
) -> Message:
    """One rendered frame, for a step whose output *is* a picture.

    Step 7 is the only place where what the worker produced cannot be described
    in numbers: "the model found a cat" is a claim about a photo, and the photo
    is in the worker's process. The GUI has no OpenCV and no video decoder --
    deliberately, so that it still launches when the managed environment is
    broken -- so the worker draws the boxes and sends the result as an
    already-compressed image, base64 in the ``image`` field.

    It carries its own numbers rather than leaning on ``progress``: what the
    user is comparing is a picture against the milliseconds that produced *that*
    picture, and pairing them in one message is what stops the two drifting
    apart on screen.

    A frame that is no longer the newest is worth nothing, so these are
    coalesced exactly like progress samples -- see ``Channel.preview``.
    """
    return Message(
        MsgType.PREVIEW,
        job_id,
        {
            "index": index,
            "image": image,
            "mime": mime,
            "width": width,
            "height": height,
            "detections": list(detections or ()),
            "timings": dict(timings or {}),
            "stats": dict(stats or {}),
        },
    )


def warning(job_id: str, code: str, msg: str, *, hint: str = "") -> Message:
    return Message(MsgType.WARNING, job_id, {"code": code, "msg": msg, "hint": hint})


def error(
    job_id: str,
    code: ErrorCode | str,
    msg: str,
    *,
    hint: str = "",
    traceback: str = "",
) -> Message:
    return Message(
        MsgType.ERROR,
        job_id,
        {
            "code": code.value if isinstance(code, ErrorCode) else str(code),
            "msg": msg,
            "hint": hint,
            "traceback": traceback,
        },
    )


def result(job_id: str, ok: bool, **payload: Any) -> Message:
    return Message(MsgType.RESULT, job_id, {"ok": ok, "payload": payload})


def heartbeat(job_id: str) -> Message:
    return Message(MsgType.HEARTBEAT, job_id, {})


def cancel(job_id: str, *, force: bool = False) -> Message:
    return Message(MsgType.CANCEL, job_id, {"force": force})
