"""Worker-side end of the GUI <-> worker channel.

Responsibilities, in rough order of how much trouble they save:

* **Throttling.** Training emits per batch. A 3000-iteration epoch would spend
  more time in Qt's event loop than in CUDA if every update crossed the wire,
  so progress is coalesced to ``FLUSH_HZ`` and only the newest sample survives.
  Messages that carry information which cannot be reconstructed -- metrics,
  artifacts, warnings, errors, results -- are never dropped.
* **Cancellation.** A reader thread watches for ``cancel`` frames and sets
  :attr:`cancelled`. Long-running commands poll it. A second cancel sets
  :attr:`force_cancelled`, which asks the command to unwind immediately rather
  than finishing the current unit of work.
* **Liveness.** A heartbeat every ``HEARTBEAT_INTERVAL_S`` so the GUI can tell
  "busy inside a long CUDA call" apart from "wedged".
* **Transport fallback.** If the loopback socket cannot be established -- rare,
  but Windows security software does block it -- frames go to stdout prefixed
  with ASCII RS instead. Progress still flows; only in-band cancellation is
  lost, and the GUI falls back to its signal ladder.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import traceback
from typing import Any

from . import protocol as p
from .i18n import tr
from .protocol import ErrorCode, Message, MsgType, Phase

FLUSH_HZ = 10.0
FLUSH_INTERVAL_S = 1.0 / FLUSH_HZ

#: Preview frames get their own, much higher rate. Ten a second is plenty for a
#: number that is only read; it is visibly stuttering for a picture of a moving
#: scene, and a camera delivering thirty is the common case. Measured on a
#: 1080p frame scaled for display: 0.5 ms to encode, 33 KB on the wire and
#: 1.9 ms to decode and paint -- so thirty a second costs a few per cent of
#: each process and about a megabyte a second of loopback.
PREVIEW_HZ = 30.0
PREVIEW_INTERVAL_S = 1.0 / PREVIEW_HZ

#: How early a frame may be sent. Without it, a source running at almost
#: exactly the preview rate loses frames to rounding: a 33.4 ms loop against a
#: 33.3 ms deadline misses by a hair, waits another whole period, and delivers
#: fifteen a second instead of thirty.
PREVIEW_SLACK_S = PREVIEW_INTERVAL_S * 0.15

#: The flush thread has to wake often enough to serve the faster of the two.
TICK_INTERVAL_S = min(FLUSH_INTERVAL_S, PREVIEW_INTERVAL_S)


class Channel:
    """Connected, thread-safe message channel to the GUI."""

    def __init__(self, job_id: str, *, port: int = 0, token: str = "") -> None:
        self.job_id = job_id
        self.cancelled = threading.Event()
        self.force_cancelled = threading.Event()

        self._sock: socket.socket | None = None
        self._lock = threading.RLock()  # re-entrant: send() may be re-entered
        self._seq = 0
        self._closed = threading.Event()
        #: Guards the two coalescing slots and nothing else. Deliberately not
        #: the send lock: handing over a frame must never wait on a slow
        #: consumer, which is the whole point of coalescing one.
        self._pending_lock = threading.Lock()
        self._pending_progress: Message | None = None
        self._pending_preview: Message | None = None
        self._preview_refused = False
        self._last_flush = 0.0
        self._preview_deadline = 0.0
        self._use_stdout = True

        if port:
            try:
                sock = socket.create_connection(("127.0.0.1", port), timeout=10.0)
                sock.settimeout(None)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = sock
                self._use_stdout = False
            except OSError as exc:
                # Not fatal: fall back to stdout framing and say so, so the
                # failure shows up in the log pane rather than as silence.
                self._stdout_note = tr(
                    "IPC socket unavailable ({error}); using stdout framing", error=exc
                )
            else:
                self._stdout_note = ""
        else:
            self._stdout_note = ""

        self.send(
            p.hello(
                job_id,
                token=token,
                pid=os.getpid(),
                versions=_collect_versions(),
            )
        )
        if self._stdout_note:
            self.log(self._stdout_note, level="warning")

        self._threads = [
            threading.Thread(target=self._flush_loop, name="channel-flush", daemon=True),
            threading.Thread(target=self._heartbeat_loop, name="channel-hb", daemon=True),
        ]
        if self._sock is not None:
            self._threads.append(
                threading.Thread(target=self._read_loop, name="channel-read", daemon=True)
            )
        for t in self._threads:
            t.start()

    # -- sending ---------------------------------------------------------- #

    def send(self, msg: Message) -> None:
        """Send immediately. Never drops."""
        if self._closed.is_set():
            return
        with self._lock:
            self._seq += 1
            frame = p.encode(msg.with_seq(self._seq), record_sep=self._use_stdout)
            self._write(frame)

    def _write(self, frame: bytes) -> None:
        if self._sock is not None:
            try:
                self._sock.sendall(frame)
                return
            except OSError:
                # The GUI went away. Downgrade to stdout so the rest of the run
                # is at least visible in the log file.
                self._sock = None
                self._use_stdout = True
        try:
            sys.stdout.buffer.write(frame)
            sys.stdout.buffer.flush()
        except (OSError, ValueError):
            pass

    # -- convenience builders --------------------------------------------- #

    def ready(self, **info: Any) -> None:
        self.send(p.ready(self.job_id, **info))

    def progress(
        self,
        phase: Phase | str,
        current: int,
        total: int | None = None,
        *,
        unit: str = "items",
        eta_s: float | None = None,
        detail: str = "",
    ) -> None:
        """Record progress. Coalesced; only the newest sample is sent."""
        msg = p.progress(
            self.job_id, phase, current, total, unit=unit, eta_s=eta_s, detail=detail
        )
        with self._lock:
            self._pending_progress = msg
        # Send straight away if we are past the interval, so a slow producer
        # (one image every few seconds) is not delayed by the flush loop.
        self._flush_pending(previews=False)

    def preview_due(self) -> bool:
        """Whether a frame sent now would actually be shown.

        Asked *before* a frame is drawn and compressed, because that work costs
        real milliseconds and they would come out of the number this step
        exists to measure. One clock decides both whether to encode and when to
        send; two clocks beating against each other is how a camera delivering
        thirty frames a second ends up displaying ten.
        """
        if self._use_stdout:
            self._refuse_previews()
            return False
        return time.monotonic() + PREVIEW_SLACK_S >= self._preview_deadline

    def preview(
        self,
        index: int,
        image: str,
        *,
        width: int,
        height: int,
        mime: str = "image/jpeg",
        detections: list[dict[str, Any]] | None = None,
        timings: dict[str, float] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        """Offer a rendered frame. Coalesced; only the newest sample is sent.

        Exactly the progress rule, for a stronger reason. A stale progress
        sample is merely out of date; a stale *frame* is a picture of what the
        camera was pointing at a second ago, shown beside a millisecond figure
        measured now. If the GUI cannot keep up -- and the socket blocking is
        how we find that out -- the right answer is to skip to the present
        rather than to play the backlog.

        Nothing is sent on the stdout transport. Frames are two orders of
        magnitude larger than every other message, stdout is also what the job
        log file is written from, and a camera session would fill a disk with
        base64 for a picture nobody can see anyway. It is refused once, out
        loud, and the timings keep flowing.
        """
        if self._use_stdout:
            self._refuse_previews()
            return
        msg = p.preview(
            self.job_id,
            index,
            image,
            width=width,
            height=height,
            mime=mime,
            detections=detections,
            timings=timings,
            stats=stats,
        )
        now = time.monotonic()
        # Advanced by a whole period rather than restarted from now, so that a
        # loop running at the preview rate keeps its place instead of slipping
        # a little later on every pass. A producer that has fallen behind
        # resynchronises rather than sending a burst to catch up.
        self._preview_deadline += PREVIEW_INTERVAL_S
        if self._preview_deadline < now:
            self._preview_deadline = now + PREVIEW_INTERVAL_S
        with self._pending_lock:
            self._pending_preview = msg
        # Deliberately not sent from here. Writing a frame can block on a GUI
        # that is busy, and blocking this caller would stall the inference this
        # step exists to measure. The flush thread carries it, and a frame
        # superseded while that thread is blocked is simply dropped.

    def _refuse_previews(self) -> None:
        if self._preview_refused:
            return
        self._preview_refused = True
        self.log(
            tr(
                "The live picture needs the loopback control socket, which could not be "
                "opened, so only the numbers are shown. Everything else in this step works."
            ),
            level="warning",
        )

    def metrics(self, epoch: int, values: dict[str, float], *, best: bool = False) -> None:
        self.send(p.metrics(self.job_id, epoch, values, best=best))

    def log(self, msg: str, *, level: str = "info") -> None:
        self.send(p.log(self.job_id, msg, level=level))

    def artifact(self, kind: str, path: str, **meta: Any) -> None:
        self.send(p.artifact(self.job_id, kind, str(path), **meta))

    def warning(self, code: str, msg: str, *, hint: str = "") -> None:
        self.send(p.warning(self.job_id, code, msg, hint=hint))

    def error(
        self,
        code: ErrorCode | str,
        msg: str,
        *,
        hint: str = "",
        exc: BaseException | None = None,
    ) -> None:
        tb = "".join(traceback.format_exception(exc)) if exc is not None else ""
        self.send(p.error(self.job_id, code, msg, hint=hint, traceback=tb))

    def result(self, ok: bool, **payload: Any) -> None:
        # Forced: the last frame of a run is the one the user is left looking
        # at, and it must not be dropped for arriving early.
        self._flush_pending(force=True)
        self.send(p.result(self.job_id, ok, **payload))

    # -- cancellation ------------------------------------------------------ #

    def check_cancelled(self) -> None:
        """Raise :class:`Cancelled` if the GUI asked us to stop.

        Commands call this at a natural boundary -- per image, per batch, per
        epoch -- so the process unwinds through normal exception handling and
        leaves consistent files behind.
        """
        if self.cancelled.is_set():
            raise Cancelled()

    # -- lifecycle --------------------------------------------------------- #

    def close(self) -> None:
        self._flush_pending(force=True)
        self._closed.set()
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # -- internals --------------------------------------------------------- #

    def _flush_pending(self, *, force: bool = False, previews: bool = True) -> None:
        """Send whatever is pending and due.

        ``previews`` is false for the inline call from :meth:`progress`, which
        runs on the producer's own thread: a progress sample is a few hundred
        bytes and a frame is tens of kilobytes, and only one of those is worth
        risking a stall for.

        The slots are taken under their own lock and written outside it, so a
        blocked write cannot keep a producer from handing over its next sample.
        """
        now = time.monotonic()
        with self._pending_lock:
            progress = preview = None
            if force or now - self._last_flush >= FLUSH_INTERVAL_S:
                progress, self._pending_progress = self._pending_progress, None
                self._last_flush = now
            if previews:
                preview, self._pending_preview = self._pending_preview, None
        for msg in (progress, preview):
            if msg is not None:
                self.send(msg)

    def _flush_loop(self) -> None:
        while not self._closed.wait(TICK_INTERVAL_S):
            self._flush_pending()

    def _heartbeat_loop(self) -> None:
        while not self._closed.wait(p.HEARTBEAT_INTERVAL_S):
            self.send(p.heartbeat(self.job_id))

    def _read_loop(self) -> None:
        decoder = p.FrameDecoder()
        sock = self._sock
        if sock is None:
            return
        while not self._closed.is_set():
            try:
                chunk = sock.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            for msg in decoder.feed(chunk):
                if msg.type is MsgType.CANCEL:
                    if self.cancelled.is_set() or msg.data.get("force"):
                        self.force_cancelled.set()
                    self.cancelled.set()
        # A closed socket means the GUI is gone. Treat that as a cancel so the
        # worker does not keep a GPU busy for a window nobody is watching.
        if not self._closed.is_set():
            self.cancelled.set()


class Cancelled(Exception):
    """Raised inside a command when the user cancels the job."""


def _collect_versions() -> dict[str, str]:
    """Report what is actually importable, without requiring any of it."""
    versions = {"python": sys.version.split()[0]}
    for name, attr in (
        ("torch", "__version__"),
        ("ultralytics", "__version__"),
        ("transformers", "__version__"),
        ("cv2", "__version__"),
    ):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, attr, "unknown"))
        except Exception:
            continue
    try:
        import torch  # noqa: PLC0415

        versions["cuda"] = str(torch.version.cuda or "")
        versions["cuda_available"] = str(torch.cuda.is_available())
    except Exception:
        pass
    return versions
