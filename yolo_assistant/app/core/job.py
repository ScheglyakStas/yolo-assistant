"""Run a worker subprocess and surface everything it says.

Design notes worth keeping in mind when changing this file:

* **``subprocess.Popen``, not ``QProcess``.** We need Windows'
  ``CREATE_NEW_PROCESS_GROUP`` to deliver a console control event, argv passed as
  a list (this project's own path contains a space), and the ability to kill
  *grandchildren*. Dataloader workers, ``uv``'s children and TensorRT builders
  are all grandchildren, and ``QProcess.kill()`` leaves them running.
* **The control channel is a loopback socket, not stdout.** stdout carries
  tqdm's carriage returns and TensorRT's C++ logging, which would shred line
  framing. stdout is captured verbatim for the log pane and the log file, and
  scanned only for RS-prefixed frames in case the socket could not be bound.
* **Cancellation escalates.** Cooperative cancel over the socket, then a signal
  to the process group, then a hard kill of the whole tree. Each rung exists
  because the one before it can legitimately fail to land.
"""

from __future__ import annotations

import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from PySide6.QtCore import QByteArray, QObject, QTimer, Signal
from PySide6.QtNetwork import QHostAddress, QTcpServer, QTcpSocket

from . import protocol as p
from .i18n import tr
from .protocol import ErrorCode, Message, MsgType
from .worker_env import build_worker_env

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a hard dependency in practice
    psutil = None

#: Seconds to wait after a cooperative cancel before signalling the process.
GRACE_COOPERATIVE_S = 5.0
#: Seconds to wait after signalling before killing the tree.
GRACE_SIGNAL_S = 10.0


class JobState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobSpec:
    """Everything needed to launch one worker command."""

    command: str
    job: dict[str, Any] = field(default_factory=dict)
    #: Interpreter to run. The managed venv's python, or ``sys.executable`` in
    #: ``--fake-worker`` mode.
    python: str = sys.executable
    #: Directory containing the ``worker`` package (loose source).
    worker_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    project_dir: Path | None = None
    cache_dir: Path | None = None
    cwd: Path | None = None
    log_path: Path | None = None
    env_extra: dict[str, str] = field(default_factory=dict)

    #: Run this command line verbatim instead of ``python -m worker <command>``.
    #: Used for the environment bootstrap, where the thing being run is ``uv``
    #: and no managed Python exists yet. Such a process speaks no protocol, so
    #: there is no control socket and cancellation starts at the signal rung --
    #: which is why ``kill_tree`` matters: uv spawns downloaders of its own.
    argv: list[str] | None = None
    #: Whether a protocol ``result`` frame is expected. A raw process reports
    #: success with an exit code instead.
    expect_result: bool = True
    #: Error code reported when a raw process exits non-zero.
    error_code: str = ErrorCode.UNKNOWN.value
    #: Human-readable hint attached to that error.
    error_hint: str = ""


class JobRunner(QObject):
    """Owns one worker process at a time."""

    state_changed = Signal(JobState)
    hello = Signal(dict)          # versions/pid reported by the worker
    ready = Signal(dict)          # imports finished; real work starting
    progress = Signal(dict)       # {phase, current, total, unit, eta_s}
    metrics = Signal(int, dict, bool)   # epoch, values, is_best
    artifact = Signal(str, str, dict)   # kind, path, meta
    preview = Signal(dict)        # one rendered frame: {image, width, height, timings, ...}
    log_line = Signal(str, str)   # text, level ("stdout"/"stderr"/"info"/...)
    warned = Signal(str, str, str)      # code, msg, hint
    failed = Signal(str, str, str, str) # code, msg, hint, traceback
    finished = Signal(bool, dict)       # ok, payload
    stalled = Signal(float)             # seconds since last sign of life

    #: Internal: emitted from the waiter thread. Qt queues it onto the GUI
    #: thread, which is the only thread here with an event dispatcher -- calling
    #: QTimer from the waiter thread silently does nothing.
    _process_exited = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state = JobState.IDLE
        self._spec: JobSpec | None = None
        self._proc: subprocess.Popen | None = None
        self._server: QTcpServer | None = None
        self._socket: QTcpSocket | None = None
        self._decoder = p.FrameDecoder()
        self._stdout_decoder = p.FrameDecoder(require_record_sep=True)
        self._token = ""
        self._job_id = ""
        self._result: tuple[bool, dict] | None = None
        self._error_seen = False
        self._last_sign_of_life = 0.0
        self._log_file = None
        self._readers: list[threading.Thread] = []

        self._connect_timer = QTimer(self)
        self._connect_timer.setSingleShot(True)
        self._connect_timer.timeout.connect(self._on_connect_timeout)

        self._escalate_timer = QTimer(self)
        self._escalate_timer.setSingleShot(True)
        # One permanent connection, dispatched on a stage field. Reconnecting a
        # timer per rung means disconnecting a signal that may have no
        # connections, which PySide warns about.
        self._escalate_stage = ""
        self._escalate_timer.timeout.connect(self._on_escalate)

        self._process_exited.connect(self._on_process_exit)

        self._watchdog = QTimer(self)
        self._watchdog.setInterval(2000)
        self._watchdog.timeout.connect(self._check_liveness)

    # -- public API -------------------------------------------------------- #

    @property
    def state(self) -> JobState:
        return self._state

    @property
    def job_id(self) -> str:
        return self._job_id

    def is_active(self) -> bool:
        return self._state in (JobState.STARTING, JobState.RUNNING, JobState.CANCELLING)

    def start(self, spec: JobSpec) -> None:
        if self.is_active():
            raise RuntimeError("a job is already running")

        self._spec = spec
        self._job_id = secrets.token_hex(6)
        self._token = secrets.token_hex(32)
        self._result = None
        self._error_seen = False
        self._decoder = p.FrameDecoder()
        self._stdout_decoder = p.FrameDecoder(require_record_sep=True)

        raw = spec.argv is not None

        server = None if raw else QTcpServer(self)
        if raw:
            port = 0
        elif not server.listen(QHostAddress.SpecialAddress.LocalHost, 0):
            # Loopback binding blocked (some Windows security software). The
            # worker falls back to RS-prefixed frames on stdout; we lose in-band
            # cancellation but keep the event stream.
            self.log_line.emit(
                tr(
                    "Could not bind a loopback port ({error}); "
                    "falling back to stdout framing.",
                    error=server.errorString(),
                ),
                "warning",
            )
            port = 0
            server = None
        else:
            server.newConnection.connect(self._on_new_connection)
            port = server.serverPort()
        self._server = server

        if raw:
            argv = list(spec.argv or [])
        else:
            argv = [
                spec.python,
                "-u",
                "-m",
                "worker",
                spec.command,
                "--job",
                p.json.dumps(spec.job, separators=(",", ":")),
                "--job-id",
                self._job_id,
            ]
            if port:
                argv += ["--ipc-port", str(port), "--ipc-token", self._token]

        env = build_worker_env(
            spec.worker_dir,
            # Only for a command running *inside* the managed environment. A raw
            # process is uv during the bootstrap, which is not in that
            # environment and must not have its scripts shadowing the system's.
            python=None if raw else spec.python,
            project_dir=spec.project_dir,
            cache_dir=spec.cache_dir,
            extra=spec.env_extra,
        )

        if spec.log_path:
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = spec.log_path.open("a", encoding="utf-8", errors="replace")
            self._log_file.write(f"\n=== job {self._job_id}: {' '.join(argv)} ===\n")

        self._set_state(JobState.STARTING)
        try:
            self._proc = subprocess.Popen(
                argv,
                cwd=str(spec.cwd) if spec.cwd else None,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                bufsize=0,
                **_process_group_kwargs(),
            )
        except OSError as exc:
            self._fail(
                ErrorCode.VENV_BROKEN.value, tr("Could not start worker: {error}", error=exc), ""
            )
            return

        self._last_sign_of_life = time.monotonic()
        self._start_reader(self._proc.stdout, "stdout")
        self._start_reader(self._proc.stderr, "stderr")
        waiter = threading.Thread(target=self._wait_for_exit, name="job-waiter", daemon=True)
        waiter.start()
        self._readers.append(waiter)

        if raw:
            # Nothing will connect, and there is no hello frame to wait for.
            self._set_state(JobState.RUNNING)
        else:
            self._connect_timer.start(int(p.CONNECT_TIMEOUT_S * 1000))
        self._watchdog.start()

    def cancel(self) -> None:
        """Ask the job to stop, escalating if it does not."""
        if not self.is_active() or self._proc is None:
            return

        if self._state is JobState.CANCELLING:
            # Second press: skip the polite rung.
            self._escalate_timer.stop()
            self._escalate_signal()
            return

        self._set_state(JobState.CANCELLING)
        self.log_line.emit(tr("Cancelling..."), "info")
        if self._socket is not None and self._socket.state() == QTcpSocket.SocketState.ConnectedState:
            self._send(p.cancel(self._job_id))
            self._restart_escalation(GRACE_COOPERATIVE_S, "signal")
        else:
            # No control channel: go straight to signalling.
            self._escalate_signal()

    # -- IPC --------------------------------------------------------------- #

    def _on_new_connection(self) -> None:
        server = self._server
        if server is None:
            return
        socket = server.nextPendingConnection()
        if socket is None:
            return
        if self._socket is not None:
            socket.abort()  # only one worker per job
            return
        self._socket = socket
        socket.readyRead.connect(self._on_ready_read)
        socket.disconnected.connect(self._on_disconnected)
        self._connect_timer.stop()

    def _on_ready_read(self) -> None:
        socket = self._socket
        if socket is None:
            return
        data = bytes(socket.readAll().data())
        self._last_sign_of_life = time.monotonic()
        for msg in self._decoder.feed(data):
            self._handle(msg)
        for err in self._decoder.errors:
            self.log_line.emit(f"protocol: {err}", "warning")
        self._decoder.errors.clear()

    def _on_disconnected(self) -> None:
        self._socket = None

    def _send(self, msg: Message) -> None:
        if self._socket is None:
            return
        self._socket.write(QByteArray(p.encode(msg)))
        self._socket.flush()

    def _handle(self, msg: Message) -> None:
        self._last_sign_of_life = time.monotonic()

        if msg.type is MsgType.HELLO:
            if msg.data.get("token") != self._token:
                self.log_line.emit(
                    tr("Rejected a worker connection with a bad token."), "warning"
                )
                if self._socket is not None:
                    self._socket.abort()
                    self._socket = None
                return
            self._set_state(JobState.RUNNING)
            self.hello.emit(dict(msg.data))
        elif msg.type is MsgType.READY:
            self.ready.emit(dict(msg.data))
        elif msg.type is MsgType.PROGRESS:
            self.progress.emit(dict(msg.data))
        elif msg.type is MsgType.METRICS:
            self.metrics.emit(
                int(msg.data.get("epoch", 0)),
                dict(msg.data.get("values") or {}),
                bool(msg.data.get("best", False)),
            )
        elif msg.type is MsgType.ARTIFACT:
            self.artifact.emit(
                str(msg.data.get("kind", "")),
                str(msg.data.get("path", "")),
                dict(msg.data.get("meta") or {}),
            )
        elif msg.type is MsgType.PREVIEW:
            self.preview.emit(dict(msg.data))
        elif msg.type is MsgType.LOG:
            self.log_line.emit(str(msg.data.get("msg", "")), str(msg.data.get("level", "info")))
        elif msg.type is MsgType.WARNING:
            self.warned.emit(
                str(msg.data.get("code", "")),
                str(msg.data.get("msg", "")),
                str(msg.data.get("hint", "")),
            )
        elif msg.type is MsgType.ERROR:
            self._error_seen = True
            self.failed.emit(
                str(msg.data.get("code", ErrorCode.UNKNOWN.value)),
                str(msg.data.get("msg", "")),
                str(msg.data.get("hint", "")),
                str(msg.data.get("traceback", "")),
            )
        elif msg.type is MsgType.RESULT:
            self._result = (bool(msg.data.get("ok")), dict(msg.data.get("payload") or {}))
        # HEARTBEAT needs no handling beyond the liveness touch above.

    # -- process plumbing -------------------------------------------------- #

    def _start_reader(self, stream, name: str) -> None:
        if stream is None:
            return

        def read() -> None:
            for raw in iter(stream.readline, b""):
                self._last_sign_of_life = time.monotonic()
                if self._log_file is not None:
                    try:
                        self._log_file.write(raw.decode("utf-8", errors="replace"))
                        self._log_file.flush()
                    except (OSError, ValueError):
                        pass
                # Only relevant when the socket could not be bound, but cheap
                # enough to always scan: RS-prefixed frames on stdout.
                for msg in self._stdout_decoder.feed(raw):
                    self._handle(msg)
                text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if text and p.RECORD_SEP not in text:
                    self.log_line.emit(text, name)
            try:
                stream.close()
            except OSError:
                pass

        thread = threading.Thread(target=read, name=f"job-{name}", daemon=True)
        thread.start()
        self._readers.append(thread)

    def _wait_for_exit(self) -> None:
        proc = self._proc
        if proc is None:
            return
        code = proc.wait()
        # Hand off to the GUI thread; the result frame may still be in flight.
        self._process_exited.emit(code)

    def _on_process_exit(self, code: int) -> None:
        # A result frame can still be sitting in the socket buffer when the
        # process exits; drain it before deciding how the job ended.
        if self._socket is not None:
            self._socket.waitForReadyRead(200)
            self._on_ready_read()

        self._watchdog.stop()
        self._connect_timer.stop()
        self._escalate_timer.stop()
        if self._log_file is not None:
            try:
                self._log_file.write(f"=== exit {code} ===\n")
                self._log_file.close()
            except (OSError, ValueError):
                pass
            self._log_file = None
        if self._server is not None:
            self._server.close()
            self._server = None
        if self._socket is not None:
            self._socket.abort()
            self._socket = None

        cancelling = self._state is JobState.CANCELLING
        spec = self._spec

        if spec is not None and not spec.expect_result:
            if cancelling:
                self._set_state(JobState.CANCELLED)
                self.finished.emit(False, {"cancelled": True})
            elif code == 0:
                self._set_state(JobState.SUCCEEDED)
                self.finished.emit(True, {"exit_code": 0})
            else:
                self.failed.emit(
                    spec.error_code,
                    tr(
                        "{program} exited with code {code}.",
                        program=Path(spec.argv[0]).name if spec.argv else tr("Process"),
                        code=code,
                    ),
                    spec.error_hint,
                    "",
                )
                self._set_state(JobState.FAILED)
                self.finished.emit(False, {"exit_code": code})
            return

        if self._result is not None:
            ok, payload = self._result
            if ok:
                self._set_state(JobState.SUCCEEDED)
            else:
                self._set_state(JobState.CANCELLED if payload.get("cancelled") or cancelling
                                else JobState.FAILED)
            self.finished.emit(ok, payload)
            return

        if cancelling:
            self._set_state(JobState.CANCELLED)
            self.finished.emit(False, {"cancelled": True})
            return

        # Died without saying why -- a segfault, an OOM kill, or a missing
        # interpreter. Surface it as a coded error rather than silence.
        if not self._error_seen:
            hint = tr(
                "The worker exited without reporting. This usually means a crash in a "
                "native library or the process was killed by the system (out of memory)."
            )
            self.failed.emit(
                ErrorCode.UNKNOWN.value,
                tr("Worker exited with code {code}.", code=code),
                hint,
                "",
            )
        self._set_state(JobState.FAILED)
        self.finished.emit(False, {"exit_code": code})

    # -- cancellation ladder ----------------------------------------------- #

    def _restart_escalation(self, seconds: float, stage: str) -> None:
        self._escalate_stage = stage
        self._escalate_timer.start(int(seconds * 1000))

    def _on_escalate(self) -> None:
        if self._escalate_stage == "signal":
            self._escalate_signal()
        elif self._escalate_stage == "kill":
            self._escalate_kill()

    def _escalate_signal(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self.log_line.emit(tr("Worker did not stop; signalling the process group."), "warning")
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ValueError, AttributeError):
            # A console control event only reaches a process sharing the
            # sender's console, and the installed Windows app has none, so on
            # Windows this is the usual path rather than an edge case. Not
            # ``terminate()``: that ends the worker alone, and the kill rung
            # then finds it gone and leaves its dataloader workers running.
            kill_tree(proc.pid)
            return
        self._restart_escalation(GRACE_SIGNAL_S, "kill")

    def _escalate_kill(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self.log_line.emit(tr("Worker still alive; killing the process tree."), "warning")
        kill_tree(proc.pid)

    # -- liveness ---------------------------------------------------------- #

    def _check_liveness(self) -> None:
        if self._state is not JobState.RUNNING:
            return
        quiet = time.monotonic() - self._last_sign_of_life
        if quiet > p.STALL_TIMEOUT_S:
            self.stalled.emit(quiet)

    def _on_connect_timeout(self) -> None:
        if self._socket is not None or self._proc is None or self._proc.poll() is not None:
            return
        self.log_line.emit(
            tr(
                "Worker did not connect within {seconds:.0f}s; "
                "continuing on stdout framing.",
                seconds=p.CONNECT_TIMEOUT_S,
            ),
            "warning",
        )

    # -- misc -------------------------------------------------------------- #

    def _set_state(self, state: JobState) -> None:
        if state is not self._state:
            self._state = state
            self.state_changed.emit(state)

    def _fail(self, code: str, msg: str, hint: str) -> None:
        self.failed.emit(code, msg, hint, "")
        self._set_state(JobState.FAILED)
        self.finished.emit(False, {})


def kill_tree(pid: int, *, timeout: float = 3.0) -> int:
    """Kill a process and every descendant. Returns how many were killed.

    Grandchildren are the point: Ultralytics' dataloader workers, ``uv``'s
    download children and TensorRT's builder all outlive a bare ``terminate()``
    on the worker, and would otherwise keep a GPU or a lock file busy.
    """
    if psutil is None:  # pragma: no cover
        try:
            os.kill(pid, signal.SIGKILL if os.name != "nt" else signal.SIGTERM)
            return 1
        except OSError:
            return 0
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0

    victims = parent.children(recursive=True) + [parent]
    for victim in victims:
        try:
            victim.kill()
        except psutil.NoSuchProcess:
            pass
    gone, _alive = psutil.wait_procs(victims, timeout=timeout)
    return len(gone)


def _process_group_kwargs() -> dict[str, Any]:
    """Put the worker in its own group so signals reach the whole tree.

    On Windows the worker also gets no window. The installed app is a windowed
    program with no console of its own, so every ``python.exe`` or ``uv.exe``
    it started would otherwise open a black console window for the length of
    the job -- and closing that window kills the job.
    """
    if os.name == "nt":
        return {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        }
    return {"start_new_session": True}
