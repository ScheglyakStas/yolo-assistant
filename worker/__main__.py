"""Worker entry point.

    python -m worker <command> --job <path-or-json> [--ipc-port N] [--ipc-token T]

Every command is runnable standalone from a shell with no GUI attached -- that
is a deliberate testability property, not an accident. Without ``--ipc-port``
the channel falls back to framed stdout, so::

    python -m worker fake_train --job '{"epochs": 3}'

prints the same event stream the GUI would have consumed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
import uuid
from pathlib import Path

# Allow both `python -m worker` and direct execution from an extracted dir.
if __package__ in (None, ""):  # pragma: no cover - only hit when run as a file
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worker import i18n  # noqa: E402
from worker.channel import Cancelled, Channel  # noqa: E402
from worker.i18n import tr  # noqa: E402
from worker.protocol import ErrorCode  # noqa: E402

COMMANDS = {
    "probe": "worker.commands.probe",
    "prepare_dataset": "worker.commands.prepare_dataset",
    "autolabel": "worker.commands.autolabel",
    "train": "worker.commands.train",
    "export": "worker.commands.export",
    "infer": "worker.commands.infer",
    "cameras": "worker.commands.cameras",
    "fake_train": "worker.commands.fake_train",
}


def _load_job(raw: str | None) -> dict:
    if not raw:
        return {}
    candidate = Path(raw)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(raw)


def _classify(exc: BaseException) -> tuple[ErrorCode, str]:
    """Map an exception onto a coded error with an actionable hint.

    Kept in one place so the GUI's remedies stay consistent no matter which
    command failed.
    """
    # A command that knows what went wrong says so; see worker/util/errors.py.
    # Guessing is for exceptions thrown out of the depths of a native library.
    declared = getattr(exc, "code", None)
    if isinstance(declared, str):
        try:
            return ErrorCode(declared), str(getattr(exc, "hint", "") or "")
        except ValueError:
            pass

    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return (
            ErrorCode.OUT_OF_MEMORY,
            tr("Reduce the batch size or image size, or close other GPU applications."),
        )
    if "no kernel image is available" in lowered:
        return (
            ErrorCode.NO_CUDA_RUNTIME,
            tr(
                "The installed PyTorch build does not support this GPU. "
                "Re-run environment setup and pick a newer CUDA backend."
            ),
        )
    # ImportError, not just ModuleNotFoundError: a library that checks its own
    # optional dependencies raises the plain one -- transformers answers a
    # missing scipy with "requires the scipy library", which named the cause
    # perfectly and still arrived as an uncoded, unactionable failure.
    if isinstance(exc, ImportError):
        name = getattr(exc, "name", None)
        what = repr(name) if name else tr("a package this step needs")
        return (
            ErrorCode.VENV_BROKEN,
            tr(
                "The managed environment is missing {what}. Open Step 1 and press "
                "Repair: it installs what is missing without downloading everything again.",
                what=what,
            ),
        )
    return ErrorCode.UNKNOWN, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="worker")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--job", default="", help="job spec: a JSON file path or inline JSON")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--ipc-port", type=int, default=0)
    parser.add_argument("--ipc-token", default="")
    args = parser.parse_args(argv)
    # Messages are for the user, so they follow the GUI's language.
    i18n.configure_from_environment()

    job_id = args.job_id or uuid.uuid4().hex[:12]
    try:
        job = _load_job(args.job)
    except ValueError as exc:
        print(f"invalid --job: {exc}", file=sys.stderr)
        return 2

    channel = Channel(job_id, port=args.ipc_port, token=args.ipc_token)
    exit_code = 0
    try:
        module = importlib.import_module(COMMANDS[args.command])
        payload = module.run(channel, job) or {}
        channel.result(True, **payload)
    except Cancelled:
        channel.error(ErrorCode.CANCELLED, tr("Cancelled by user."))
        channel.result(False, cancelled=True)
        exit_code = 130
    except BaseException as exc:  # noqa: BLE001 - the process boundary
        code, hint = _classify(exc)
        channel.error(code, f"{type(exc).__name__}: {exc}", hint=hint, exc=exc)
        channel.result(False)
        traceback.print_exc()
        exit_code = 1
    finally:
        channel.close()
    return exit_code


if __name__ == "__main__":
    # Unbuffered output matters: the GUI reads stdout live for the log pane, and
    # on the fallback transport it carries protocol frames too.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
