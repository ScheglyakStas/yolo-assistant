"""Which cameras this machine actually has, found by opening them.

There is no portable way to ask an operating system what its cameras are, so
this opens candidates and reports the ones that answer. That is slower and
ruder than a registry lookup -- it briefly lights the indicator LED -- and it
is the only method that answers the question the user is really asking, which
is not "is a camera plugged in" but "can this app read from it right now".

Three things are worth knowing.

**Opening is not enough; a frame has to come back.** On Linux a single USB
webcam registers two device nodes: the capture stream and a metadata stream
beside it. Both open. Only one of them ever produces an image, so a list built
from ``isOpened()`` offers the user a camera that will never show anything.

**A camera another application is holding is not a camera.** Most drivers hand
one out to a single process at a time, so a device in use by a video call
simply fails to open. Reporting it anyway would put an entry in the list that
cannot be chosen; leaving it out matches what the user can actually do, and the
page says why a camera might be missing.

**The names come from the system, where there is one to ask.** Linux publishes
a human name per node under ``/sys``; Windows and macOS do not expose one
through OpenCV, so those fall back to the index. A list of "Camera 0, Camera 1"
is a poor thing to choose from, and worth avoiding wherever it can be.

**Each camera is probed the way it will be used.** A camera opened with no
request runs at whatever the driver likes, which for most webcams is 640x480 --
on hardware perfectly capable of 1920x1080 at the same frame rate. Listing that
default would describe the camera by a mode Step 7 is not going to use, so the
probe asks for exactly what a run asks for by default and reports what it was
given. A run with a size or frame rate chosen by hand says what it got itself.

This command runs in the managed environment because OpenCV lives there, and it
is deliberately cheap: no torch, no model, no GPU.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from worker.channel import Channel
from worker.i18n import tr
from worker.protocol import Phase

#: How many indices to try when the platform cannot list its own devices.
#: Past a handful these are all misses costing a timeout each, and a machine
#: with nine cameras is not the machine this app is for.
DEFAULT_MAX_INDEX = 5

#: Linux publishes one directory per video node here, each with the device's
#: advertised name. Where it exists, it replaces guessing at indices entirely.
V4L_ROOT = Path("/sys/class/video4linux")


def run(channel: Channel, job: dict) -> dict:
    import cv2

    candidates = _candidates(int(job.get("max_index") or DEFAULT_MAX_INDEX))
    channel.progress(Phase.SETUP, 0, len(candidates) or 1, unit="devices")

    found: list[dict] = []
    for position, (index, name) in enumerate(candidates, start=1):
        channel.check_cancelled()
        described = _probe(cv2, index, name)
        if described is not None:
            found.append(described)
            if described.get("fps"):
                channel.log(
                    tr(
                        "Camera {index}: {name} ({width}x{height}, {fps} frames a second).",
                        **described,
                    )
                )
            else:
                channel.log(tr("Camera {index}: {name} ({width}x{height}).", **described))
        channel.progress(Phase.SETUP, position, len(candidates), unit="devices")

    if not found:
        channel.log(
            tr(
                "No camera answered. One that another application already has open "
                "cannot be opened again here."
            ),
            level="warning",
        )
    return {"cameras": found}


def _candidates(max_index: int) -> list[tuple[int, str]]:
    """Indices worth trying, with a name where the system publishes one."""
    listed = _v4l_devices()
    if listed:
        return listed
    return [(index, "") for index in range(max_index + 1)]


def _v4l_devices() -> list[tuple[int, str]]:
    """``/dev/videoN`` as Linux describes them, in index order.

    Both the capture node and its metadata sibling appear here; which is which
    is not reliably marked, so both are probed and the one that produces no
    frame drops out.
    """
    try:
        entries = sorted(V4L_ROOT.iterdir())
    except OSError:
        return []

    devices: list[tuple[int, str]] = []
    for entry in entries:
        digits = "".join(ch for ch in entry.name if ch.isdigit())
        if not digits:
            continue
        try:
            name = (entry / "name").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            name = ""
        devices.append((int(digits), name))
    return devices


def _probe(cv2, index: int, name: str) -> dict | None:
    """Open one camera, take one frame, and describe it. ``None`` if it cannot."""
    capture = None
    try:
        capture = open_camera(index)
        if capture is None:
            return None
        # A frame, not just an open handle: see the module docstring. Some
        # drivers need a moment to deliver the first one, so it is worth one
        # retry rather than writing the device off.
        ok, frame = capture.read()
        if not ok or frame is None:
            time.sleep(0.2)
            ok, frame = capture.read()
        if not ok or frame is None:
            return None
        # Measured off the frame rather than off the capture's properties: a
        # driver will report a size it agreed to and then hand over another.
        height, width = (int(frame.shape[0]), int(frame.shape[1]))
        return {
            "index": index,
            "name": _tidy(name) or tr("Camera {index}", index=index),
            "width": width,
            "height": height,
            "fps": _fps_of(capture),
        }
    except Exception:  # noqa: BLE001 - a missing device raises whatever it likes
        return None
    finally:
        if capture is not None:
            try:
                capture.release()
            except Exception:  # noqa: BLE001
                pass


def open_camera(index: int):
    """Open one camera as Step 7 opens it by default, or ``None``.

    Shared with ``commands/infer.py`` on purpose: a list that describes a
    camera by one mode while the run uses another is worse than no list.
    """
    from worker.commands.infer import open_camera as open_for_inference
    from worker.util.errors import CameraUnavailable

    try:
        return open_for_inference(index)
    except CameraUnavailable:
        return None


def _tidy(name: str) -> str:
    """Make a device name fit to put in a list.

    Linux publishes these as a colon-joined pair of the card and the bus name,
    which for most webcams is the same string twice -- "UCB Camera: UCB Camera"
    -- or the same string and a 32-character truncation of it. Neither half
    tells the user anything the other does not, and "Integrated Camera" is a
    better thing to choose from than "Integrated Camera: Integrated C".
    """
    parts = [part.strip() for part in str(name).split(":") if part.strip()]
    kept: list[str] = []
    for part in parts:
        # A part that is merely the start of one already kept, or that one
        # already kept starts with, adds nothing.
        if any(other.startswith(part) or part.startswith(other) for other in kept):
            continue
        kept.append(part)
    return ": ".join(kept)


def _fps_of(capture) -> int:
    """The frame rate the driver reports, or 0 where it does not know one.

    DirectShow reports -1 for a camera that was not asked for a rate, which is
    "unknown" and not something to put in a list as "-1 fps".
    """
    from worker.commands.infer import granted_mode

    return int(granted_mode(capture).get("fps") or 0)


def capture_backend(cv2) -> int:
    """The capture backend to ask for.

    ``CAP_ANY`` on Windows picks Microsoft Media Foundation, which takes
    seconds to fail on an index with nothing behind it -- long enough that
    scanning five of them looks like the app has hung. DirectShow fails
    immediately and reads the same cameras.
    """
    if os.name == "nt":
        return getattr(cv2, "CAP_DSHOW", 0)
    if sys.platform == "darwin":
        return getattr(cv2, "CAP_AVFOUNDATION", 0)
    return getattr(cv2, "CAP_V4L2", 0) if V4L_ROOT.exists() else 0
