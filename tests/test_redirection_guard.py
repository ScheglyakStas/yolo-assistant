"""Noticing that Windows started the app under RedirectionGuard.

The real check runs in a child process that switches the policy on for itself:
it cannot be switched off again, so it must never be the test runner.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from yolo_assistant.app import main as app_main
from yolo_assistant.app.core import redirection_guard

ROOT = Path(__file__).resolve().parents[1]

ENABLE_THEN_CHECK = """
import ctypes
from ctypes import wintypes
from yolo_assistant.app.core import redirection_guard

print(redirection_guard.is_enforced())
flags = wintypes.DWORD(redirection_guard.ENFORCE)
kernel32 = ctypes.WinDLL("kernel32")
kernel32.SetProcessMitigationPolicy.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
if not kernel32.SetProcessMitigationPolicy(
    redirection_guard.REDIRECTION_TRUST_POLICY, ctypes.byref(flags), ctypes.sizeof(flags)
):
    print("unsupported")
else:
    print(redirection_guard.is_enforced())
"""


def test_off_windows_there_is_nothing_to_enforce(monkeypatch):
    monkeypatch.setattr(redirection_guard.os, "name", "posix")

    assert redirection_guard.is_enforced() is False


@pytest.mark.skipif(os.name != "nt", reason="a Windows process mitigation")
def test_the_policy_is_seen_once_it_is_on():
    result = subprocess.run(
        [sys.executable, "-c", ENABLE_THEN_CHECK],
        capture_output=True, text=True, cwd=ROOT, timeout=60,
    )

    before, after = result.stdout.split()
    assert before == "False"
    if after == "unsupported":
        pytest.skip("this Windows does not have RedirectionGuard")
    assert after == "True"


def test_the_user_is_told_only_when_the_policy_is_on(monkeypatch):
    shown = []
    monkeypatch.setattr(app_main.QMessageBox, "warning", lambda *args: shown.append(args))

    monkeypatch.setattr(redirection_guard, "is_enforced", lambda: False)
    app_main.warn_if_python_is_unreachable(None)
    assert shown == []

    monkeypatch.setattr(redirection_guard, "is_enforced", lambda: True)
    app_main.warn_if_python_is_unreachable(None)
    [(_parent, title, text)] = shown
    assert "Start menu" in text
