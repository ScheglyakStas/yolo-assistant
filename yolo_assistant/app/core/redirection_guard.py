"""Notice when Windows has started the app unable to run its own Python.

RedirectionGuard is a Windows mitigation that stops a process -- and every
process it starts -- from following a junction an ordinary user created. uv
reaches its managed Python through exactly such a junction
(``pythons\\cpython-3.12-windows-x86_64-none``), so under it no worker starts:
each one dies with "uv trampoline failed to spawn Python child process".

Nothing in the app turns it on. It is inherited: recent Inno Setup releases
enable it in Setup, so the copy started from the installer's last page had it
until the installer was built with ``RedirectionGuard=no``. An installer built
before that, or anything else that launches the app with the policy on, still
passes it down, and a process cannot switch it off for itself. Saying so at
startup turns an error that names no cause into one sentence and a fix.

Kept free of Qt so it can be tested directly.
"""

from __future__ import annotations

import ctypes
import os

#: ``PROCESS_MITIGATION_POLICY.ProcessRedirectionTrustPolicy``.
REDIRECTION_TRUST_POLICY = 16
#: ``EnforceRedirectionTrust``, the low bit of the policy's flags. The next bit
#: is audit-only and blocks nothing.
ENFORCE = 0x1


def is_enforced() -> bool:
    """True if this process cannot follow junctions an ordinary user made.

    False wherever the question cannot be asked: off Windows, and on a
    Windows too old to know the policy, where it cannot be on either.
    """
    if os.name != "nt":
        return False
    try:
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        query = kernel32.GetProcessMitigationPolicy
        query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        query.restype = wintypes.BOOL
        flags = wintypes.DWORD(0)
        if not query(kernel32.GetCurrentProcess(), REDIRECTION_TRUST_POLICY,
                     ctypes.byref(flags), ctypes.sizeof(flags)):
            return False
    except (AttributeError, OSError):
        return False
    return bool(flags.value & ENFORCE)
