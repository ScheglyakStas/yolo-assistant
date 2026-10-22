"""Worker commands.

Each module exposes ``run(channel, job) -> dict``. Imports of heavy libraries
belong *inside* ``run``, never at module scope, so that dispatching a command
which needs nothing (``probe``) does not pay for torch.
"""
