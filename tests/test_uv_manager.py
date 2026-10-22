"""uv is downloaded and then executed, so the checksum is not optional."""

import hashlib
import io
import os
import tarfile
from pathlib import Path

import pytest

from yolo_assistant.app.core import paths
from yolo_assistant.app.core.bootstrap import uv_manager


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path))
    return tmp_path


def make_archive(contents: bytes = b"#!/bin/sh\necho uv\n") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        info = tarfile.TarInfo("uv-x86_64-unknown-linux-gnu/uv")
        info.size = len(contents)
        info.mode = 0o755
        bundle.addfile(info, io.BytesIO(contents))
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, body: bytes):
        self._stream = io.BytesIO(body)
        self.headers = {"Content-Length": str(len(body))}

    def read(self, size=-1):
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def install_fake(monkeypatch, body: bytes, digest: str | None = None):
    asset = uv_manager.UvAsset(
        "uv-x86_64-unknown-linux-gnu.tar.gz",
        digest if digest is not None else hashlib.sha256(body).hexdigest(),
    )
    monkeypatch.setattr(uv_manager, "asset_for_this_machine", lambda: asset)
    monkeypatch.setattr(uv_manager.urllib.request, "urlopen", lambda *_a, **_k: FakeResponse(body))


def test_a_verified_download_is_installed_and_executable(isolated, monkeypatch):
    body = make_archive()
    install_fake(monkeypatch, body)
    path = uv_manager.download_uv()
    assert path == uv_manager.managed_uv()
    assert path.exists()
    assert path.stat().st_mode & 0o111, "uv must be executable"


def test_a_tampered_download_is_rejected_and_nothing_is_installed(isolated, monkeypatch):
    install_fake(monkeypatch, make_archive(), digest="0" * 64)
    with pytest.raises(uv_manager.UvUnavailable, match="checksum"):
        uv_manager.download_uv()
    assert not uv_manager.managed_uv().exists()


def test_progress_is_reported_while_downloading(isolated, monkeypatch):
    # Incompressible, or gzip turns 300 kB into a single chunk.
    install_fake(monkeypatch, make_archive(os.urandom(300_000)))
    seen = []
    uv_manager.download_uv(progress=lambda done, total: seen.append((done, total)), chunk_size=1024)
    assert len(seen) > 1
    assert seen[-1][0] == seen[-1][1] > 0


def test_a_download_can_be_abandoned_mid_stream(isolated, monkeypatch):
    """Closing the window during this download must not leave a thread writing
    into a destroyed object -- urllib cannot be interrupted from outside, so
    the loop has to check for itself."""
    install_fake(monkeypatch, make_archive(os.urandom(300_000)))
    calls = []

    def should_stop():
        calls.append(1)
        return len(calls) > 2

    with pytest.raises(uv_manager.UvCancelled):
        uv_manager.download_uv(should_stop=should_stop, chunk_size=1024)
    assert not uv_manager.managed_uv().exists(), "a cancelled download must install nothing"


def test_an_archive_without_uv_is_an_error(isolated, monkeypatch):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        info = tarfile.TarInfo("readme.txt")
        info.size = 2
        bundle.addfile(info, io.BytesIO(b"hi"))
    install_fake(monkeypatch, buffer.getvalue())
    with pytest.raises(uv_manager.UvUnavailable, match="did not contain"):
        uv_manager.download_uv()


def test_every_pinned_platform_has_a_checksum():
    for key, asset in uv_manager.UV_ASSETS.items():
        assert len(asset.sha256) == 64, key
        assert uv_manager.UV_VERSION in asset.url


def test_this_machine_is_covered():
    assert uv_manager.asset_for_this_machine().name


def test_torch_comes_from_the_backend_index_and_nothing_else_does():
    """``--torch-backend`` routes only the PyTorch packages to the CUDA index.

    The index flags it replaced put PyPI *ahead* of the CUDA index -- in uv an
    extra index outranks ``--index-url`` -- which on Windows installed the
    CPU-only torch into every CUDA environment.
    """
    argv = uv_manager.pip_install_argv(
        Path("/uv"),
        Path("/venv/bin/python"),
        requirements=Path("/r.txt"),
        torch_backend="cu128",
    )
    assert argv[argv.index("--torch-backend") + 1] == "cu128"
    assert "--index-url" not in argv and "--extra-index-url" not in argv


def test_no_torch_backend_is_passed_for_the_cpu_backend():
    argv = uv_manager.pip_install_argv(Path("/uv"), Path("/venv/bin/python"), packages=["torch"])
    assert "--torch-backend" not in argv and "--index-url" not in argv


def test_uv_keeps_its_cache_and_pythons_inside_our_data_directory(isolated):
    env = uv_manager.uv_env(base={})
    assert str(isolated) in env["UV_CACHE_DIR"]
    assert str(isolated) in env["UV_PYTHON_INSTALL_DIR"]
