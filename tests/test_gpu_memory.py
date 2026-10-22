"""Capping PyTorch's allocator so that Windows cannot spill VRAM into system RAM.

torch is replaced by a stand-in that records what it was asked to do: the
tests have no GPU, and the arithmetic is the part worth checking.
"""

from __future__ import annotations

import sys
import types

import pytest

from worker.util import gpu_memory

GIB = 1 << 30


@pytest.fixture
def fake_torch(monkeypatch):
    calls = []
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        mem_get_info=lambda index: (int(14.7 * GIB), int(15.9 * GIB)),
        set_per_process_memory_fraction=lambda fraction, index: calls.append((fraction, index)),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(gpu_memory.os, "name", "nt")
    return calls


def test_the_cap_is_what_was_free_less_a_margin(fake_torch):
    cap = gpu_memory.cap_to_free_memory(0)

    assert cap == int(14.7 * GIB) - GIB
    [(fraction, index)] = fake_torch
    assert index == 0
    assert fraction * int(15.9 * GIB) == pytest.approx(cap)


def test_a_large_card_keeps_back_a_proportional_margin(fake_torch, monkeypatch):
    sys.modules["torch"].cuda.mem_get_info = lambda index: (90 * GIB, 96 * GIB)

    cap = gpu_memory.cap_to_free_memory("cuda:0")

    assert cap == 90 * GIB - int(96 * GIB * gpu_memory.MARGIN_FRACTION)


@pytest.mark.parametrize("device", ["cpu", "mps", "0,1", "", "2"])
def test_nothing_is_capped_where_there_is_no_single_cuda_device(fake_torch, device):
    assert gpu_memory.cap_to_free_memory(device) is None
    assert fake_torch == []


def test_linux_is_left_alone(fake_torch, monkeypatch):
    """Linux raises out of memory by itself."""
    monkeypatch.setattr(gpu_memory.os, "name", "posix")

    assert gpu_memory.cap_to_free_memory(0) is None
    assert fake_torch == []


def test_a_card_with_almost_nothing_free_is_not_capped_to_nothing(fake_torch):
    sys.modules["torch"].cuda.mem_get_info = lambda index: (GIB // 2, 8 * GIB)

    assert gpu_memory.cap_to_free_memory(0) is None
    assert fake_torch == []


@pytest.mark.parametrize(
    ("device", "index"),
    [(0, 0), ("0", 0), ("1", 1), ("cuda", 0), ("cuda:1", 1), ("CUDA:0", 0),
     ("cpu", None), ("0,1", None), ("mps", None)],
)
def test_the_device_is_read_the_way_ultralytics_reads_it(device, index):
    assert gpu_memory.cuda_index(device) == index
