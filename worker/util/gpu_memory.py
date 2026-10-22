"""Make running out of GPU memory an error on Windows, as it is on Linux.

On Linux, asking for more memory than the card has raises ``out of memory``
at once. Ultralytics counts on that: in the first epoch it catches the error,
halves the batch and starts the epoch again, until the batch fits.

On Windows the NVIDIA driver does not refuse. Its "sysmem fallback" quietly
pages the overflow into system RAM and carries on, across PCIe, for every
batch. Nothing fails: the run is correct and many times slower.
The card reports full utilisation, since it is busy waiting for memory, while
it draws a fraction of its power and its fans never start. Ultralytics' memory
column shows more than the card has, which is the one visible sign of it.

The driver setting that turns this off ("CUDA - Sysmem Fallback Policy") is
per machine, in the NVIDIA Control Panel, and not something an app should
change. So PyTorch's own allocator is capped instead, at what the card had free
when the run started. Going past that raises the same ``out of memory`` error
Linux would, and Ultralytics' halving takes it from there.
"""

from __future__ import annotations

import os

#: Kept back from the cap for memory that does not go through PyTorch's
#: allocator: cuDNN and cuBLAS handles, and kernels loaded the first time they
#: run. Too little and the overflow spills anyway, which costs the whole run;
#: too much costs at most one halving of the batch.
MARGIN_BYTES = 1 << 30
MARGIN_FRACTION = 0.05


def cap_to_free_memory(device: object) -> int | None:
    """Cap PyTorch's allocator on a Windows CUDA device. The cap in bytes, or ``None``.

    Has to run before training allocates anything: what the card has free at
    that moment is what the cap is made of. ``None`` means nothing was
    capped -- not Windows, not a single CUDA device, or no CUDA at all.
    """
    if os.name != "nt":
        return None
    index = cuda_index(device)
    if index is None:
        return None
    import torch

    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        return None
    # Creates the CUDA context first, so what it reports as free already
    # leaves room for it.
    free, total = torch.cuda.mem_get_info(index)
    cap = free - max(MARGIN_BYTES, int(total * MARGIN_FRACTION))
    if cap <= 0:
        return None
    torch.cuda.set_per_process_memory_fraction(cap / total, index)
    return cap


def cuda_index(device: object) -> int | None:
    """The CUDA device index Ultralytics will train on, if it is exactly one.

    Several devices mean DDP, which trains in child processes this cap does
    not reach; ``cpu`` and ``mps`` have nothing to cap.
    """
    text = str(device).strip().lower()
    if text == "cuda":
        return 0
    text = text.removeprefix("cuda:")
    return int(text) if text.isdigit() else None
