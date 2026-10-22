"""A training run with no machine learning in it.

This exists so the load-bearing parts of the app -- framing, throttling, the
cancel ladder, process-tree reaping, ETA maths and the live metrics chart -- can
be built and tested before torch is involved, on any machine, in seconds. The
GUI's ``--fake-worker`` mode routes jobs here.

It deliberately reproduces the hostile behaviours of the real thing:

``noisy``
    writes a tqdm-style progress bar to stdout using bare carriage returns, which
    is what would destroy naive line framing.
``ignore_cancel``
    ignores the cooperative cancel, so the GUI's SIGTERM/kill escalation gets
    exercised.
``children``
    spawns idle grandchildren, standing in for dataloader workers and TensorRT
    builders, so the tree reaper has something to reap.
``ignore_sigterm``
    additionally ignores SIGTERM, forcing escalation all the way to the tree
    kill -- which is what a process wedged inside a native library looks like.
``crash_after_batches``
    exits via ``os._exit`` with no result frame, standing in for a segfault or
    an OOM kill, so the "died without saying why" path gets exercised.
"""

from __future__ import annotations

import os
import random
import signal
import subprocess
import sys
import time

from worker.channel import Channel
from worker.i18n import tr, trn
from worker.protocol import Phase


def run(channel: Channel, job: dict) -> dict:
    epochs = int(job.get("epochs", 10))
    batches = int(job.get("batches_per_epoch", 40))
    epoch_seconds = float(job.get("epoch_seconds", 2.0))
    ignore_cancel = bool(job.get("ignore_cancel", False))
    noisy = bool(job.get("noisy", False))
    n_children = int(job.get("children", 0))
    seed = int(job.get("seed", 0))
    crash_after = int(job.get("crash_after_batches", 0))

    if job.get("ignore_sigterm"):
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except (ValueError, OSError, AttributeError):
            pass

    rng = random.Random(seed)
    children = [_spawn_idle_child() for _ in range(n_children)]

    try:
        channel.progress(Phase.SETUP, 0, 1, unit="steps")
        time.sleep(0.3)
        channel.log(
            trn(
                "Simulating {n} epochs x {batches} batches.",
                "Simulating {n} epochs x {batches} batches.",
                epochs,
                batches=batches,
            )
        )
        channel.ready(epochs=epochs, batches_per_epoch=batches)
        channel.progress(Phase.SETUP, 1, 1, unit="steps")

        batch_delay = epoch_seconds / max(batches, 1)
        started = time.monotonic()
        # Exponential moving average of per-epoch cost. A global mean would be
        # skewed by a slow first epoch (warmup, cudnn autotune, model download),
        # which is exactly when the user is watching the estimate.
        ema_epoch_s: float | None = None
        best_fitness = 0.0

        for epoch in range(1, epochs + 1):
            epoch_start = time.monotonic()
            phase = Phase.WARMUP if epoch == 1 else Phase.TRAIN

            for batch in range(1, batches + 1):
                if not ignore_cancel:
                    channel.check_cancelled()
                time.sleep(batch_delay)
                done = (epoch - 1) * batches + batch
                if crash_after and done >= crash_after:
                    # Deliberately violent: no result frame, no traceback, no
                    # chance for the channel to flush.
                    os._exit(3)
                channel.progress(
                    phase,
                    done,
                    epochs * batches,
                    unit="batches",
                    eta_s=_eta(ema_epoch_s, epochs, epoch, batch, batches),
                )
                if noisy:
                    pct = int(100 * batch / batches)
                    sys.stdout.write(f"\r{epoch}/{epochs} [{'=' * (pct // 5):<20}] {pct}%")
                    sys.stdout.flush()

            if noisy:
                sys.stdout.write("\n")
                sys.stdout.flush()

            channel.progress(Phase.VAL, epoch, epochs, unit="epochs")
            time.sleep(min(0.2, batch_delay * 2))

            # Metrics that improve with diminishing returns plus noise, so the
            # chart looks like a real run and "best epoch" actually moves.
            progress_frac = epoch / epochs
            base = 0.30 + 0.55 * (1 - pow(2.718281828, -3 * progress_frac))
            values = {
                "metrics/mAP50(B)": round(min(0.99, base + rng.uniform(-0.02, 0.02)), 4),
                "metrics/mAP50-95(B)": round(min(0.95, base * 0.68 + rng.uniform(-0.02, 0.02)), 4),
                "metrics/precision(B)": round(min(0.99, base * 1.05 + rng.uniform(-0.03, 0.03)), 4),
                "metrics/recall(B)": round(min(0.99, base * 0.95 + rng.uniform(-0.03, 0.03)), 4),
                "train/box_loss": round(max(0.2, 2.4 * (1 - progress_frac) + rng.uniform(0, 0.1)), 4),
                "train/cls_loss": round(max(0.1, 1.8 * (1 - progress_frac) + rng.uniform(0, 0.1)), 4),
                "lr/pg0": round(0.01 * (1 - 0.9 * progress_frac), 6),
            }
            fitness = values["metrics/mAP50-95(B)"]
            is_best = fitness > best_fitness
            best_fitness = max(best_fitness, fitness)
            channel.metrics(epoch, values, best=is_best)

            if is_best:
                channel.artifact("weights", "runs/fake/weights/best.pt", epoch=epoch)
            channel.artifact("weights", "runs/fake/weights/last.pt", epoch=epoch)

            elapsed = time.monotonic() - epoch_start
            ema_epoch_s = elapsed if ema_epoch_s is None else 0.6 * ema_epoch_s + 0.4 * elapsed

        channel.progress(Phase.FINAL_VAL, 0, 1, unit="steps")
        time.sleep(0.4)
        channel.progress(Phase.FINAL_VAL, 1, 1, unit="steps")

        total = time.monotonic() - started
        channel.log(
            trn(
                "Finished {n} epochs in {seconds:.1f}s.",
                "Finished {n} epochs in {seconds:.1f}s.",
                epochs,
                seconds=total,
            )
        )
        return {
            "epochs": epochs,
            "best_fitness": round(best_fitness, 4),
            "elapsed_s": round(total, 2),
            "weights": "runs/fake/weights/best.pt",
        }
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()


def _eta(
    ema_epoch_s: float | None,
    epochs: int,
    epoch: int,
    batch: int,
    batches: int,
) -> float | None:
    if ema_epoch_s is None:
        return None
    remaining_epochs = epochs - epoch
    remaining_in_epoch = (batches - batch) / batches
    return round(ema_epoch_s * (remaining_epochs + remaining_in_epoch), 1)


def _spawn_idle_child() -> subprocess.Popen:
    """A grandchild that outlives a plain ``terminate()`` on the worker."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time\nwhile True: time.sleep(3600)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
