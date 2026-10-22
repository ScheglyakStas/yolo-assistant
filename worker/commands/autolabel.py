"""Draw the first set of boxes, with OWLv2, from a sentence per class.

The user has thousands of photos in per-class folders and no boxes. OWLv2 is an
open-vocabulary detector: given "a photo of a delivery van" it finds delivery
vans in an image it was never trained on. That is what makes labelling a dataset
in an afternoon possible, and it is the whole of Step 3.

Four things in here are not obvious and are the reason this file is longer than
"run the model, write the files":

**The padding.** OWLv2's processor pads every image to a square before resizing
it, adding grey to the right and bottom. The boxes it returns are normalised
against *that* square, not against the photo. Post-processing them with the
photo's own size -- the obvious thing, and what the equivalent OWL-ViT code does
-- stretches every box, and the error is invisible: boxes still land on objects,
just slightly wrong, in a way that quietly caps the accuracy of everything
trained afterwards. They are scaled by the longer edge instead, then clipped to
the photo.

**The label file is the state.** A photo has been labelled when its label file
exists, so a run resumes by listing the files that are not there yet. Nothing
records progress; cancelling is just stopping, and starting again picks up where
it stopped.

**An empty label file is an answer.** It says "no objects in this photo", which
YOLO understands as a background image. It is also exactly what a wrong prompt
produces for every photo in a class, so the share of empties per class is
reported rather than buried.

**The batch is measured, not guessed.** transformers has no memory-efficient
attention kernel for OWLv2, so every photo in a batch materialises a
heads x patches x patches tensor: four photos through the large model is one
6.4 GB allocation, and eight was 12.8 GB and an immediate crash on a 16 GB
card. How many photos go through at once is therefore worked out from this
card's free memory and this model's measured appetite, and an out-of-memory
error halves it and retries rather than ending a two-hour run.

**Half precision and the fast image processor are asked for explicitly.** The
defaults are fp32 weights with an fp32 attention softmax, and a resize that
runs on one CPU core while the GPU waits. Both cost real money here -- on eight
photos, 12.7 GB reserved against 5.3 GB, and 703 ms of preparation against 93
-- and on real photos the boxes agree to within a pixel.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from worker.channel import Channel
from worker.i18n import tr, trn
from worker.protocol import Phase
from worker.util.errors import DatasetError, GpuUnavailable, ModelUnavailable

#: Kept in step with the GUI's ``app/core/dataset.py``; the worker cannot import
#: it, having only ``worker`` on its path.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})

LABEL_SUFFIX = ".txt"
CLASSES_FILENAME = "classes.txt"
PREPARED_FILENAME = "prepared.json"
AUTOLABEL_FILENAME = "autolabel.json"

#: The most photos per forward pass, whatever the card could hold. Past this,
#: throughput barely moves -- measured on a 5070 Ti, the base model costs 54 ms
#: a photo at eight and 67 ms at four -- while the failure gets more expensive.
#: The batch actually used is measured per run; see ``suggest_batch``.
MAX_BATCH = 8

#: Kept under its old name because a job may still ask for a fixed batch.
DEFAULT_BATCH = MAX_BATCH

#: The share of free VRAM a run will fill. See ``suggest_batch``.
VRAM_HEADROOM = 0.5

#: Expandable segments let the allocator grow one region instead of hoarding a
#: block per tensor shape it has seen. OWLv2 asks for a differently-shaped
#: multi-gigabyte tensor whenever the batch changes, which is the worst case for
#: the default allocator.
ALLOCATOR_CONFIG = "expandable_segments:True"

#: Photos in the synthetic batch used to time image processors against each
#: other. Enough that per-call overhead does not dominate, few enough to be over
#: in about a second.
PROBE_PHOTOS = 4

#: How much faster a more exotic processor has to be to be worth choosing.
FASTER_ENOUGH = 0.8

GB = 1024**3

#: Boxes thinner than this in either direction are dropped. They are always
#: artefacts of a box clipped against the edge of the photo.
MIN_BOX_PIXELS = 2.0

#: A class where this share of photos got no box at all is reported. Almost
#: always a prompt that describes something the photos do not contain.
EMPTY_SHARE_WARNING = 0.5


@dataclass(frozen=True)
class Detection:
    """One box OWLv2 returned, in pixels of the original photo."""

    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1
    query: int = 0


@dataclass(frozen=True)
class Photo:
    """What the detector saw in one photo."""

    width: int
    height: int
    detections: tuple[Detection, ...] = ()


def run(channel: Channel, job: dict) -> dict:
    project = Path(job["project"]).expanduser()
    dataset = project / "dataset"
    images_root = dataset / "images"
    labels_root = dataset / "labels"

    threshold = float(job.get("threshold", 0.2))
    max_detections = int(job.get("max_detections", 10))
    iou = float(job.get("iou", 0.55))
    relabel = bool(job.get("relabel", False))
    model_id = str(job.get("model") or "google/owlv2-base-patch16-ensemble")
    # A job may still fix the batch -- the tests do, and so does anyone
    # debugging a card. Left alone, it is measured once the model is loaded.
    requested_batch = job.get("batch")

    if not (dataset / PREPARED_FILENAME).is_file():
        raise DatasetError(
            tr("{dataset} does not hold a prepared dataset. Run Step 2 again.", dataset=dataset)
        )
    if not 0.0 < threshold < 1.0:
        raise DatasetError(
            tr(
                "The confidence threshold must be between 0 and 1, not {threshold}.",
                threshold=threshold,
            )
        )

    # Class indices come from classes.txt and nowhere else: that order is baked
    # into every label file, and a caller that disagreed about it would write
    # labels naming the wrong class -- a fault no later step can detect.
    order = _class_order(dataset)
    plan = _plan(job.get("classes") or [], order, images_root, labels_root, relabel=relabel)

    todo = sum(len(work) for _, work in plan)
    if todo == 0:
        channel.log(tr("Every photo already has a label; there is nothing to do."))
        return {"labelled": 0, "boxes": 0, "empty": 0, "skipped": True}

    missing_prompts = [entry["name"] for entry, work in plan if work and not entry["prompt"]]
    if missing_prompts:
        raise DatasetError(
            tr(
                "No description was given for {names}. "
                "Each class needs a sentence saying what to look for.",
                names=", ".join(tr('"{name}"', name=name) for name in missing_prompts),
            )
        )

    # Written before the model loads, so that a run cancelled during the
    # download still leaves the user's prompts behind to come back to. It
    # records what was asked for, never what was finished.
    record = {
        "schema": int(job.get("schema", 1)),
        "model": model_id,
        "threshold": threshold,
        "max_detections": max_detections,
        "iou": iou,
        "relabel": relabel,
        "started_at": time.time(),
        "classes": [
            {"index": e["index"], "name": e["name"], "slug": e["slug"], "prompt": e["prompt"]}
            for e, _ in plan
        ],
    }
    _write_json(dataset / AUTOLABEL_FILENAME, record)

    channel.progress(Phase.DOWNLOAD, 0, None, unit="files")
    channel.log(
        tr(
            "Loading {model}. The first run downloads it, which can take a few minutes.",
            model=model_id,
        )
    )
    detector = load_detector(model_id, device=str(job.get("device") or "auto"), channel=channel)
    describe = getattr(detector, "describe", None)
    if callable(describe):
        channel.log(describe())

    if requested_batch:
        batch_size = max(1, int(requested_batch))
    else:
        batch_size, sizing = _measure_batch(
            detector, plan, threshold=threshold, ceiling=MAX_BATCH
        )
        if sizing:
            channel.log(sizing)

    channel.ready(total=todo, classes=len(plan), model=model_id)
    channel.log(
        tr("Labelling {todo:,} photos in {classes} classes.", todo=todo, classes=len(plan))
    )

    state = {"batch": batch_size}
    started = time.monotonic()
    done = 0
    rate: float | None = None
    totals = {"labelled": 0, "boxes": 0, "empty": 0}
    per_class: list[dict] = []

    for entry, work in plan:
        if not work:
            per_class.append({**entry, "labelled": 0, "boxes": 0, "empty": 0})
            continue

        prompts = _prompts(entry["prompt"])
        labelled = boxes_written = empty = 0
        (labels_root / entry["slug"]).mkdir(parents=True, exist_ok=True)

        index = 0
        while index < len(work):
            channel.check_cancelled()
            batch = work[index : index + state["batch"]]
            index += len(batch)

            batch_started = time.monotonic()
            photos = _detect(
                detector,
                [image for image, _label in batch],
                prompts,
                threshold=threshold,
                channel=channel,
                state=state,
            )
            # Timed per batch and shared out: within a batch every photo goes
            # through the same forward pass, so a per-photo time would be a
            # fiction. An EMA over batches, because the first one pays for
            # cudnn autotuning and a cold page cache, and a plain mean would
            # still be apologising for it halfway through the run.
            per_photo = (time.monotonic() - batch_started) / max(1, len(batch))
            rate = per_photo if rate is None else 0.8 * rate + 0.2 * per_photo

            for (_image, label), photo in zip(batch, photos):
                # Also between photos, not only between batches: the runner
                # gives a cancelled job five seconds to stop before it starts
                # signalling, and a batch can outlast that.
                channel.check_cancelled()
                rows = [
                    _to_yolo(det.box, photo.width, photo.height)
                    for det in _select(
                        photo.detections,
                        threshold=threshold,
                        max_detections=max_detections,
                        iou=iou,
                    )
                ]
                rows = [row for row in rows if row is not None]
                _write_label(label, entry["index"], rows)
                labelled += 1
                boxes_written += len(rows)
                if not rows:
                    empty += 1

                done += 1
                channel.progress(
                    Phase.LABEL,
                    done,
                    todo,
                    unit="photos",
                    eta_s=round(rate * (todo - done), 1),
                )

        totals["labelled"] += labelled
        totals["boxes"] += boxes_written
        totals["empty"] += empty
        per_class.append({**entry, "labelled": labelled, "boxes": boxes_written, "empty": empty})
        channel.log(
            tr(
                "{name}: {labelled:,} photos labelled, {boxes:,} boxes, "
                "{empty:,} with nothing found.",
                name=entry["name"],
                labelled=labelled,
                boxes=boxes_written,
                empty=empty,
            )
        )

    release = getattr(detector, "release", None)
    if callable(release):
        release()

    record["finished_at"] = time.time()
    record["classes"] = [
        {
            "index": c["index"],
            "name": c["name"],
            "slug": c["slug"],
            "prompt": c["prompt"],
            "labelled": c["labelled"],
            "boxes": c["boxes"],
            "empty": c["empty"],
        }
        for c in per_class
    ]
    _write_json(dataset / AUTOLABEL_FILENAME, record)

    warnings = _warnings(per_class, totals)
    for line in warnings:
        channel.warning("autolabel", line)
    channel.artifact("labels", str(labels_root), classes=len(per_class))
    channel.log(
        tr(
            "Labelled {labelled:,} photos with {boxes:,} boxes in {seconds:.1f}s.",
            labelled=totals["labelled"],
            boxes=totals["boxes"],
            seconds=time.monotonic() - started,
        )
    )

    return {
        "project": str(project),
        "labels": str(labels_root),
        **totals,
        "classes": [
            {k: c[k] for k in ("name", "slug", "prompt", "labelled", "boxes", "empty")}
            for c in per_class
        ],
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# What to run over
# --------------------------------------------------------------------------- #


def _class_order(dataset: Path) -> dict[str, int]:
    try:
        text = (dataset / CLASSES_FILENAME).read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(
            tr("{path} cannot be read: {error}", path=dataset / CLASSES_FILENAME, error=exc)
        ) from exc
    names = [line.strip() for line in text.splitlines() if line.strip()]
    if not names:
        raise DatasetError(
            tr("{path} is empty. Run Step 2 again.", path=dataset / CLASSES_FILENAME)
        )
    return {name: index for index, name in enumerate(names)}


def _plan(
    classes: list[dict],
    order: dict[str, int],
    images_root: Path,
    labels_root: Path,
    *,
    relabel: bool,
) -> list[tuple[dict, list[tuple[Path, Path]]]]:
    """Per class, the (photo, label) pairs this run has to produce."""
    plan: list[tuple[dict, list[tuple[Path, Path]]]] = []
    for raw in classes:
        name = str(raw.get("name") or "")
        slug = str(raw.get("slug") or name)
        if name not in order:
            raise DatasetError(
                tr(
                    'The dataset has no class named "{name}". It was prepared with a '
                    "different set of classes -- go back to Step 2 and prepare it again.",
                    name=name,
                )
            )
        entry = {
            "index": order[name],
            "name": name,
            "slug": slug,
            "prompt": str(raw.get("prompt") or "").strip(),
        }
        directory = images_root / slug
        try:
            images = [
                p
                for p in directory.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            ]
        except OSError:
            # A class folder the user deleted between steps. Report it as a
            # class with no work rather than failing the whole run.
            plan.append((entry, []))
            continue
        images.sort(key=lambda p: p.name.casefold())

        work = []
        for image in images:
            label = labels_root / slug / (image.stem + LABEL_SUFFIX)
            if relabel or not label.exists():
                work.append((image, label))
        plan.append((entry, work))
    return plan


def _prompts(prompt: str) -> list[str]:
    """One class's prompt, as the list of queries OWLv2 is given.

    Semicolons separate alternatives -- "a lorry; a truck" -- because the same
    object genuinely has different names, and a detector given both finds more
    of them. Commas are left alone: they appear inside ordinary noun phrases.
    """
    parts = [part.strip() for part in str(prompt).split(";")]
    return [part for part in parts if part] or [str(prompt).strip()]


# --------------------------------------------------------------------------- #
# Boxes
# --------------------------------------------------------------------------- #


def _select(
    detections,
    *,
    threshold: float,
    max_detections: int,
    iou: float,
) -> list[Detection]:
    """Thresholded, de-duplicated, best first.

    The de-duplication is across prompts as well as within one: "a lorry" and
    "a truck" find the same vehicle twice, and two boxes on one object teaches
    the model that objects come in pairs.
    """
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda d: -float(d.score)):
        if float(candidate.score) < threshold:
            break
        if any(_iou(candidate.box, chosen.box) > iou for chosen in kept):
            continue
        kept.append(candidate)
        if len(kept) >= max_detections:
            break
    return kept


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = _ordered(a)
    bx0, by0, bx1, by1 = _ordered(b)
    inter_w = min(ax1, bx1) - max(ax0, bx0)
    inter_h = min(ay1, by1) - max(ay0, by0)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return intersection / union if union > 0 else 0.0


def _ordered(box) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in box)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _to_yolo(box, width: int, height: int) -> tuple[float, float, float, float] | None:
    """One box in YOLO's format: centre and size, as fractions of the photo.

    Clipping happens here rather than at the model, because OWLv2 is allowed to
    predict a box that runs off the edge of the image and frequently does -- a
    half-visible object is still an object.
    """
    if width <= 0 or height <= 0:
        return None
    x0, y0, x1, y1 = _ordered(box)
    x0 = min(max(x0, 0.0), float(width))
    x1 = min(max(x1, 0.0), float(width))
    y0 = min(max(y0, 0.0), float(height))
    y1 = min(max(y1, 0.0), float(height))
    box_w = x1 - x0
    box_h = y1 - y0
    if box_w < MIN_BOX_PIXELS or box_h < MIN_BOX_PIXELS:
        return None
    return (
        (x0 + box_w / 2) / width,
        (y0 + box_h / 2) / height,
        box_w / width,
        box_h / height,
    )


def _write_label(path: Path, index: int, rows: list) -> None:
    """Write one label file, atomically.

    Zero rows is a real answer -- YOLO reads an empty label file as "this photo
    contains no objects" -- so the file is written either way. It is also what
    makes the file's existence mean "this photo has been looked at".
    """
    text = "".join(
        f"{index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


def _detect(detector, paths, prompts, *, threshold, channel, state) -> list[Photo]:
    """One batch, halving and retrying if the card runs out of memory.

    VRAM is shared with everything else on the machine, so the batch size that
    worked at the start of a run can fail forty minutes in when a browser opens
    a video. Ending a long run over that would be indefensible.
    """
    # What a failure taught applies to the rest of the run, not just to the half
    # that failed. Without this the second half of a halved batch is the same
    # size as the first, so it runs out of memory too and the user reads the
    # same warning twice for one lesson.
    limit = max(1, int(state["batch"]))
    if len(paths) > limit:
        photos: list[Photo] = []
        for start in range(0, len(paths), limit):
            photos += _detect(
                detector,
                paths[start : start + limit],
                prompts,
                threshold=threshold,
                channel=channel,
                state=state,
            )
        return photos

    try:
        return list(detector.detect(list(paths), list(prompts), threshold=threshold))
    except Exception as exc:  # noqa: BLE001 - torch raises several OOM types
        if not _is_oom(exc) or len(paths) <= 1:
            raise
        half = max(1, len(paths) // 2)

    # The retry happens out here, after the ``except`` block has closed, and
    # that is the whole point of the empty-looking block above. Inside it, the
    # exception holds its traceback, the traceback holds every frame beneath it,
    # and those frames hold the activations that did not fit -- gigabytes of
    # them for OWLv2. Retrying in there retries with the failed attempt still
    # resident, so the halved batch fails too, and the one after that, until the
    # halving reaches a single photo and gives up. That is what turned "it ran
    # out of memory once" into "it crashed immediately".
    state["batch"] = half
    channel.log(
        tr(
            "Ran out of GPU memory on {count} photos at once; continuing {half} at a time.",
            count=len(paths),
            half=half,
        ),
        level="warning",
    )
    _free(detector)
    first = _detect(
        detector, paths[:half], prompts, threshold=threshold, channel=channel, state=state
    )
    second = _detect(
        detector, paths[half:], prompts, threshold=threshold, channel=channel, state=state
    )
    return first + second


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in f"{type(exc).__name__}: {exc}".lower()


def _free(detector) -> None:
    empty = getattr(detector, "empty_cache", None)
    if callable(empty):
        empty()


def _measure_batch(detector, plan, *, threshold: float, ceiling: int) -> tuple[int, str]:
    """How many photos to send at once, asked of the detector that will run them."""
    suggest = getattr(detector, "suggest_batch", None)
    if not callable(suggest):
        return ceiling, ""
    for entry, work in plan:
        if work:
            return suggest(
                work[0][0], _prompts(entry["prompt"]), threshold=threshold, ceiling=ceiling
            )
    return ceiling, ""


def load_detector(model_id: str, *, device: str = "auto", channel: Channel | None = None):
    """Load OWLv2. Replaced wholesale in tests, which have no GPU and no torch."""
    # Read once, at the first CUDA allocation, so it has to be set before torch
    # reaches the card. OWLv2's attention tensor is by far the largest single
    # allocation in this application and its shape changes with the batch --
    # exactly the pattern that fragments the default allocator, which keeps one
    # block per shape it has seen. Measured on a 16 GB card: four photos through
    # the large model reserved 10.9 GB for 7.5 GB of live tensors, and 7.6 GB
    # with this. ``setdefault`` so that an operator debugging something can still
    # override it from outside.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ALLOCATOR_CONFIG)

    # A missing torch is deliberately left to propagate: the entry point turns a
    # ModuleNotFoundError into "the managed environment is missing 'torch'",
    # which is both true and the thing to act on.
    import torch

    resolved = _resolve_device(device, torch)
    try:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
    except ImportError as exc:
        raise ModelUnavailable(
            tr("The managed environment cannot load OWLv2: {error}", error=exc)
        ) from exc

    # Only reached on the fp32 path, which is the CPU one today -- but it costs
    # nothing and stops a future fp32 fallback from running at a third speed.
    torch.set_float32_matmul_precision("high")

    dtype = torch.float16 if resolved.startswith("cuda") else torch.float32
    try:
        processor, call_kwargs, backend = _load_processor(Owlv2Processor, model_id, resolved)
        model = _load_model(Owlv2ForObjectDetection, model_id, dtype, torch)
    except OSError as exc:
        # Almost always no internet on the first run, or a proxy in the way.
        raise ModelUnavailable(
            tr("{model} could not be downloaded: {error}", model=model_id, error=exc)
        ) from exc

    model = model.to(resolved).eval()
    if channel is not None:
        channel.log(backend)
    return _Owlv2Detector(
        model, processor, resolved, torch, dtype=dtype, call_kwargs=call_kwargs
    )


def _load_processor(cls, model_id: str, device: str):
    """The fastest image processor this transformers will give us, chosen by
    measuring the candidates rather than by asking which is which.

    The slow processor resizes and normalises a 960x960 float buffer per photo
    on one CPU core while the GPU waits; the torchvision-backed one does it on
    the card. Measured on eight photos: 703 ms against 93 ms, which on a
    thousand-photo class is eleven minutes against one and a half.

    Asking for the fast one is version-dependent and, worse, silently so.
    transformers 4.5x selects a separate ``Owlv2ImageProcessorFast`` through
    ``use_fast=True``; 5.x drops that class and renames the switch to
    ``backend="torchvision"``; and ``from_pretrained`` files an unrecognised
    keyword into the config instead of raising, so the wrong name gets the slow
    processor and no error. Timing cannot be wrong about the thing that actually
    matters, and it costs about a second, once per run.

    ``device=`` on the CALL is a separate axis, and the one that moves the work
    off the CPU: the fast backend resizes on whatever device it is handed. The
    slow one does not accept the argument at all, so that is measured too rather
    than assumed.
    """
    candidates = []
    for load_kwargs in ({}, {"use_fast": True}, {"backend": "torchvision"}):
        try:
            processor = cls.from_pretrained(model_id, **load_kwargs)
        except OSError:
            # No network, or a proxy. Worth reporting only for the plain load:
            # the variants are asking the same cache the same question.
            if not load_kwargs:
                raise
            continue
        except Exception:  # noqa: BLE001 - an unsupported switch raises anything
            continue
        for call_kwargs in ({"device": device}, {}):
            try:
                cost = _time_processor(processor, call_kwargs)
            except Exception:  # noqa: BLE001 - the slow one rejects device=
                continue
            candidates.append((cost, load_kwargs, call_kwargs, processor))

    if not candidates:
        raise ModelUnavailable(
            tr("No working image processor could be loaded for {model}.", model=model_id)
        )

    # A more exotic candidate has to be clearly faster to win, not merely
    # luckier on the clock: where all three loads give the same processor, the
    # plain one is the one that will not be deprecated out from under us.
    best = candidates[0]
    for candidate in candidates[1:]:
        if candidate[0] < best[0] * FASTER_ENOUGH:
            best = candidate

    cost, load_kwargs, call_kwargs, processor = best
    slowest = max(c for c, _, _, _ in candidates)
    asked = ", ".join(f"{k}={v!r}" for k, v in {**load_kwargs, **call_kwargs}.items())
    name = type(getattr(processor, "image_processor", processor)).__name__
    values = {"name": name, "options": asked or tr("defaults"), "cost": cost}
    if slowest > cost * 1.5:
        described = trn(
            "Preparing photos with {name} ({options}): {cost:.0f} ms per {n} photos, "
            "against {slowest:.0f} ms the slowest way of asking.",
            "Preparing photos with {name} ({options}): {cost:.0f} ms per {n} photos, "
            "against {slowest:.0f} ms the slowest way of asking.",
            PROBE_PHOTOS,
            slowest=slowest,
            **values,
        )
    else:
        described = trn(
            "Preparing photos with {name} ({options}): {cost:.0f} ms per {n} photos. "
            "This transformers offers no faster path for OWLv2.",
            "Preparing photos with {name} ({options}): {cost:.0f} ms per {n} photos. "
            "This transformers offers no faster path for OWLv2.",
            PROBE_PHOTOS,
            **values,
        )
    return processor, call_kwargs, described


def _time_processor(processor, call_kwargs: dict, *, repeats: int = 2) -> float:
    """Milliseconds for one batch, on blank photos at a realistic size.

    The last of ``repeats`` is the one reported: the first pays for lazy imports
    and for whatever the backend warms up, and that cost is paid once per run
    rather than once per batch.
    """
    from PIL import Image

    images = [Image.new("RGB", (1280, 720), (40, 40, 40)) for _ in range(PROBE_PHOTOS)]
    cost = 0.0
    for _ in range(repeats):
        started = time.perf_counter()
        processor(
            text=[["a photo of a cat"]] * PROBE_PHOTOS,
            images=images,
            return_tensors="pt",
            padding=True,
            **call_kwargs,
        )
        cost = time.perf_counter() - started
    return cost * 1000.0


def _load_model(cls, model_id: str, dtype, torch):
    """The model with its weights actually at ``dtype``.

    Half precision by loading rather than by ``torch.autocast``, which was the
    first version of this and is a trap. autocast leaves the fp32 weights on the
    card and casts copies per operation, and -- the part that actually hurts --
    it runs the attention softmax in fp32. transformers has no memory-efficient
    attention kernel for OWLv2, so that softmax is a real tensor of
    batch x heads x patches x patches: eight photos through the large model ask
    the allocator for 12.8 GB in one piece and the run ends before the first
    photo is written. The same tensor is 6.4 GB in fp16, and the weights are
    half the size besides.

    Measured against the fp32 path on the user's own photos, the boxes agree to
    within a pixel (IoU 0.99 and better) and the scores move by about 0.005 --
    which matters only for a photo sitting exactly on the confidence threshold,
    where it can fall on the other side of it. OWLv2's scores are uncalibrated
    to begin with, so that is noise either way.

    ``torch_dtype`` is the old name for ``dtype``. Both are accepted by recent
    versions, and an older one files the unknown name into the config and hands
    back fp32 weights without complaining -- so what came out is checked rather
    than assumed. Getting this wrong is invisible: nothing fails, the run simply
    uses twice the memory it should.
    """
    model = None
    for key in ("dtype", "torch_dtype"):
        try:
            model = cls.from_pretrained(model_id, **{key: dtype})
        except (TypeError, ValueError):
            continue
        break
    if model is None:
        model = cls.from_pretrained(model_id)
    if next(model.parameters()).dtype != dtype:
        model = model.to(dtype)
    return model


def _resolve_device(device: str, torch) -> str:
    if device in ("", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise GpuUnavailable(
            tr(
                "PyTorch cannot see the GPU, so autolabeling would take days instead of "
                "minutes."
            )
        )
    return device


class _Owlv2Detector:
    """OWLv2 behind the two-method interface :func:`run` uses."""

    def __init__(self, model, processor, device: str, torch, *, dtype=None, call_kwargs=None) -> None:
        self._model = model
        self._processor = processor
        self._device = device
        self._torch = torch
        self._dtype = dtype if dtype is not None else torch.float32
        self._call_kwargs = dict(call_kwargs or {})

    def describe(self) -> str:
        precision = str(self._dtype).replace("torch.", "")
        if self._device.startswith("cuda"):
            name = self._torch.cuda.get_device_name(0)
            free, total = self._torch.cuda.mem_get_info()
            return tr(
                "Running on {name}: {free:.1f} GB of {total:.1f} GB free, "
                "weights in {precision}.",
                name=name,
                free=free / GB,
                total=total / GB,
                precision=precision,
            )
        return tr("Running on the CPU in {precision}. This will be slow.", precision=precision)

    def suggest_batch(self, path, prompts, *, threshold: float, ceiling: int) -> tuple[int, str]:
        """How many photos fit on this card at once, measured on a real one.

        The alternative is what this used to do -- send eight and hope -- which
        is wrong in both directions: the large model needs 6.4 GB in a single
        allocation for four photos, while a 24 GB card labelling with the base
        model has room to spare and was being told to run at a quarter speed.

        One photo goes through first, and the rest is arithmetic. Peak memory
        for a batch of one is the weights plus one photo's activations, the
        weights are already on the card and therefore already counted, and
        activations scale with the batch because the attention tensor does.

        Half of what is free rather than all of it: the model is linear and the
        truth is a little worse than linear, the desktop this is running on
        wants its share, and an underestimate costs a few percent of throughput
        where an overestimate costs the out-of-memory error the whole thing
        exists to avoid. That photo is labelled again with its batch -- one
        duplicated forward pass, against guessing for the rest of the run.
        """
        torch = self._torch
        if not self._device.startswith("cuda"):
            return 1, tr("Running on the CPU, so one photo at a time.")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        weights = torch.cuda.memory_allocated()
        self.detect([path], list(prompts), threshold=threshold)
        peak = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()
        free, _total = torch.cuda.mem_get_info()

        per_photo = max(peak - weights, 1)
        batch = max(1, min(int(ceiling), int((free * VRAM_HEADROOM) // per_photo)))
        return batch, tr(
            "{per_photo:.1f} GB of GPU memory per photo, {free:.1f} GB free: "
            "labelling {batch} at a time.",
            per_photo=per_photo / GB,
            free=free / GB,
            batch=batch,
        )

    def detect(self, paths, prompts, *, threshold: float) -> list[Photo]:
        from PIL import Image

        torch = self._torch
        images = []
        sizes = []
        for path in paths:
            with Image.open(path) as opened:
                # Step 2 already baked in any EXIF rotation, so what is on disk
                # is what the trainer will see and what these boxes describe.
                image = opened.convert("RGB")
            images.append(image)
            sizes.append(image.size)  # (width, height)

        inputs = self._processor(
            text=[list(prompts)] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
            **self._call_kwargs,
        )
        # Only the floats. ``input_ids`` is an index tensor, and casting it to
        # fp16 corrupts the prompt silently -- the tokens come back as whatever
        # number is nearest in half precision.
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        if self._dtype != torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        # Post-processing does arithmetic that fp16 rounds badly at these
        # magnitudes, and its output is pixel coordinates. Casting here does not
        # recover what the network already rounded; it stops the decode adding
        # more.
        for field in ("logits", "pred_boxes"):
            value = getattr(outputs, field, None)
            if value is not None:
                setattr(outputs, field, value.float())

        # The square the boxes are actually normalised against: the processor
        # pads to the longer edge before it resizes. See the module docstring.
        target_sizes = torch.tensor(
            [[float(max(w, h)), float(max(w, h))] for w, h in sizes], device=outputs.logits.device
        )
        results = self._post_process(outputs, threshold, target_sizes)

        photos: list[Photo] = []
        for (width, height), result in zip(sizes, results):
            detections = tuple(
                Detection(
                    score=float(score),
                    box=tuple(float(v) for v in box),
                    query=int(label),
                )
                for score, label, box in zip(
                    result["scores"].tolist(),
                    result["labels"].tolist(),
                    result["boxes"].tolist(),
                )
            )
            photos.append(Photo(width=width, height=height, detections=detections))
        return photos

    def empty_cache(self) -> None:
        if self._device.startswith("cuda"):
            self._torch.cuda.empty_cache()

    def release(self) -> None:
        self._model = None
        self.empty_cache()

    def _post_process(self, outputs, threshold: float, target_sizes):
        """Whichever of the two post-processors this transformers release has.

        ``post_process_object_detection`` is deprecated in favour of the
        ``grounded`` name, and the supported version range spans the rename.
        """
        method = getattr(self._processor, "post_process_grounded_object_detection", None)
        if method is None:
            method = self._processor.post_process_object_detection
        try:
            return method(outputs=outputs, threshold=threshold, target_sizes=target_sizes)
        except TypeError:
            return method(outputs, threshold, target_sizes)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _warnings(per_class: list[dict], totals: dict) -> list[str]:
    lines: list[str] = []
    for entry in per_class:
        if not entry["labelled"]:
            continue
        share = entry["empty"] / entry["labelled"]
        if share >= EMPTY_SHARE_WARNING:
            lines.append(
                trn(
                    '"{name}": nothing was found in {empty:,} of {n:,} photos. '
                    "Those are now labelled as containing no objects. If that is wrong, "
                    "try describing the object differently or lowering the confidence "
                    "threshold.",
                    '"{name}": nothing was found in {empty:,} of {n:,} photos. '
                    "Those are now labelled as containing no objects. If that is wrong, "
                    "try describing the object differently or lowering the confidence "
                    "threshold.",
                    entry["labelled"],
                    name=entry["name"],
                    empty=entry["empty"],
                )
            )
    if totals["labelled"] and not totals["boxes"]:
        lines.append(
            tr(
                "Not a single box was found. The descriptions probably do not match what "
                "is in the photos, or the threshold is too high."
            )
        )
    return lines


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
