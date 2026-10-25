"""What Step 3 is about to do, worked out before it is asked to do it.

Autolabeling is the first step that costs real time -- minutes on a small
dataset, hours on a large one -- and the first where a mistake is invisible.
A prompt that finds nothing produces a dataset full of empty label files, every
one of which is a photo the model will be told contains no objects. That does
not fail; it trains, and it reports a low accuracy that says nothing about why.

So this module exists to answer, cheaply and before anything starts: which
photos still need a label, which classes have a prompt, and what would happen
if the user pressed the button. The page renders the answer and the worker is
handed the same numbers.

Standard library only, and no Qt. The GUI process has no torch and never loads
OWLv2 -- that happens in the worker, inside the managed environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .i18n import N_, tr, trn
from .project import DatasetStatus, Project

AUTOLABEL_FILENAME = "autolabel.json"

#: Bump when the record's shape changes incompatibly. It is only ever read to
#: restore what the user typed, so an unreadable one costs a retyped prompt.
AUTOLABEL_SCHEMA = 1

#: OWLv2's scores are not calibrated the way a classifier's are: a correct box
#: often scores 0.2. Anything near the 0.5 a beginner would reach for finds
#: almost nothing, which is why this default is low and the tooltip says so.
DEFAULT_THRESHOLD = 0.20

#: Boxes kept per photo, best first. Most datasets photograph one object.
DEFAULT_MAX_DETECTIONS = 10

#: Two boxes overlapping by more than this are treated as the same object.
DEFAULT_IOU = 0.55


@dataclass(frozen=True)
class Model:
    """One OWLv2 checkpoint the user can choose between."""

    key: str
    """The Hugging Face id the worker loads."""

    label: str
    download_gb: float
    note: str

    @property
    def caption(self) -> str:
        return f"{self.label}"


#: Deliberately two, not the whole family. The ensemble checkpoints are the
#: ones Google recommends for open-vocabulary use, and offering eight variants
#: to someone labelling their first dataset is not a kindness.
MODELS: tuple[Model, ...] = (
    Model(
        key="google/owlv2-base-patch16-ensemble",
        label="OWLv2 base",
        download_gb=1.3,
        note=N_("quick, and fits a small card. Start here."),
    ),
    Model(
        key="google/owlv2-large-patch14-ensemble",
        label="OWLv2 large",
        download_gb=6.5,
        note=N_("three times slower and wants more GPU memory, but finds harder objects."),
    ),
)

DEFAULT_MODEL = MODELS[0].key


@dataclass(frozen=True)
class ClassPlan:
    """One class: what it holds, and what the prompt would be run against."""

    index: int
    name: str
    slug: str
    images: int
    labels: int
    prompt: str = ""

    @property
    def missing(self) -> int:
        return max(0, self.images - self.labels)

    @property
    def is_labelled(self) -> bool:
        return self.images > 0 and self.missing == 0

    def todo(self, *, relabel: bool) -> int:
        """Photos this run would send through the model."""
        return self.images if relabel else self.missing


@dataclass(frozen=True)
class Plan:
    """What pressing Start would do, in numbers the page can show."""

    classes: tuple[ClassPlan, ...] = ()
    relabel: bool = False
    threshold: float = DEFAULT_THRESHOLD
    max_detections: int = DEFAULT_MAX_DETECTIONS
    model: str = DEFAULT_MODEL

    @property
    def total_images(self) -> int:
        return sum(c.images for c in self.classes)

    @property
    def total_labels(self) -> int:
        return sum(c.labels for c in self.classes)

    @property
    def todo(self) -> int:
        return sum(c.todo(relabel=self.relabel) for c in self.classes)

    @property
    def is_complete(self) -> bool:
        """Whether Step 4 is reachable without labelling anything.

        An empty label file is a real answer -- "no objects in this photo" --
        so this counts files, never their contents.
        """
        return bool(self.classes) and all(c.is_labelled for c in self.classes)

    @property
    def working_classes(self) -> tuple[ClassPlan, ...]:
        return tuple(c for c in self.classes if c.todo(relabel=self.relabel))

    def problems(self) -> list[str]:
        """Why Start is disabled, phrased for the person who has to fix it."""
        if not self.classes:
            return [tr("No prepared dataset. Finish Step 2 first.")]
        missing = [c.name for c in self.working_classes if not c.prompt.strip()]
        if missing:
            names = ", ".join(
                tr('"{name}"', name=name) for name in sorted(missing, key=str.casefold)
            )
            return [
                trn(
                    "{names} still need a description of what to find. "
                    "Type it in the Prompt column.",
                    "{names} still need a description of what to find. "
                    "Type it in the Prompt column.",
                    len(missing),
                    names=names,
                )
            ]
        if not self.todo:
            return [
                tr(
                    "Every photo already has a label. Tick \"Label everything again\" to "
                    "redo them, or carry on to Step 4."
                )
            ]
        return []

    @property
    def is_runnable(self) -> bool:
        return not self.problems()

    def summary(self) -> str:
        if not self.classes:
            return tr("No prepared dataset yet.")
        if self.is_complete and not self.relabel:
            return trn(
                "All {n:,} photos are labelled. You can go straight to Step 4.",
                "All {n:,} photos are labelled. You can go straight to Step 4.",
                self.total_images,
            )
        return trn(
            "{todo:,} of {total:,} photos still need a label across {n} class.",
            "{todo:,} of {total:,} photos still need a label across {n} classes.",
            len(self.working_classes),
            todo=self.todo,
            total=self.total_images,
        )

    def job_classes(self) -> list[dict]:
        """The class list as the worker wants it.

        Classes with nothing to do are still sent: the worker decides what to
        skip by looking at the filesystem, and leaving them out here would mean
        two places had to agree about that.
        """
        return [
            {
                "index": c.index,
                "name": c.name,
                "slug": c.slug,
                "prompt": c.prompt.strip(),
            }
            for c in self.classes
        ]


def build_plan(
    status: DatasetStatus,
    prompts: dict[str, str] | None = None,
    *,
    relabel: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    model: str = DEFAULT_MODEL,
) -> Plan:
    """Combine what is on disk with what the user has typed."""
    prompts = prompts or {}
    classes = tuple(
        ClassPlan(
            index=stat.index,
            name=stat.name,
            slug=stat.slug,
            images=stat.images,
            labels=stat.labels,
            prompt=prompts.get(stat.slug, ""),
        )
        for stat in status.classes
    )
    return Plan(
        classes=classes,
        relabel=relabel,
        threshold=threshold,
        max_detections=max_detections,
        model=model,
    )


# --------------------------------------------------------------------------- #
# The record of what was asked for
# --------------------------------------------------------------------------- #


def record_path(project: Project) -> Path:
    return project.dataset_dir / AUTOLABEL_FILENAME


def read_record(project: Project | None) -> dict | None:
    """The settings the last autolabeling run was given, if there was one.

    This records *what was asked for*, never that the work finished -- which
    class has labels is answered by counting label files, as everywhere else.
    Its only job is that a user returning tomorrow does not retype eight
    prompts, so an unreadable one is not worth an error.
    """
    if project is None:
        return None
    try:
        obj = json.loads(record_path(project).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def prompts_from(record: dict | None) -> dict[str, str]:
    """Slug -> prompt, from a record written by a previous run."""
    if not record:
        return {}
    found: dict[str, str] = {}
    for entry in record.get("classes") or []:
        if isinstance(entry, dict) and entry.get("slug"):
            found[str(entry["slug"])] = str(entry.get("prompt") or "")
    return found


def settings_from(record: dict | None) -> dict:
    """Threshold, box limit and model from a previous run, with defaults."""
    record = record or {}
    try:
        threshold = float(record.get("threshold", DEFAULT_THRESHOLD))
    except (TypeError, ValueError):
        threshold = DEFAULT_THRESHOLD
    try:
        max_detections = int(record.get("max_detections", DEFAULT_MAX_DETECTIONS))
    except (TypeError, ValueError):
        max_detections = DEFAULT_MAX_DETECTIONS
    model = str(record.get("model") or DEFAULT_MODEL)
    if model not in {m.key for m in MODELS}:
        model = DEFAULT_MODEL
    return {
        "threshold": min(0.95, max(0.01, threshold)),
        "max_detections": min(100, max(1, max_detections)),
        "model": model,
    }
