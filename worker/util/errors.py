"""Failures a command can raise that already know their own remedy.

``worker/__main__.py`` guesses an :class:`ErrorCode` from the text of whatever
exception escaped, which is the right thing to do for a ``RuntimeError`` thrown
out of the depths of CUDA. When a command knows exactly what went wrong there is
no reason to guess: raising one of these carries the code and the sentence the
user should read straight to the GUI's error catalogue.
"""

from __future__ import annotations

from worker.i18n import N_, tr
from worker.protocol import ErrorCode


class WorkerError(Exception):
    """A failure with a code and a hint attached."""

    code: str = ErrorCode.UNKNOWN.value
    default_hint: str = ""

    def __init__(self, msg: str, *, hint: str = "") -> None:
        super().__init__(msg)
        self.hint = hint or tr(self.default_hint)


class DatasetError(WorkerError):
    code = ErrorCode.DATASET_INVALID.value
    default_hint = N_(
        "Check the photo folder in Step 2: it needs one sub-folder per class, "
        "with that class's photos directly inside it."
    )


class GpuUnavailable(WorkerError):
    """The work needs a GPU and torch cannot see one."""

    code = ErrorCode.NO_CUDA_RUNTIME.value
    default_hint = N_(
        "Go back to Step 1 and let the app set up its Python environment again. "
        "If the driver is missing, install it first and restart the computer."
    )


class QatUnavailable(WorkerError):
    """INT8 quantisation-aware training was asked for and cannot be done.

    Its own code rather than a generic one because the remedy is specific and
    the app can offer it: train without the toggle and quantise at export time
    instead, which costs some accuracy and needs nothing installed.
    """

    code = ErrorCode.QAT_UNAVAILABLE.value
    default_hint = N_(
        "Open Step 1 and press Repair, which installs what the environment is "
        "missing without downloading everything again. Failing that, train with "
        "INT8 QAT switched off and choose INT8 when you export in Step 6: that "
        "quantises the finished model instead of training it to expect "
        "quantisation, which needs nothing extra and usually costs a little accuracy."
    )


class ModelUnavailable(WorkerError):
    """A pretrained model could not be fetched or loaded."""

    code = ErrorCode.DOWNLOAD_FAILED.value
    default_hint = N_(
        "The model is downloaded once, the first time it is used, and needs an "
        "internet connection. Check the connection, or any proxy that blocks "
        "huggingface.co, and try again."
    )


class ExportFailed(WorkerError):
    """A conversion in Step 6 did not produce a usable file."""

    code = ErrorCode.EXPORT_FAILED.value
    default_hint = N_(
        "ONNX is the format most likely to succeed; try it first to find out "
        "whether the problem is the model or the format. Exporting at FP32 also "
        "rules out the precision as the cause."
    )


class QatFormatMismatch(WorkerError):
    """An INT8 QAT model was asked for at a precision or format it cannot have.

    Its own code because the app can offer the remedy directly, and because the
    exporter's own complaint names a dtype or a node type rather than the
    quantisation-aware training two steps back that put it there.
    """

    code = ErrorCode.QAT_FORMAT_MISMATCH.value
    default_hint = N_(
        "Choose INT8 in the precision list. A model trained with INT8 QAT already "
        "carries its quantisers, and exporting it at any other precision throws "
        "that training away."
    )


class SourceUnreadable(WorkerError):
    """The photo or video Step 7 was pointed at cannot be read.

    Distinct from a missing file, which the command checks for separately: this
    is a file that exists and that OpenCV opens to nothing -- a container it was
    not built with, a codec that needs a system library, or a download that
    stopped halfway. The remedy differs from every other failure in the step,
    so it does not share a code with them.
    """

    code = ErrorCode.SOURCE_UNREADABLE.value
    default_hint = N_(
        "Try a different file to find out whether the problem is this one or the "
        "format. MP4 with H.264, and JPEG or PNG, are the safest to test with."
    )


class CameraUnavailable(WorkerError):
    """A camera was chosen and cannot be opened.

    Almost always because something else already has it -- a video call, the
    system camera app, another copy of this one -- because most drivers hand a
    camera to one process at a time.
    """

    code = ErrorCode.CAMERA_UNAVAILABLE.value
    default_hint = N_(
        "Close anything else using the camera -- a video call or the system "
        "camera app -- and press Refresh. On Linux, check that your user is in "
        "the 'video' group."
    )


class InferenceFailed(WorkerError):
    """The model loaded and then could not be run.

    Its own code because the remedy is about the *file* rather than about the
    step: an engine built on another machine, an ONNX asked for a GPU whose
    runtime is not installed, a model whose input size the frames do not match.
    """

    code = ErrorCode.INFERENCE_FAILED.value
    default_hint = N_(
        "Export the model again in Step 6, or export it to ONNX, which is the "
        "format most likely to run anywhere. A TensorRT engine only works on the "
        "machine and driver that built it."
    )
