# PyInstaller spec for the GUI. Run through packaging/build.py, which prepares
# the stage directory this reads from; it is not meant to be run by hand.
#
# What goes in, and why each file sits where it does: the app finds its files
# relative to its own modules (``Path(__file__).parents[n]``), and in a frozen
# build every module's ``__file__`` is under the bundle directory at its
# package path. So each data file is placed at the same path, relative to the
# bundle, that it has relative to the repository root -- and the code that
# finds it needs no frozen-build branch at all.

import os
from pathlib import Path

ROOT = Path(SPECPATH).parent  # noqa: F821 - defined by PyInstaller
STAGE = Path(os.environ["UMLA_STAGE_DIR"])
NAME = os.environ["UMLA_EXE_NAME"]

datas = [
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "requirements" / "heavy.txt"), "requirements"),
]


def add_tree(directory, keep):
    for source in sorted((ROOT / directory).rglob("*")):
        if source.is_file() and "__pycache__" not in source.parts and keep(source):
            datas.append((str(source), str(source.parent.relative_to(ROOT))))


# Every resource, rather than a list of patterns: a file added to resources/
# later and missed here would be a crash in the installed app only.
add_tree("yolo_assistant/app/resources", lambda path: path.suffix != ".py")
# The worker runs in the managed environment's Python, which cannot import
# anything out of PyInstaller's archive -- so it ships as loose source, and
# app/core/paths.py copies it out to the data directory on first use.
add_tree("worker", lambda path: path.suffix == ".py")

binaries = []
uv = STAGE / "uv" / ("uv.exe" if os.name == "nt" else "uv")
if uv.exists():
    # Found by uv_manager.bundled_uv(); saves Step 1 a download from GitHub.
    binaries.append((str(uv), "tools"))

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    # Nothing heavy is ever imported by the GUI, but a stray environment can
    # make the analysis find them anyway, and each is hundreds of megabytes.
    excludes=["torch", "torchvision", "ultralytics", "transformers", "cv2", "numpy",
              "scipy", "matplotlib", "tkinter", "worker"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    console=False,
    icon=str(STAGE / "icon.ico") if os.name == "nt" else None,
    version=str(STAGE / "version_info.txt") if os.name == "nt" else None,
    # UPX-compressed Qt libraries are a common source of antivirus false
    # positives on Windows, and save little against the installer's own
    # compression.
    upx=False,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    upx=False,
    name=NAME,
)
