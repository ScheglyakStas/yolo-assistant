"""Build the installer for the platform this runs on.

    python packaging/build.py                  # Ubuntu: a .deb. Windows: a setup .exe
    python packaging/build.py --no-installer   # stop at the PyInstaller folder
    python packaging/build.py --no-uv          # leave uv out; Step 1 downloads it

Run it with the GUI's own interpreter, with ``requirements/gui.txt`` and
PyInstaller installed (``pip install -e .[build]``). PyInstaller cannot
cross-compile, so the Windows installer is built on Windows -- with Inno Setup
6 installed -- and the .deb on Ubuntu. Build the .deb on the *oldest* Ubuntu it
should install on: the bundle carries this machine's glibc requirement, and the
package says so in its ``Depends``.

Output goes to ``dist/``. Nothing here needs a GPU or the managed environment;
the only network access is fetching uv, once, checked against the same pinned
SHA-256 the app itself uses.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
DIST = ROOT / "dist"

PACKAGE = "yolo-assistant"
DISPLAY_NAME = "YOLO Assistant"
EXE_NAME = "YOLOAssistant" if os.name == "nt" else PACKAGE
ICON_SVG = ROOT / "yolo_assistant" / "app" / "resources" / "icon.svg"

#: Sizes a desktop asks for. The .ico carries the Windows ones; Linux gets a PNG
#: per size beside the SVG, for the panels and docks that cannot draw an SVG.
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)

DEFAULT_MAINTAINER = "YOLO Assistant <yolo-assistant@localhost>"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-installer", action="store_true",
                        help="Build and check the application folder, but do not package it.")
    parser.add_argument("--no-uv", action="store_true",
                        help="Do not bundle uv. Step 1 then downloads it on first run.")
    parser.add_argument("--maintainer", default=os.environ.get("UMLA_MAINTAINER", DEFAULT_MAINTAINER),
                        help="The .deb's Maintainer field, as 'Name <email>'.")
    args = parser.parse_args(argv)

    if sys.platform not in ("linux", "win32"):
        parser.error("installers are built on Linux (.deb) or Windows (setup .exe) only")

    version = read_version()
    stage = BUILD / "stage"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    step(f"Building {DISPLAY_NAME} {version}")
    render_icons(stage)
    if not args.no_uv:
        stage_uv(stage)
    if os.name == "nt":
        write_version_info(stage, version)

    app_dir = run_pyinstaller(stage)
    self_check(app_dir)

    if args.no_installer:
        step(f"Application folder: {app_dir}")
        return 0
    if os.name == "nt":
        installer = build_setup_exe(app_dir, version, stage)
    else:
        installer = build_deb(app_dir, version, stage, args.maintainer)
    step(f"Installer: {installer}  ({installer.stat().st_size / 1e6:.0f} MB)")
    return 0


def step(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def read_version() -> str:
    """The version in ``yolo_assistant/__init__.py``, without importing Qt."""
    text = (ROOT / "yolo_assistant" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise SystemExit("no __version__ in yolo_assistant/__init__.py")
    return match.group(1)


# -- staged inputs ----------------------------------------------------------- #


def render_icons(stage: Path) -> None:
    """PNGs, and on Windows an .ico, all drawn from the one SVG."""
    step("Rendering icons")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    _app = QGuiApplication.instance() or QGuiApplication([])
    renderer = QSvgRenderer(str(ICON_SVG))
    if not renderer.isValid():
        raise SystemExit(f"{ICON_SVG} is not a valid SVG")
    icons = stage / "icons"
    icons.mkdir()
    for size in ICON_SIZES:
        image = QImage(size, size, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
        painter.end()
        image.save(str(icons / f"{size}.png"))

    if os.name == "nt":
        from PIL import Image

        # Pillow downscales the largest image for each size it is asked for, so
        # the small sizes come from its resampler rather than Qt's renderer.
        # That is what every other .ico tool does too.
        with Image.open(icons / "256.png") as largest:
            largest.save(stage / "icon.ico", sizes=[(s, s) for s in ICO_SIZES])


def stage_uv(stage: Path) -> None:
    """The pinned uv for this platform, fetched and checksum-verified by the app's own code.

    Kept in ``build/uv-<version>`` between builds, so only the first one needs
    the network.
    """
    sys.path.insert(0, str(ROOT))
    from yolo_assistant.app.core import paths
    from yolo_assistant.app.core.bootstrap import uv_manager

    step(f"Fetching uv {uv_manager.UV_VERSION}")
    previous = os.environ.get(paths.DATA_DIR_ENV)
    os.environ[paths.DATA_DIR_ENV] = str(BUILD / f"uv-{uv_manager.UV_VERSION}")
    try:
        uv = uv_manager.managed_uv()
        if uv.exists():
            print(f"    reusing {uv}")
        else:
            uv = uv_manager.download_uv()
            print(f"    verified and saved {uv}")
    finally:
        if previous is None:
            os.environ.pop(paths.DATA_DIR_ENV, None)
        else:
            os.environ[paths.DATA_DIR_ENV] = previous
    (stage / "uv").mkdir()
    shutil.copy2(uv, stage / "uv" / uv.name)


def write_version_info(stage: Path, version: str) -> None:
    """The Windows version resource: what Explorer's Details tab and Task Manager show."""
    from packaging.version import Version

    numbers = (tuple(Version(version).release) + (0, 0, 0, 0))[:4]
    strings = {
        "CompanyName": DISPLAY_NAME,
        "FileDescription": DISPLAY_NAME,
        "FileVersion": version,
        "InternalName": EXE_NAME,
        "LegalCopyright": "Licensed under the GNU AGPL v3",
        "OriginalFilename": f"{EXE_NAME}.exe",
        "ProductName": DISPLAY_NAME,
        "ProductVersion": version,
    }
    table = ",\n          ".join(f"StringStruct({k!r}, {v!r})" for k, v in strings.items())
    (stage / "version_info.txt").write_text(
        f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={numbers}, prodvers={numbers}, mask=0x3f, flags=0x0,
                    OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
          {table}])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])]),
  ],
)
""",
        encoding="utf-8",
    )


# -- the application folder -------------------------------------------------- #


def run_pyinstaller(stage: Path) -> Path:
    step("Running PyInstaller")
    env = dict(os.environ, UMLA_STAGE_DIR=str(stage), UMLA_EXE_NAME=EXE_NAME)
    # A ROS or similar PYTHONPATH would be analysed into the bundle.
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [
            sys.executable, "-m", "PyInstaller", str(ROOT / "packaging" / "umla.spec"),
            "--noconfirm", "--clean",
            "--distpath", str(BUILD / "pyinstaller" / "dist"),
            "--workpath", str(BUILD / "pyinstaller" / "work"),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    return BUILD / "pyinstaller" / "dist" / EXE_NAME


def executable(app_dir: Path) -> Path:
    return app_dir / (f"{EXE_NAME}.exe" if os.name == "nt" else EXE_NAME)


def self_check(app_dir: Path) -> None:
    """Run the frozen app's ``--self-check`` against a throwaway data directory."""
    step("Checking the frozen application")
    with tempfile.TemporaryDirectory(prefix="umla-self-check-") as data:
        env = dict(os.environ, UMLA_DATA_DIR=data, QT_QPA_PLATFORM="offscreen")
        env.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [str(executable(app_dir)), "--self-check"], env=env, timeout=300, check=False
        )
    if completed.returncode != 0:
        raise SystemExit(
            f"the frozen application failed its self-check (exit {completed.returncode}); "
            "not packaging it"
        )


# -- Ubuntu ------------------------------------------------------------------ #

DESKTOP_ENTRY = f"""[Desktop Entry]
Type=Application
Version=1.5
Name={DISPLAY_NAME}
GenericName=YOLO training assistant
GenericName[de]=YOLO-Trainingsassistent
GenericName[pl]=Asystent treningu YOLO
GenericName[ru]=Помощник обучения YOLO
GenericName[be]=Памочнік навучання YOLO
Comment=Label a photo dataset, then train, export and test a YOLO model
Comment[de]=Einen Fotodatensatz beschriften, dann ein YOLO-Modell trainieren, exportieren und testen
Comment[pl]=Oznacz zbiór zdjęć, a potem wytrenuj, wyeksportuj i przetestuj model YOLO
Comment[ru]=Разметить набор фотографий, затем обучить, экспортировать и проверить модель YOLO
Comment[be]=Размеціць набор фотаздымкаў, потым навучыць, экспартаваць і праверыць мадэль YOLO
Exec={PACKAGE}
Icon={PACKAGE}
Terminal=false
Categories=Science;ArtificialIntelligence;
Keywords=YOLO;Ultralytics;object detection;training;dataset;labeling;
StartupWMClass={PACKAGE}
"""

DEB_DESCRIPTION = """Guided GUI for the YOLO workflow
 Walks through environment setup, dataset preparation, autolabeling with
 OWLv2, label verification, training, export and testing of Ultralytics YOLO
 models, without a terminal.
 .
 PyTorch, Ultralytics and the rest of the GPU stack are not in this package.
 On first run the app downloads them (10-14 GB), matched to the installed
 NVIDIA driver, into ~/.local/share/yolo-assistant. Removing the
 package leaves that directory in place; delete it to reclaim the space."""

DEB_COPYRIGHT = f"""Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: {DISPLAY_NAME}

Files: *
License: AGPL-3.0-only
 The full text is in /opt/{PACKAGE}/_internal/LICENSE.

Files: opt/{PACKAGE}/_internal/*
Comment: The bundled Python runtime, Qt and PySide6 (LGPL-3.0), Pillow,
 psutil, PyYAML, nvidia-ml-py, packaging and uv are distributed under their
 own licences, unmodified.
"""


def build_deb(app_dir: Path, version: str, stage: Path, maintainer: str) -> Path:
    step("Building the .deb")
    architecture = _output(["dpkg", "--print-architecture"])
    root = BUILD / "deb" / PACKAGE
    shutil.rmtree(root.parent, ignore_errors=True)

    install_dir = root / "opt" / PACKAGE
    shutil.copytree(app_dir, install_dir, symlinks=True)
    (root / "usr" / "bin").mkdir(parents=True)
    (root / "usr" / "bin" / PACKAGE).symlink_to(f"/opt/{PACKAGE}/{PACKAGE}")

    applications = root / "usr" / "share" / "applications"
    applications.mkdir(parents=True)
    desktop = applications / f"{PACKAGE}.desktop"
    desktop.write_text(DESKTOP_ENTRY, encoding="utf-8")
    if shutil.which("desktop-file-validate"):
        subprocess.run(["desktop-file-validate", str(desktop)], check=True)

    hicolor = root / "usr" / "share" / "icons" / "hicolor"
    (hicolor / "scalable" / "apps").mkdir(parents=True)
    shutil.copy2(ICON_SVG, hicolor / "scalable" / "apps" / f"{PACKAGE}.svg")
    for size in ICON_SIZES:
        target = hicolor / f"{size}x{size}" / "apps"
        target.mkdir(parents=True)
        shutil.copy2(stage / "icons" / f"{size}.png", target / f"{PACKAGE}.png")

    doc = root / "usr" / "share" / "doc" / PACKAGE
    doc.mkdir(parents=True)
    (doc / "copyright").write_text(DEB_COPYRIGHT, encoding="utf-8")

    depends = deb_depends(install_dir)
    print("    Depends: " + ", ".join(depends), flush=True)
    installed_kib = sum(
        path.lstat().st_size for path in root.rglob("*") if not path.is_dir()
    ) // 1024
    debian = root / "DEBIAN"
    debian.mkdir()
    (debian / "control").write_text(
        f"Package: {PACKAGE}\n"
        f"Version: {version}\n"
        f"Architecture: {architecture}\n"
        f"Maintainer: {maintainer}\n"
        f"Installed-Size: {installed_kib}\n"
        f"Depends: {', '.join(depends)}\n"
        "Section: science\n"
        "Priority: optional\n"
        f"Description: {DEB_DESCRIPTION}\n",
        encoding="utf-8",
    )

    _normalise_modes(root)
    DIST.mkdir(exist_ok=True)
    output = DIST / f"{PACKAGE}_{version}_{architecture}.deb"
    subprocess.run(
        ["dpkg-deb", "--root-owner-group", "-Zxz", "--build", str(root), str(output)],
        check=True,
    )
    return output


def deb_depends(app_dir: Path) -> list[str]:
    """The packages providing every shared library the bundle loads from the system.

    Worked out from the bundle rather than written down, because the list is
    whatever PyInstaller leaves out, and that changes between its releases.
    Today it bundles the xcb libraries Qt's X11 plugin needs -- including
    ``libxcb-cursor0``, which a stock Ubuntu desktop lacks -- and deliberately
    leaves out OpenGL, EGL, Wayland and DRM, which have to be the system's own
    because they are how the graphics driver is reached. A library already
    inside the bundle is skipped: the bootloader puts the bundle first on the
    search path, so the bundled copy is the one that loads.

    glibc is versioned at what this machine has, because the bundled Python and
    Qt were linked against it; on an older Ubuntu dpkg then refuses the package
    with a reason instead of installing an app that cannot start.
    """
    bundled = {path.name for path in app_dir.rglob("*") if path.is_file()}
    system: set[str] = set()
    missing: dict[str, set[str]] = {}
    for path in sorted(app_dir.rglob("*")):
        if not path.is_file() or path.is_symlink() or not _is_elf(path):
            continue
        result = subprocess.run(["ldd", str(path)], capture_output=True, text=True, check=False)
        for line in result.stdout.splitlines():
            match = re.match(r"\s*(\S+) => (.+?)(?: \(0x[0-9a-f]+\))?$", line)
            if match is None or match.group(1) in bundled:
                continue
            name, location = match.groups()
            if location == "not found":
                missing.setdefault(name, set()).add(path.relative_to(app_dir).as_posix())
            else:
                system.add(location)

    if missing:
        print("    WARNING: these libraries are not installed here, so the files needing "
              "them are not covered by Depends:")
        for name, users in sorted(missing.items()):
            print(f"      {name}: {', '.join(sorted(users)[:3])}")

    packages = _owning_packages(system)
    libc = _output(["dpkg-query", "-W", "-f=${Version}", "libc6"])
    libc_upstream = re.match(r"[\d.]+", libc.split(":")[-1])
    packages.discard("libc6")
    depends = [f"libc6 (>= {libc_upstream.group(0).rstrip('.')})" if libc_upstream else "libc6"]
    return depends + sorted(packages)


def _owning_packages(paths: set[str]) -> set[str]:
    """``dpkg -S`` for each path, trying both spellings of a merged-/usr path."""
    owners: dict[str, str] = {}
    for spelling in (lambda p: p, os.path.realpath, _toggle_usr):
        wanted = sorted({spelling(p) for p in paths if p not in owners})
        if not wanted:
            break
        result = subprocess.run(["dpkg", "-S", *wanted], capture_output=True, text=True,
                                check=False)
        found = {}
        for line in result.stdout.splitlines():
            package, _, path = line.partition(": ")
            if path and not package.startswith("diversion"):
                found[path.strip()] = package.split(":")[0].split(",")[0].strip()
        for original in paths:
            if original not in owners and spelling(original) in found:
                owners[original] = found[spelling(original)]
    unowned = sorted(set(paths) - set(owners))
    if unowned:
        raise SystemExit(
            "no installed package provides these libraries, so the .deb cannot "
            "declare them:\n  " + "\n  ".join(unowned)
        )
    return set(owners.values())


def _toggle_usr(path: str) -> str:
    return path[4:] if path.startswith("/usr/") else "/usr" + path


def _is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


def _normalise_modes(root: Path) -> None:
    """0755 for directories and anything executable, 0644 for the rest.

    The tree inherits this machine's umask and whatever modes the wheels
    shipped with; a package should not.
    """
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        if path.is_dir() or path.stat().st_mode & 0o111:
            path.chmod(0o755)
        else:
            path.chmod(0o644)


def _output(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True, check=True).stdout.strip()


# -- Windows ----------------------------------------------------------------- #


def build_setup_exe(app_dir: Path, version: str, stage: Path) -> Path:
    step("Building the setup program with Inno Setup")
    iscc = _find_iscc()
    DIST.mkdir(exist_ok=True)
    subprocess.run(
        [
            str(iscc),
            f"/DAppVersion={version}",
            f"/DAppExeName={EXE_NAME}.exe",
            f"/DSourceDir={app_dir}",
            f"/DRepoRoot={ROOT}",
            f"/DIconFile={stage / 'icon.ico'}",
            f"/DOutputDir={DIST}",
            str(ROOT / "packaging" / "windows" / "installer.iss"),
        ],
        check=True,
    )
    return DIST / f"{EXE_NAME}-{version}-setup.exe"


def _find_iscc() -> Path:
    found = shutil.which("ISCC")
    if found:
        return Path(found)
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                 os.environ.get("LOCALAPPDATA", "") + "\\Programs"):
        if base:
            candidate = Path(base) / "Inno Setup 6" / "ISCC.exe"
            if candidate.exists():
                return candidate
    raise SystemExit(
        "Inno Setup 6 was not found. Install it from https://jrsoftware.org/isinfo.php "
        "(or 'winget install JRSoftware.InnoSetup'), or pass --no-installer."
    )


if __name__ == "__main__":
    sys.exit(main())
