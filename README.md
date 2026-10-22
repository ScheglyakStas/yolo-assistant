# YOLO Assistant

A desktop app that takes you from a folder of photos to a tested, exported
object-detection model, without opening a terminal. It walks through the whole
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) workflow in seven
guided steps: environment setup, dataset preparation, automatic labelling,
label review, training, export and testing.

Every setting has a hover explanation, so you do not need to know YOLO before
you start.

- **Runs locally.** Labelling, training and testing all happen on your own GPU.
  Your photos never leave the machine.
- **Sets up its own environment.** The installer is small. On first run the app
  downloads a PyTorch build matched to your driver and GPU, plus everything else
  it needs, into a private folder. Your system Python is not touched, and no
  administrator rights are needed.
- **Windows and Ubuntu**, in English, German, Polish, Russian and Belarusian.

## The seven steps

| Step | What happens |
|---|---|
| 1. System | Checks the NVIDIA driver and GPU, shows the machine's specs, and builds or repairs the Python environment. Also checks for Ultralytics updates. |
| 2. Dataset | Point it at a folder with one sub-folder of photos per class. It copies them into a project, fixes phone-photo rotation, drops duplicates and unreadable files, and writes a fixed train/validation split. |
| 3. Autolabel | Write one short description per class, such as *"a photo of a delivery van"*. [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) draws the boxes. You can stop and resume at any time. |
| 4. Review | Browse the boxes class by class. Add a missed box, delete a wrong one, and jump straight to photos where nothing was found. Optional. |
| 5. Train | Pick a model (YOLO26, YOLO11 or YOLOv8, nano to extra-large), the number of epochs and the image size, with optional INT8 quantisation-aware training. Shows live accuracy and GPU load while it runs. |
| 6. Export | Convert a trained checkpoint to ONNX, TensorRT or another Ultralytics format at FP32, FP16 or INT8, with a one-click DeepStream preset. |
| 7. Test | Run the exported model on a photo, a video or a live camera, and see the detections and the time per frame. |

Each step unlocks once the one before it is done. A step you have already done,
or done outside the app, can be skipped.

## Requirements

- **Windows 10 or 11 (x64)**, or **Ubuntu 24.04 or newer (x86-64)**.
- **An NVIDIA GPU**, Maxwell (GTX 900 series) or newer, with a recent driver:
  527.41 or later on Windows, 525.60 or later on Linux. A newer driver unlocks
  newer CUDA builds. You do not need to install CUDA yourself.
- **About 15 GB of free disk space** and an internet connection for the first
  run. Step 1 downloads 10–14 GB from PyPI and the PyTorch package index.
  After that, only downloading new model weights needs the network.

**For INT8 quantisation-aware training on Windows only:** training runs much
faster with the Visual Studio Build Tools (the *Desktop development with C++*
workload) and a CUDA Toolkit of the same major version as the installed PyTorch.
Without them it still works, just more slowly, and Step 5 tells you what is
missing.

## Installing

Download the installer for your system from the
[Releases](../../releases) page.

**Windows:** run `YOLOAssistant-<version>-setup.exe`. It installs for the
current user only and needs no administrator rights. To uninstall, use
*Settings > Apps*. The uninstaller offers to delete the downloaded environment
too, but it never deletes your projects.

**Ubuntu:**

```bash
sudo apt install ./yolo-assistant_<version>_amd64.deb
```

Start it from the applications menu (*YOLO Assistant*), or run `yolo-assistant`.
To uninstall, run `sudo apt remove yolo-assistant`. This leaves
`~/.local/share/yolo-assistant` behind, which holds the downloaded environment
and, by default, your projects. Delete that folder yourself once you have moved
out anything you want to keep.

### Where things are stored

Everything the app downloads is kept in one folder:

| Windows | Ubuntu |
|---|---|
| `%LOCALAPPDATA%\YOLOAssistant` | `~/.local/share/yolo-assistant` |

To use a different folder, for example a larger drive, set the
`UMLA_DATA_DIR` environment variable.

## Using it

1. **Prepare a folder of photos**, with one sub-folder per class. The folder
   names become the class names. Nothing else is needed: no YAML, no label
   files, no `classes.txt`.

   ```
   my-photos/
       cat/       IMG_0001.jpg  IMG_0002.jpg  ...
       dog/       ...
       bicycle/   ...
   ```

2. **Start the app and follow the steps.** On first run, Step 1 builds the
   environment. This is the long download, and it only happens once.
3. **Find your results in the project folder.** It uses the standard
   Ultralytics layout, so you can keep working on it with the `yolo` command
   line if you outgrow the app:

   ```
   <project>/
       dataset/    photos, labels, data.yaml, and the train/val split
       runs/       one folder per training run, with weights/best.pt and last.pt
       exports/    exported models, ready to deploy
   ```

### Command-line options

| Option | What it does |
|---|---|
| `--language <code>` | Starts in a language (`en`, `de`, `pl`, `ru`, `be`) for this session only. To change it permanently, use the **Language** menu. |
| `--self-check` | Checks that the installation is intact. Prints one line per check, and exits non-zero if anything is wrong. |
| `--version` | Prints the version. |

## Running from source

You need Python 3.11 or newer.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m yolo_assistant.app.main

# The test suite needs no GPU, no network and no managed environment.
.venv/bin/python -m pytest -q
```

On Windows, use `.venv\Scripts\python` in place of `.venv/bin/python`.

### Building the installers

```bash
.venv/bin/python -m pip install -e ".[build]"
.venv/bin/python packaging/build.py      # Ubuntu: dist/*.deb   Windows: dist/*-setup.exe
```

PyInstaller cannot cross-compile, so each installer is built on its own
platform. On Windows you also need [Inno Setup 6](https://jrsoftware.org/isinfo.php)
(`winget install JRSoftware.InnoSetup`). Build the `.deb` on the oldest Ubuntu
release it should support, and set `UMLA_MAINTAINER="Name <email>"` first.

## License

YOLO Assistant is free software, licensed under the
[GNU Affero General Public License v3.0](LICENSE). It is built on
[Ultralytics](https://github.com/ultralytics/ultralytics), which is also
AGPL-3.0. If you make a modified version available to users over a network,
section 13 of the AGPL applies.

Autolabelling uses Google's OWLv2 through Hugging Face
[Transformers](https://github.com/huggingface/transformers). The interface
font is [Ubuntu Sans](https://github.com/canonical/Ubuntu-Sans-fonts), used
under the [Ubuntu Font Licence](yolo_assistant/app/resources/fonts/LICENCE.txt).
