"""Step 1 through the real widgets, on synthetic hardware.

The hardware is stubbed deliberately: these have to pass on a laptop with no
NVIDIA card and on the machine with a Blackwell GPU, and the interesting cases
-- no driver, an ancient GPU, an environment that installed but cannot see the
card -- are ones no single machine can produce on demand.
"""


import pytest
from PySide6.QtWidgets import QApplication

from yolo_assistant.app.core import env_probe, paths, updates
from yolo_assistant.app.core.bootstrap import venv_manager as vm
from yolo_assistant.app.ui.common import status_row


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path))
    return tmp_path


def machine(*, driver="595.84", cuda="13.2", compute="12.0"):
    gpus = [env_probe.GpuInfo(0, "NVIDIA GeForce RTX 5070 Ti", 16.0, compute)] if driver else []
    return env_probe.SystemReport(
        cpu=env_probe.CpuInfo("AMD Ryzen 9 9950X", 16, 32),
        memory=env_probe.MemoryInfo(64.0, 50.0),
        driver=env_probe.DriverInfo(
            present=bool(driver),
            version=driver or "",
            cuda_version=cuda,
            reason="" if driver else "No usable NVIDIA driver was found (NVML_ERROR_DRIVER_NOT_LOADED).",
        ),
        gpus=gpus,
        os_description="Linux 7.0.0 (x86_64)",
    )


@pytest.fixture
def make_page(qapp, isolated, monkeypatch):
    pages = []

    def build(report=None):
        monkeypatch.setattr(env_probe, "probe_system", lambda: report or machine())
        from yolo_assistant.app.ui.steps.system_step import SystemStep

        page = SystemStep()
        pages.append(page)
        return page

    yield build
    for page in pages:
        page.shutdown()


# -- the three requirement rows -------------------------------------------- #


def test_a_good_machine_with_no_environment_blocks_on_the_environment(make_page):
    page = make_page()
    assert page._driver_row.state == status_row.OK
    # Nothing has imported torch yet, so this must not claim either outcome.
    assert page._cuda_row.state == status_row.UNKNOWN
    assert page._env_row.state == status_row.BAD
    assert not page.is_complete()


def test_a_missing_driver_is_reported_with_somewhere_to_go(make_page):
    page = make_page(machine(driver=""))
    assert page._driver_row.state == status_row.BAD
    # isHidden rather than isVisible: the page has no parent window in a test,
    # so nothing is ever truly visible.
    assert not page._driver_row.action.isHidden()
    assert "nvidia.com" in page._driver_row._detail.text()
    assert not page.is_complete()


def test_a_ready_environment_completes_the_step(make_page):
    page = make_page()
    _install(page, cuda_available=True)
    page.refresh()

    assert page._env_row.state == status_row.OK
    assert page._env_row.action.isHidden(), "nothing left to fix, so no action"
    assert page._driver_row.action.isHidden()
    assert page._cuda_row.state == status_row.OK
    assert "RTX 5070 Ti" in page._cuda_row._detail.text()
    assert page.is_complete()


def test_an_environment_that_cannot_see_the_gpu_is_not_silently_accepted(make_page):
    """The failure this whole step exists to catch: a clean install that would
    train on the CPU for two days without anyone noticing."""
    page = make_page()
    _install(page, cuda_available=False)
    page.refresh()

    assert page._env_row.state == status_row.WARN
    assert page._cuda_row.state == status_row.BAD
    assert "days rather than minutes" in page._cuda_row._detail.text()


def test_a_torch_without_kernels_for_this_gpu_is_reported_before_training(make_page):
    page = make_page()
    _install(page, cuda_available=True, arch_list=["sm_80", "sm_86", "sm_90"])
    page.refresh()

    assert page._cuda_row.state == status_row.BAD
    assert "no kernel image" in page._cuda_row._detail.text()


def test_an_environment_missing_a_package_offers_to_repair_itself(make_page):
    page = make_page()
    _install(page, cuda_available=True, missing=["scipy"])
    page.refresh()

    assert page._env_row.state == status_row.WARN
    assert page._env_row.action.text() == "Repair"
    assert "scipy" in page._env_row._detail.text()
    assert not page.is_complete(), "Step 2 must not unlock onto an environment that cannot work"


def test_an_interrupted_install_offers_repair_rather_than_setup(make_page):
    page = make_page()
    page._spec.python.parent.mkdir(parents=True, exist_ok=True)
    page._spec.python.write_text("")
    page.refresh()

    assert page._env_row.state == status_row.WARN
    assert page._env_row.action.text() == "Repair"


# -- hardware panel, per the specification --------------------------------- #


def test_every_fact_the_spec_asks_for_is_shown(make_page):
    page = make_page()
    text = page.details_text()
    assert "AMD Ryzen 9 9950X" in text and "32 threads" in text
    assert "64.0 GB" in text
    assert "RTX 5070 Ti" in text and "16.0 GB VRAM" in text
    assert "595.84" in text and "13.2" in text
    assert "Ultralytics: Not installed yet" in text


def test_the_facts_can_wrap_instead_of_being_clipped(make_page):
    """A word-wrapped QLabel in a QFormLayout reports a minimal width hint, and
    with the default policy the layout believes it and cuts "62.4 GB total,
    56.5 GB free" off mid-sentence."""
    from PySide6.QtWidgets import QFormLayout, QSizePolicy

    page = make_page()
    assert page._facts.fieldGrowthPolicy() is QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
    for key, label in page._fact_labels.items():
        assert label.wordWrap(), key
        assert label.sizePolicy().horizontalPolicy() is QSizePolicy.Policy.MinimumExpanding, key


def test_the_ultralytics_version_appears_once_the_environment_exists(make_page):
    page = make_page()
    _install(page, cuda_available=True)
    page.refresh()
    assert "8.4.154" in page._fact_labels["ultralytics"].text()


# -- backend selection ------------------------------------------------------ #


def test_the_planned_backend_is_named_before_anything_is_downloaded(make_page):
    page = make_page()
    assert page._plan.backend.key == "cu132"
    assert "CUDA 13.2" in page._cuda_row._detail.text()


def test_choosing_a_backend_manually_moves_the_environment(make_page):
    page = make_page()
    before = page._spec.path
    page._backend.setCurrentIndex(page._backend.findData("cu126"))
    assert page._plan.backend.key == "cu126"
    assert page._spec.path != before, "a different backend must get its own directory"


def test_an_old_gpu_is_planned_onto_an_older_backend(make_page):
    page = make_page(machine(compute="6.1"))
    assert page._plan.backend.key == "cu126"


# -- controls -------------------------------------------------------------- #


def test_every_control_explains_itself_on_hover(make_page):
    """The spec's third requirement, checked on the widgets that actually exist."""
    page = make_page()
    controls = {
        "backend": page._backend,
        "set up": page._setup_button,
        "check for updates": page._updates_button,
        "rebuild": page._rebuild_button,
    }
    missing = [name for name, widget in controls.items() if not widget.toolTip().strip()]
    assert not missing, f"no hover explanation for: {missing}"


def test_update_and_rebuild_are_unavailable_without_an_environment(make_page):
    page = make_page()
    assert not page._updates_button.isEnabled()
    assert not page._rebuild_button.isEnabled()
    assert page._setup_button.isEnabled()
    assert page._setup_button.text() == "Set up environment"


def test_update_becomes_available_once_the_environment_is_ready(make_page):
    page = make_page()
    _install(page, cuda_available=True)
    page.refresh()
    assert page._updates_button.isEnabled()
    assert page._rebuild_button.isEnabled()
    assert page._setup_button.text() == "Rebuild environment"


def test_the_update_check_compares_against_what_is_installed(make_page, monkeypatch):
    page = make_page()
    _install(page, cuda_available=True, ultralytics="8.4.100")
    page.refresh()

    monkeypatch.setattr(updates, "latest_release", lambda **_kw: ("8.4.154", ""))
    seen = []
    monkeypatch.setattr(
        page, "_on_update_checked", lambda info: seen.append(info)
    )
    page._check_updates()
    page._update_check.wait(5000)
    QApplication.instance().processEvents()

    assert seen and seen[0].available
    assert seen[0].installed == "8.4.100" and seen[0].latest == "8.4.154"


def _check_on_launch(page, monkeypatch, latest="8.4.157", error=""):
    """Run the launch-time check to completion and return what it offered."""
    monkeypatch.setattr(updates, "latest_release", lambda **_kw: (latest, error))
    offered = []
    monkeypatch.setattr(
        page._updater, "preview", lambda spec, target: offered.append(target)
    )
    page.on_app_start()
    if page._update_check is not None:
        page._update_check.wait(5000)
    QApplication.instance().processEvents()
    return offered


def test_launching_with_an_environment_checks_for_updates(make_page, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    page = make_page()
    _install(page, cuda_available=True)
    page.refresh()

    clicked = _answer_dialog(QMessageBox.StandardButton.Yes)
    offered = _check_on_launch(page, monkeypatch)

    assert clicked["clicked"], "a newer release found at launch was not offered"
    assert offered == ["ultralytics==8.4.157"]


def test_launching_up_to_date_says_nothing_beyond_the_log(make_page, monkeypatch):
    page = make_page()
    _install(page, cuda_available=True)
    page.refresh()
    shown = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.information", lambda *a, **_kw: shown.append(a)
    )

    offered = _check_on_launch(page, monkeypatch, latest="8.4.154")

    assert offered == [] and shown == []
    assert "newest release" in page._console.toPlainText()
    assert page._updates_button.isEnabled()


def test_launching_offline_says_nothing_beyond_the_log(make_page, monkeypatch):
    page = make_page()
    _install(page, cuda_available=True)
    page.refresh()
    shown = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.information", lambda *a, **_kw: shown.append(a)
    )

    offered = _check_on_launch(page, monkeypatch, latest=None, error="no network")

    assert offered == [] and shown == []
    assert "Could not check for updates" in page._console.toPlainText()


def test_launching_without_an_environment_does_not_ask_pypi(make_page, monkeypatch):
    page = make_page()
    asked = []
    monkeypatch.setattr(
        updates, "latest_release", lambda **_kw: asked.append(1) or ("8.4.157", "")
    )

    page.on_app_start()

    assert page._update_check is None and asked == []


# -- the confirmation dialogs ---------------------------------------------- #
#
# These press the real buttons on the real QMessageBox. Monkeypatching
# ``QMessageBox.question`` would have tested the stub: Qt returns a plain int
# rather than a StandardButton member, so ``answer is StandardButton.Yes`` is
# always false and both buttons silently do nothing.


def _answer_dialog(button, *, timeout_ms=5000):
    """Click ``button`` on the next modal message box; returns a result dict."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QMessageBox

    state = {"clicked": False, "waited": 0, "text": "", "informative": ""}

    def attempt():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QMessageBox) and widget.isVisible():
                target = widget.button(button)
                if target is not None:
                    state["text"] = widget.text()
                    state["informative"] = widget.informativeText()
                    target.click()
                    state["clicked"] = True
                    return
        state["waited"] += 25
        if state["waited"] >= timeout_ms:
            # Never leave a modal exec() with nobody to dismiss it.
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, QMessageBox):
                    widget.reject()
            return
        QTimer.singleShot(25, attempt)

    QTimer.singleShot(0, attempt)
    return state


def _offered_update(page, latest="8.4.157"):
    page._update_info = None
    return updates.UpdateInfo(installed="8.4.154", latest=latest)


def test_accepting_the_update_offer_runs_the_preview(make_page, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    page = make_page()
    _install(page, cuda_available=True)
    page.refresh()
    asked = []
    monkeypatch.setattr(page._updater, "preview", lambda spec, target: asked.append(target))

    clicked = _answer_dialog(QMessageBox.StandardButton.Yes)
    page._on_update_checked(_offered_update(page))

    assert clicked["clicked"], "the offer dialog never appeared"
    assert asked == ["ultralytics==8.4.157"]


def test_declining_the_update_offer_changes_nothing(make_page, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    page = make_page()
    _install(page, cuda_available=True)
    page.refresh()
    asked = []
    monkeypatch.setattr(page._updater, "preview", lambda spec, target: asked.append(target))

    clicked = _answer_dialog(QMessageBox.StandardButton.No)
    page._on_update_checked(_offered_update(page))

    assert clicked["clicked"]
    assert asked == []
    # A button that does nothing and says nothing is indistinguishable from a
    # broken one, which is exactly how this bug was reported.
    assert "declined" in page._console.toPlainText()


def test_accepting_a_rebuild_deletes_the_environment_and_starts_again(make_page):
    from PySide6.QtWidgets import QMessageBox

    page = make_page()
    _install(page, cuda_available=True)
    page.refresh()
    started = []
    page._bootstrap.start = lambda spec: started.append(spec)

    clicked = _answer_dialog(QMessageBox.StandardButton.Yes)
    page._rebuild()

    assert clicked["clicked"]
    assert not page._spec.path.exists()
    assert started == [page._spec]


def test_declining_a_rebuild_leaves_the_environment_alone(make_page):
    from PySide6.QtWidgets import QMessageBox

    page = make_page()
    _install(page, cuda_available=True)
    page.refresh()
    started = []
    page._bootstrap.start = lambda spec: started.append(spec)

    clicked = _answer_dialog(QMessageBox.StandardButton.No)
    page._rebuild()

    assert clicked["clicked"]
    assert page._spec.python.exists()
    assert started == []


def test_the_update_plan_dialog_installs_only_on_the_install_button(make_page, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    page = make_page()
    _install(page, cuda_available=True)
    page.refresh()
    applied = []
    monkeypatch.setattr(page._updater, "apply", lambda spec, target: applied.append(target))
    page._pending_update = "ultralytics==8.4.157"

    clicked = _answer_dialog(QMessageBox.StandardButton.Cancel)
    page._on_update_plan(
        ["Resolved 60 packages", "- torch==2.14.0", "+ torch==2.14.0", "- ultralytics==8.4.154"]
    )
    assert clicked["clicked"] and applied == []
    # The torch lines must be readable without opening "Show Details", because
    # the dialog asks the user to check exactly those.
    assert "- torch==2.14.0" in clicked["informative"]

    clicked = _answer_dialog(QMessageBox.StandardButton.Ok)
    page._on_update_plan(["- ultralytics==8.4.154", "+ ultralytics==8.4.157"])
    assert clicked["clicked"]
    assert applied == ["ultralytics==8.4.157"]


def test_windows_long_paths_are_warned_about_before_the_download(make_page):
    """A disabled long-path setting fails the install halfway through with an
    error that never mentions path length, so it is called out up front."""
    report = machine()
    page = make_page(env_probe.SystemReport(**{**report.__dict__, "long_paths": False}))
    page._bootstrap.start = lambda _spec: None  # no download in a unit test
    page._start_setup()
    assert "long path support is turned off" in page._console.toPlainText()
    assert "DISABLED" in page.details_text()


def test_nothing_is_said_about_long_paths_off_windows(make_page):
    page = make_page()
    assert "long paths" not in page.details_text().lower()


def test_step_one_is_always_reachable(make_page):
    assert make_page().can_enter()


def _install(
    page,
    *,
    cuda_available: bool,
    ultralytics: str = "8.4.154",
    arch_list: list[str] | None = None,
    missing: list[str] | None = None,
) -> None:
    """Fake a finished installation for the page's current backend."""
    spec = page._spec
    spec.python.parent.mkdir(parents=True, exist_ok=True)
    spec.python.write_text("")
    vm.write_ready(
        spec,
        {
            "torch": "2.14.0+cu132",
            "cuda_build": "13.2",
            "cuda_available": cuda_available,
            "ultralytics": ultralytics,
            "packages": {
                name: (None if name in (missing or []) else "1.0")
                for name in vm.PACKAGE_NAMES
            },
            "arch_list": arch_list if arch_list is not None else ["sm_90", "sm_100", "sm_120"],
            "devices": (
                [{"index": 0, "name": "NVIDIA GeForce RTX 5070 Ti", "compute_capability": "12.0"}]
                if cuda_available
                else []
            ),
        },
    )
