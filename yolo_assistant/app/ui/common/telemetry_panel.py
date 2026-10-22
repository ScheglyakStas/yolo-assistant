"""Live GPU panel: utilisation, VRAM, power and temperature.

The spec asks for this during autolabeling and training; it is equally useful
during inference, so one widget serves all three. It owns its sampler, which
runs whether or not a job is active -- an idle reading is how you notice another
application is already holding 6 GB of your VRAM.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from ...core.telemetry import GpuSample, GpuStatic, TelemetrySampler
from .meters import Meter


class TelemetryPanel(QGroupBox):
    def __init__(self, parent: QWidget | None = None, *, interval_ms: int = 500) -> None:
        super().__init__("GPU", parent)
        self._statics: list[GpuStatic] = []

        self._device_picker = QComboBox()
        self._device_picker.setVisible(False)  # only shown for multi-GPU machines
        self._device_picker.currentIndexChanged.connect(self._on_device_changed)

        self._status = QLabel(tr("Starting telemetry..."))
        self._status.setWordWrap(True)

        self._utilisation = Meter(tr("Utilisation"))
        self._memory = Meter(tr("VRAM"))
        self._power = Meter(tr("Power"))
        self._temperature = Meter(tr("Temperature"))

        header = QHBoxLayout()
        header.addWidget(self._status, 1)
        header.addWidget(self._device_picker)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.addLayout(header)
        for meter in (self._utilisation, self._memory, self._power, self._temperature):
            layout.addWidget(meter)

        self._sampler = TelemetrySampler(interval_ms=interval_ms, parent=self)
        self._sampler.devices_discovered.connect(self._on_devices)
        self._sampler.sampled.connect(self._on_sample)
        self._sampler.unavailable.connect(self._on_unavailable)
        self._sampler.start()

    # -- slots ------------------------------------------------------------- #

    def _on_devices(self, statics: list[GpuStatic]) -> None:
        self._statics = statics
        self._device_picker.blockSignals(True)
        self._device_picker.clear()
        for static in statics:
            self._device_picker.addItem(f"GPU {static.index}: {static.name}")
        self._device_picker.blockSignals(False)
        self._device_picker.setVisible(len(statics) > 1)
        if statics:
            first = statics[0]
            self._status.setText(
                tr(
                    "{name} - driver {driver}, CUDA {cuda}, compute {capability}",
                    name=first.name,
                    driver=first.driver_version,
                    cuda=first.cuda_driver_version,
                    capability=first.compute_capability,
                )
            )

    def _on_sample(self, sample: GpuSample) -> None:
        if sample.utilisation_pct is None:
            self._utilisation.set_unsupported()
        else:
            self._utilisation.set_value(sample.utilisation_pct / 100.0, f"{sample.utilisation_pct:.0f} %")

        if sample.memory_total_gb:
            self._memory.set_value(
                sample.memory_fraction,
                tr(
                    "{used:.2f} / {total:.2f} GB",
                    used=sample.memory_used_gb,
                    total=sample.memory_total_gb,
                ),
            )
        else:
            self._memory.set_unsupported()

        if sample.power_w is None:
            self._power.set_unsupported()
        elif sample.power_limit_w:
            self._power.set_value(sample.power_fraction, f"{sample.power_w:.0f} / {sample.power_limit_w:.0f} W")
        else:
            self._power.set_value(None, f"{sample.power_w:.0f} W")

        if sample.temperature_c is None:
            self._temperature.set_unsupported()
        else:
            # No hard maximum is reported, so scale against a throttling-ish
            # ceiling purely to give the bar a meaningful sweep.
            self._temperature.set_value(sample.temperature_c / 95.0, f"{sample.temperature_c:.0f} °C")

    def _on_unavailable(self, reason: str) -> None:
        self._status.setText(reason)
        for meter in (self._utilisation, self._memory, self._power, self._temperature):
            meter.clear()

    def _on_device_changed(self, index: int) -> None:
        if 0 <= index < len(self._statics):
            self._sampler.set_device(index)

    # -- lifecycle --------------------------------------------------------- #

    def shutdown(self) -> None:
        self._sampler.stop()
        self._sampler.wait(2000)
