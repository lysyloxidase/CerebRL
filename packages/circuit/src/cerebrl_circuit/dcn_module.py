"""Deep cerebellar nuclei module."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cerebrl_neurons import DCNInhibitoryNeuron, DCNProjectionNeuron

from .purkinje_layer import PurkinjeLayerOutput


@dataclass(frozen=True)
class DCNOutput:
    projection_rate_hz: float
    baseline_projection_rate_hz: float
    rebound_rate_hz: float
    rebound_spikes_ms: tuple[float, ...]
    motor_command: float
    io_feedback_rate_hz: float
    pc_inhibition_fraction: float


class DCNModule:
    """Deep cerebellar nuclei: integrate PC inhibition + MF excitation.

    Projection neurons are tonically active, suppressed by Purkinje inhibition,
    and rebound-burst when learned PC pauses release T-type Ca channel
    deinactivation. Inhibitory DCN neurons provide nucleo-olivary feedback.
    """

    def __init__(self, projection_count: int, inhibitory_count: int) -> None:
        if projection_count <= 0:
            msg = "projection_count must be positive"
            raise ValueError(msg)
        if inhibitory_count <= 0:
            msg = "inhibitory_count must be positive"
            raise ValueError(msg)
        self.projection_count = projection_count
        self.inhibitory_count = inhibitory_count
        self._projection = DCNProjectionNeuron()
        self._inhibitory = DCNInhibitoryNeuron()

    def response_to_pc_rate(
        self,
        pc_rate_hz: float,
        *,
        mf_collateral_drive: float = 0.0,
        cf_collateral_drive: float = 0.0,
        pc_pause_windows_ms: Sequence[tuple[float, float]] = (),
        duration_ms: float = 500.0,
    ) -> DCNOutput:
        baseline = 50.0 + 35.0 * max(0.0, min(1.0, mf_collateral_drive))
        error_boost = 8.0 * max(0.0, min(1.0, cf_collateral_drive))
        inhibition_fraction = min(0.85, max(0.0, pc_rate_hz) / 70.0 * 0.75)
        suppressed_rate = max(2.0, baseline * (1.0 - inhibition_fraction) + error_boost)

        rebound_trace = self._projection.simulate(
            duration_ms=duration_ms,
            pc_inhibition_windows=pc_pause_windows_ms,
        )
        rebound_spikes = rebound_trace.labels["rebound"]
        rebound_rate = 0.0
        if pc_pause_windows_ms:
            window_ms = sum(
                max(1.0, min(duration_ms, end + 35.0) - end)
                for _, end in pc_pause_windows_ms
            )
            rebound_rate = len(rebound_spikes) / (window_ms / 1000.0)

        projection_rate = suppressed_rate + min(80.0, rebound_rate * 0.25)
        motor_command = projection_rate / 160.0
        inhibitory_trace = self._inhibitory.simulate(duration_ms=duration_ms)
        io_feedback = inhibitory_trace.firing_rate_hz(label="nucleo_olivary") * motor_command
        return DCNOutput(
            projection_rate_hz=projection_rate,
            baseline_projection_rate_hz=baseline,
            rebound_rate_hz=rebound_rate,
            rebound_spikes_ms=rebound_spikes,
            motor_command=motor_command,
            io_feedback_rate_hz=io_feedback,
            pc_inhibition_fraction=inhibition_fraction,
        )

    def integrate(
        self,
        purkinje_output: PurkinjeLayerOutput,
        *,
        mf_collateral_drive: float,
        cf_collateral_drive: float,
        duration_ms: float = 500.0,
    ) -> DCNOutput:
        return self.response_to_pc_rate(
            purkinje_output.dcn_inhibition_rate_hz,
            mf_collateral_drive=mf_collateral_drive,
            cf_collateral_drive=cf_collateral_drive,
            pc_pause_windows_ms=purkinje_output.pause_windows_ms,
            duration_ms=duration_ms,
        )
