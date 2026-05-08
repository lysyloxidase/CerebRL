"""Purkinje layer linear readout of the parallel-fiber expansion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cerebrl_neurons import PurkinjeCell

from .granular_layer import GranularLayerOutput
from .molecular_layer import MolecularLayerOutput


@dataclass(frozen=True)
class PurkinjeLayerOutput:
    pc_count: int
    simple_spike_rate_hz: float
    complex_spike_count: int
    spike_times_ms: tuple[float, ...]
    simple_spike_times_ms: tuple[float, ...]
    complex_event_times_ms: tuple[float, ...]
    complex_burst_spikes_ms: tuple[float, ...]
    pause_windows_ms: tuple[tuple[float, float], ...]
    dcn_inhibition_rate_hz: float


class PurkinjeLayer:
    """PC population, the sole output of cerebellar cortex."""

    def __init__(self, purkinje_count: int) -> None:
        if purkinje_count <= 0:
            msg = "purkinje_count must be positive"
            raise ValueError(msg)
        self.purkinje_count = purkinje_count
        self._representative = PurkinjeCell()

    def readout(
        self,
        granular_output: GranularLayerOutput,
        molecular_output: MolecularLayerOutput,
        *,
        climbing_fiber_spikes_ms: Sequence[float] = (),
        duration_ms: float = 1000.0,
        dt_ms: float = 0.1,
    ) -> PurkinjeLayerOutput:
        pf_drive = max(
            0.0,
            min(
                0.24,
                granular_output.parallel_fiber_drive * molecular_output.dendritic_gain
                - molecular_output.somatic_inhibition * 0.02,
            ),
        )
        trace = self._representative.simulate(
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            parallel_fiber_drive=pf_drive,
            climbing_fiber_times_ms=climbing_fiber_spikes_ms,
        )
        pause_windows = tuple(
            (event_ms, min(duration_ms, event_ms + 15.0))
            for event_ms in trace.labels["complex"]
        )
        simple_spikes = tuple(
            spike
            for spike in trace.labels["simple"]
            if not any(start <= spike <= end for start, end in pause_windows)
        )
        simple_rate = len(simple_spikes) / (duration_ms / 1000.0)
        return PurkinjeLayerOutput(
            pc_count=self.purkinje_count,
            simple_spike_rate_hz=simple_rate,
            complex_spike_count=len(trace.labels["complex"]),
            spike_times_ms=trace.spike_times_ms,
            simple_spike_times_ms=simple_spikes,
            complex_event_times_ms=trace.labels["complex"],
            complex_burst_spikes_ms=trace.labels["complex_burst_spikes"],
            pause_windows_ms=pause_windows,
            dcn_inhibition_rate_hz=simple_rate,
        )
