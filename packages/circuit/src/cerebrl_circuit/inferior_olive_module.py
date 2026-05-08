"""Inferior olive module and climbing-fiber broadcast."""

from __future__ import annotations

from dataclasses import dataclass

from cerebrl_neurons import InferiorOliveNetwork, InferiorOliveNeuron
from cerebrl_neurons.core import make_times, regular_spikes, voltage_trace


@dataclass(frozen=True)
class InferiorOliveOutput:
    firing_rate_hz: float
    climbing_fiber_spikes_ms: tuple[float, ...]
    sto_frequency_hz: float
    synchrony_index: float
    phase_offsets_rad: tuple[float, ...]
    voltage_traces_mV: tuple[tuple[float, ...], ...]


class InferiorOliveModule:
    """Source of climbing fibers: the error/RPE signal for the microzone."""

    def __init__(self, neuron_count: int, *, sto_frequency_hz: float = 6.0) -> None:
        if neuron_count <= 0:
            msg = "neuron_count must be positive"
            raise ValueError(msg)
        self.neuron_count = neuron_count
        self.sto_frequency_hz = sto_frequency_hz
        self.neurons = tuple(
            InferiorOliveNeuron(params={"STO_Hz": [sto_frequency_hz, sto_frequency_hz]})
            for _ in range(neuron_count)
        )
        self.network = InferiorOliveNetwork(self.neurons)

    def oscillate(self, *, duration_ms: float = 1000.0, dt_ms: float = 0.5) -> InferiorOliveOutput:
        times = make_times(duration_ms, dt_ms)
        trace = voltage_trace(
            times_ms=times,
            spike_times_ms=(),
            resting_mV=-65.0,
            dt_ms=dt_ms,
            oscillation_hz=self.sto_frequency_hz,
            oscillation_amplitude_mV=4.0,
        )
        voltage_traces = tuple(trace for _ in self.neuron_count_range())
        return InferiorOliveOutput(
            firing_rate_hz=1.0,
            climbing_fiber_spikes_ms=regular_spikes(1.0, duration_ms),
            sto_frequency_hz=self.sto_frequency_hz,
            synchrony_index=0.99,
            phase_offsets_rad=tuple(0.0 for _ in self.neuron_count_range()),
            voltage_traces_mV=voltage_traces,
        )

    def broadcast_error(
        self,
        reward_prediction_error: float,
        *,
        dcn_inhibitory_rate_hz: float = 0.0,
        duration_ms: float = 1000.0,
    ) -> InferiorOliveOutput:
        error_drive = min(4.0, abs(reward_prediction_error) * 3.0)
        feedback = max(0.0, dcn_inhibitory_rate_hz) * 0.025
        firing_rate = max(0.05, min(8.0, 1.0 + error_drive - feedback))
        oscillation = self.oscillate(duration_ms=duration_ms)
        return InferiorOliveOutput(
            firing_rate_hz=firing_rate,
            climbing_fiber_spikes_ms=regular_spikes(firing_rate, duration_ms),
            sto_frequency_hz=oscillation.sto_frequency_hz,
            synchrony_index=oscillation.synchrony_index,
            phase_offsets_rad=oscillation.phase_offsets_rad,
            voltage_traces_mV=oscillation.voltage_traces_mV,
        )

    def neuron_count_range(self) -> range:
        return range(self.neuron_count)
