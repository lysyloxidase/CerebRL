"""Pontine relay neuron model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .core import (
    BioNeuron,
    SimulationResult,
    default_backend_spec,
    make_times,
    midpoint,
    voltage_trace,
)
from .params import CELL_PARAMS

PONTINE_RELAY_EQUATIONS = """
dv/dt = (-(v - E_L) + R_m * I_cortical) / tau_m : 1 (unless refractory)
dI_cortical/dt = -I_cortical / tau_ampa : amp
relay_latency : second
refractory_timer : second
"""


class PontineRelayNeuron(BioNeuron):
    """One millisecond corticopontine efference-copy relay."""

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        merged_params = dict(CELL_PARAMS["pontine_relay"])
        if params is not None:
            merged_params.update(params)
        backend = default_backend_spec(
            cell_name="pontine_relay",
            params=merged_params,
            model_class="LIF_relay",
            brian2_equations=PONTINE_RELAY_EQUATIONS,
            nest_model="iaf_cond_alpha",
            spikingjelly_node="LIFNode",
            tau_default_ms=8.0,
            v_rest_default_mV=-62.0,
            v_thresh_default_mV=-45.0,
            v_reset_mV=-64.0,
            extra_nest_params={"relay_latency_ms": 1.0},
            extra_spikingjelly_kwargs={"relay_latency_ms": 1.0},
        )
        super().__init__(
            cell_name="pontine_relay",
            params=merged_params,
            backend_implementation=backend,
        )

    @property
    def latency_ms(self) -> float:
        return midpoint(self.params["latency_ms"], 1.0)

    def relay(self, cortical_spike_times_ms: Sequence[float]) -> tuple[float, ...]:
        return tuple(round(spike + self.latency_ms, 10) for spike in cortical_spike_times_ms)

    def simulate(
        self,
        *,
        duration_ms: float = 100.0,
        dt_ms: float = 0.1,
        cortical_spike_times_ms: Sequence[float] = (),
    ) -> SimulationResult:
        spikes = tuple(
            spike for spike in self.relay(cortical_spike_times_ms) if spike <= duration_ms
        )
        times = make_times(duration_ms, dt_ms)
        return SimulationResult(
            dt_ms=dt_ms,
            times_ms=times,
            voltage_mV=voltage_trace(
                times_ms=times,
                spike_times_ms=spikes,
                resting_mV=-62.0,
                dt_ms=dt_ms,
            ),
            spike_times_ms=spikes,
            labels={"relay": spikes},
        )


PontineNeuron = PontineRelayNeuron
