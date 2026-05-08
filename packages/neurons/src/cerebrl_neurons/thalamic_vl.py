"""Ventrolateral thalamic relay neuron model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .core import (
    BioNeuron,
    SimulationResult,
    default_backend_spec,
    make_times,
    midpoint,
    regular_spikes,
    voltage_trace,
)
from .params import CELL_PARAMS

VL_RELAY_EQUATIONS = """
dv/dt = (-(v - E_L) + R_m * (I_dcn + I_T_burst)) / tau_m : 1 (unless refractory)
dI_dcn/dt = -I_dcn / tau_ampa : amp
dh_T/dt = (h_inf - h_T) / tau_h_T : 1
I_T_burst = g_T * h_T * (E_Ca - v) : amp
refractory_timer : second
"""


class VLThalamicRelayNeuron(BioNeuron):
    """VL thalamic relay from DCN output back into M1."""

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        merged_params = dict(CELL_PARAMS["vl_thalamic_relay"])
        if params is not None:
            merged_params.update(params)
        backend = default_backend_spec(
            cell_name="vl_thalamic_relay",
            params=merged_params,
            model_class="LIF_relay_with_burst",
            brian2_equations=VL_RELAY_EQUATIONS,
            nest_model="iaf_cond_alpha",
            spikingjelly_node="LIFNode",
            tau_default_ms=12.0,
            v_rest_default_mV=-64.0,
            v_thresh_default_mV=-46.0,
            v_reset_mV=-66.0,
            extra_nest_params={"T_type_burst": True, "relay_latency_ms": 5.5},
            extra_spikingjelly_kwargs={"burst_gate": True, "relay_latency_ms": 5.5},
        )
        super().__init__(
            cell_name="vl_thalamic_relay",
            params=merged_params,
            backend_implementation=backend,
        )

    @property
    def latency_ms(self) -> float:
        return midpoint(self.params["latency_ms"], 5.5)

    def relay_to_m1(self, dcn_spike_times_ms: Sequence[float]) -> tuple[float, ...]:
        return tuple(round(spike + self.latency_ms, 10) for spike in dcn_spike_times_ms)

    def simulate(
        self,
        *,
        duration_ms: float = 100.0,
        dt_ms: float = 0.1,
        dcn_spike_times_ms: Sequence[float] = (),
        rebound_burst: bool = False,
    ) -> SimulationResult:
        relayed = tuple(
            spike for spike in self.relay_to_m1(dcn_spike_times_ms) if spike <= duration_ms
        )
        burst = ()
        if rebound_burst and dcn_spike_times_ms:
            first = min(dcn_spike_times_ms) + self.latency_ms
            burst = regular_spikes(180.0, duration_ms, start_ms=first, end_ms=first + 18.0)
        spikes = tuple(sorted({*relayed, *burst}))
        times = make_times(duration_ms, dt_ms)
        return SimulationResult(
            dt_ms=dt_ms,
            times_ms=times,
            voltage_mV=voltage_trace(
                times_ms=times,
                spike_times_ms=spikes,
                resting_mV=-64.0,
                dt_ms=dt_ms,
            ),
            spike_times_ms=spikes,
            labels={"relay_to_m1": relayed, "burst": burst},
        )


ThalamicVLNeuron = VLThalamicRelayNeuron
