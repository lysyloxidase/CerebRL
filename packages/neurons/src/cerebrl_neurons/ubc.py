"""Unipolar brush cell model with slow mossy-fiber amplification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .core import (
    BioNeuron,
    SimulationResult,
    default_backend_spec,
    make_times,
    merge_spikes,
    midpoint,
    regular_spikes,
    voltage_trace,
)
from .params import CELL_PARAMS

UBC_EQUATIONS = """
dv/dt = (-(v - E_L) + R_m * (I_fast + I_slow)) / tau_m : 1 (unless refractory)
dI_fast/dt = -I_fast / tau_ampa : amp
dI_slow/dt = -I_slow / tau_mglur : amp
refractory_timer : second
"""


class UnipolarBrushCell(BioNeuron):
    """UBC model with ON/OFF slow metabotropic mossy-fiber responses."""

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        merged_params = dict(CELL_PARAMS["ubc"])
        if params is not None:
            merged_params.update(params)
        backend = default_backend_spec(
            cell_name="ubc",
            params=merged_params,
            model_class="LIF_slow_synaptic",
            brian2_equations=UBC_EQUATIONS,
            nest_model="iaf_cond_exp",
            spikingjelly_node="LIFNode",
            tau_default_ms=35.0,
            v_rest_default_mV=-63.0,
            v_thresh_default_mV=-43.0,
            v_reset_mV=-66.0,
            extra_nest_params={"tau_syn_ex": 25.0, "tau_mglur": 180.0},
            extra_spikingjelly_kwargs={"slow_synapse_tau": 180.0},
        )
        super().__init__(cell_name="ubc", params=merged_params, backend_implementation=backend)

    def simulate(
        self,
        *,
        duration_ms: float = 1000.0,
        dt_ms: float = 0.1,
        mossy_fiber_event_times_ms: Sequence[float] = (),
    ) -> SimulationResult:
        baseline = midpoint(self.params["spontaneous_Hz"], 15.0)
        spontaneous = regular_spikes(baseline, duration_ms)
        slow_response: list[float] = []
        for event_ms in mossy_fiber_event_times_ms:
            slow_response.extend(
                regular_spikes(
                    50.0,
                    duration_ms,
                    start_ms=event_ms + 20.0,
                    end_ms=min(duration_ms, event_ms + 220.0),
                )
            )
        slow_spikes = tuple(sorted({round(spike, 10) for spike in slow_response}))
        spikes = merge_spikes(spontaneous, slow_spikes)
        times = make_times(duration_ms, dt_ms)
        return SimulationResult(
            dt_ms=dt_ms,
            times_ms=times,
            voltage_mV=voltage_trace(
                times_ms=times,
                spike_times_ms=spikes,
                resting_mV=-63.0,
                dt_ms=dt_ms,
                oscillation_hz=baseline,
                oscillation_amplitude_mV=0.8,
            ),
            spike_times_ms=spikes,
            labels={"spontaneous": spontaneous, "slow_mossy_response": slow_spikes},
        )


UBCNeuron = UnipolarBrushCell

