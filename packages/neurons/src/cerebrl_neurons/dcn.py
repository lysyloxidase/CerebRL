"""Deep cerebellar nuclei projection and inhibitory neuron models."""

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

DCN_REBOUND_EQUATIONS = """
dv/dt = (
    -(v - E_L)
    + R_m * (I_mf - I_pc + I_NaP + I_CaT)
    - w
) / tau_m : 1 (unless refractory)
dw/dt = (a * (v - E_L) - w) / tau_w : 1
dh_CaT/dt = (h_inf - h_CaT) / tau_h_CaT : 1
I_CaT = g_CaT * h_CaT * (E_Ca - v) : amp
I_NaP = g_NaP * (E_Na - v) : amp
refractory_timer : second
"""

DCN_INHIBITORY_EQUATIONS = """
dv/dt = (-(v - E_L) + R_m * (I_mf - I_pc)) / tau_m : 1 (unless refractory)
dI_nucleo_olivary/dt = -I_nucleo_olivary / tau_gaba : amp
refractory_timer : second
"""


class DCNProjectionNeuron(BioNeuron):
    """Glutamatergic DCN projection neuron with post-inhibitory rebound."""

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        merged_params = dict(CELL_PARAMS["dcn_projection"])
        if params is not None:
            merged_params.update(params)
        backend = default_backend_spec(
            cell_name="dcn_projection",
            params=merged_params,
            model_class="AdEx_with_rebound",
            brian2_equations=DCN_REBOUND_EQUATIONS,
            nest_model="aeif_cond_alpha",
            spikingjelly_node="ParametricLIFNode",
            tau_default_ms=7.0,
            v_rest_default_mV=-58.0,
            v_thresh_default_mV=-45.0,
            v_reset_mV=-62.0,
            extra_nest_params={"g_CaT": 1.8, "g_NaP": 1.2, "rebound_firing": True},
            extra_spikingjelly_kwargs={"rebound_gate": True},
        )
        super().__init__(
            cell_name="dcn_projection",
            params=merged_params,
            backend_implementation=backend,
        )

    def simulate(
        self,
        *,
        duration_ms: float = 500.0,
        dt_ms: float = 0.1,
        pc_inhibition_windows: Sequence[tuple[float, float]] = (),
    ) -> SimulationResult:
        baseline_rate = midpoint(self.params["spontaneous_Hz"], 50.0)
        baseline = regular_spikes(baseline_rate, duration_ms)
        baseline = tuple(
            spike
            for spike in baseline
            if not any(start <= spike <= end for start, end in pc_inhibition_windows)
        )
        rebound_trains: list[float] = []
        for _, release_ms in pc_inhibition_windows:
            rebound_trains.extend(
                regular_spikes(
                    midpoint(self.params["max_Hz"], 160.0),
                    duration_ms,
                    start_ms=release_ms,
                    end_ms=min(duration_ms, release_ms + 35.0),
                )
            )
        rebound = tuple(sorted({round(spike, 10) for spike in rebound_trains}))
        spikes = merge_spikes(baseline, rebound)
        times = make_times(duration_ms, dt_ms)
        return SimulationResult(
            dt_ms=dt_ms,
            times_ms=times,
            voltage_mV=voltage_trace(
                times_ms=times,
                spike_times_ms=spikes,
                resting_mV=-58.0,
                dt_ms=dt_ms,
                inhibition_windows=pc_inhibition_windows,
                hyperpolarized_mV=-80.0,
            ),
            spike_times_ms=spikes,
            labels={"tonic": baseline, "rebound": rebound},
        )

    def simulate_pc_release_rebound(
        self,
        *,
        inhibition_start_ms: float = 100.0,
        inhibition_end_ms: float = 250.0,
        duration_ms: float = 400.0,
        dt_ms: float = 0.1,
    ) -> SimulationResult:
        return self.simulate(
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            pc_inhibition_windows=((inhibition_start_ms, inhibition_end_ms),),
        )


class DCNInhibitoryNeuron(BioNeuron):
    """GABAergic nucleo-olivary DCN neuron."""

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        merged_params = dict(CELL_PARAMS["dcn_inhibitory"])
        if params is not None:
            merged_params.update(params)
        backend = default_backend_spec(
            cell_name="dcn_inhibitory",
            params=merged_params,
            model_class="LIF_nucleo_olivary",
            brian2_equations=DCN_INHIBITORY_EQUATIONS,
            nest_model="iaf_cond_alpha",
            spikingjelly_node="LIFNode",
            tau_default_ms=10.0,
            v_rest_default_mV=-60.0,
            v_thresh_default_mV=-47.0,
            v_reset_mV=-64.0,
            extra_nest_params={"target": "inferior_olive", "tau_syn_in": 12.0},
            extra_spikingjelly_kwargs={"nucleo_olivary_feedback": True},
        )
        super().__init__(
            cell_name="dcn_inhibitory",
            params=merged_params,
            backend_implementation=backend,
        )

    def simulate(self, *, duration_ms: float = 1000.0, dt_ms: float = 0.1) -> SimulationResult:
        spikes = regular_spikes(25.0, duration_ms)
        times = make_times(duration_ms, dt_ms)
        return SimulationResult(
            dt_ms=dt_ms,
            times_ms=times,
            voltage_mV=voltage_trace(
                times_ms=times,
                spike_times_ms=spikes,
                resting_mV=-60.0,
                dt_ms=dt_ms,
            ),
            spike_times_ms=spikes,
            labels={"nucleo_olivary": spikes},
        )


DCNNeuron = DCNProjectionNeuron

