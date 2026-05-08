"""Purkinje cell model with simple-spike pacemaking and CF complex spikes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .core import (
    BioNeuron,
    SimulationResult,
    clamp,
    default_backend_spec,
    make_times,
    merge_spikes,
    midpoint,
    regular_spikes,
    voltage_trace,
)
from .params import CELL_PARAMS

PURKINJE_EGLIF_EQUATIONS = """
dv/dt = (
    -(v - E_L)
    + Delta_T * exp((v - V_T) / Delta_T)
    - w / g_L
    + I_syn / g_L
) / tau_m : 1 (unless refractory)
dw/dt = (a * (v - E_L) - w) / tau_w : 1
dCa/dt = -Ca / tau_Ca : 1
dI_syn/dt = -I_syn / tau_syn : 1
refractory_timer : second
"""


class PurkinjeCell(BioNeuron):
    """Adult Purkinje cell: tonic GABAergic output plus CF-driven complex spike."""

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        merged_params = dict(CELL_PARAMS["purkinje"])
        if params is not None:
            merged_params.update(params)
        backend = default_backend_spec(
            cell_name="purkinje",
            params=merged_params,
            model_class="EGLIF_multicompartment",
            brian2_equations=PURKINJE_EGLIF_EQUATIONS,
            nest_model="aeif_cond_alpha",
            spikingjelly_node="ParametricLIFNode",
            tau_default_ms=65.0,
            v_rest_default_mV=-65.0,
            v_thresh_default_mV=-55.0,
            v_reset_mV=-68.0,
            extra_nest_params={"Delta_T": 2.5, "tau_w": 80.0, "a": 4.0, "b": 0.08},
            extra_spikingjelly_kwargs={"detach_reset": False, "surrogate_scale": 4.0},
        )
        super().__init__(cell_name="purkinje", params=merged_params, backend_implementation=backend)

    def simulate(
        self,
        *,
        duration_ms: float = 1000.0,
        dt_ms: float = 0.1,
        parallel_fiber_drive: float = 0.0,
        climbing_fiber_times_ms: Sequence[float] = (),
    ) -> SimulationResult:
        base_rate = midpoint(self.params["spontaneous_simple_spike_Hz"], 50.0)
        max_rate = midpoint(self.params["max_driven_Hz"], 200.0)
        simple_rate = clamp(base_rate + 80.0 * parallel_fiber_drive, 30.0, max_rate)
        simple_spikes = regular_spikes(simple_rate, duration_ms)

        complex_bursts: list[float] = []
        complex_events: list[float] = []
        for cf_time in climbing_fiber_times_ms:
            if 0.0 < cf_time <= duration_ms:
                complex_events.append(round(cf_time, 10))
                complex_bursts.extend([cf_time, cf_time + 1.2, cf_time + 2.4])
        complex_spikes = tuple(round(t, 10) for t in complex_bursts if t <= duration_ms)
        spikes = merge_spikes(simple_spikes, complex_spikes)
        times = make_times(duration_ms, dt_ms)
        return SimulationResult(
            dt_ms=dt_ms,
            times_ms=times,
            voltage_mV=voltage_trace(
                times_ms=times,
                spike_times_ms=spikes,
                resting_mV=-62.0,
                dt_ms=dt_ms,
                oscillation_hz=7.0,
                oscillation_amplitude_mV=1.5,
            ),
            spike_times_ms=spikes,
            labels={
                "simple": simple_spikes,
                "complex": tuple(complex_events),
                "complex_burst_spikes": complex_spikes,
            },
        )


PurkinjeNeuron = PurkinjeCell

