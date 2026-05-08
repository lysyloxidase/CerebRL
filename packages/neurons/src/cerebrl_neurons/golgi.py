"""Golgi cell model with intrinsic AdEx pacemaking."""

from __future__ import annotations

from collections.abc import Mapping

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

GOLGI_ADEX_EQUATIONS = """
dv/dt = (
    -(v - E_L)
    + Delta_T * exp((v - V_T) / Delta_T)
    - w / g_L
    + I_pacemaker / g_L
) / tau_m : 1 (unless refractory)
dw/dt = (a * (v - E_L) - w) / tau_w : 1
dI_pacemaker/dt = (I_tonic - I_pacemaker) / tau_pace : amp
refractory_timer : second
"""


class GolgiCell(BioNeuron):
    """Pacemaking Golgi interneuron for glomerular feedback inhibition."""

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        merged_params = dict(CELL_PARAMS["golgi"])
        if params is not None:
            merged_params.update(params)
        backend = default_backend_spec(
            cell_name="golgi",
            params=merged_params,
            model_class="AdEx_pacemaker",
            brian2_equations=GOLGI_ADEX_EQUATIONS,
            nest_model="aeif_cond_alpha",
            spikingjelly_node="ParametricLIFNode",
            tau_default_ms=18.0,
            v_rest_default_mV=-60.0,
            v_thresh_default_mV=-48.0,
            v_reset_mV=-62.0,
            extra_nest_params={"I_e": 85.0, "Delta_T": 2.0, "tau_w": 120.0},
            extra_spikingjelly_kwargs={"learnable_tau": True},
        )
        super().__init__(cell_name="golgi", params=merged_params, backend_implementation=backend)

    def simulate(self, *, duration_ms: float = 2000.0, dt_ms: float = 0.1) -> SimulationResult:
        rate = midpoint(self.params["spontaneous_Hz"], 5.0)
        spikes = regular_spikes(rate, duration_ms)
        times = make_times(duration_ms, dt_ms)
        return SimulationResult(
            dt_ms=dt_ms,
            times_ms=times,
            voltage_mV=voltage_trace(
                times_ms=times,
                spike_times_ms=spikes,
                resting_mV=-60.0,
                dt_ms=dt_ms,
                oscillation_hz=rate,
                oscillation_amplitude_mV=2.0,
            ),
            spike_times_ms=spikes,
            labels={"pacemaker": spikes},
        )


GolgiNeuron = GolgiCell

