"""Motor cortex layer 5 pyramidal neuron model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

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

MovementState = Literal["quiet", "movement"]

M1_ADEX_EQUATIONS = """
dv/dt = (
    -(v - E_L)
    + Delta_T * exp((v - V_T) / Delta_T)
    - w / g_L
    + I_corticospinal / g_L
) / tau_m : 1 (unless refractory)
dw/dt = (a * (v - E_L) - w) / tau_w : 1
apical_drive : amp
refractory_timer : second
"""


class M1Layer5PyramidalNeuron(BioNeuron):
    """Layer 5 corticospinal/corticopontine M1 neuron."""

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        merged_params = dict(CELL_PARAMS["m1_layer5_pyramidal"])
        if params is not None:
            merged_params.update(params)
        backend = default_backend_spec(
            cell_name="m1_layer5_pyramidal",
            params=merged_params,
            model_class="AdEx_or_Izhikevich",
            brian2_equations=M1_ADEX_EQUATIONS,
            nest_model="aeif_cond_alpha",
            spikingjelly_node="ParametricLIFNode",
            tau_default_ms=17.5,
            v_rest_default_mV=-52.9,
            v_thresh_default_mV=-42.0,
            v_reset_mV=-58.0,
            extra_nest_params={"Delta_T": 2.0, "tau_w": 140.0, "apical_drive": True},
            extra_spikingjelly_kwargs={"corticopontine_projection": True},
        )
        super().__init__(
            cell_name="m1_layer5_pyramidal",
            params=merged_params,
            backend_implementation=backend,
        )

    def simulate(
        self,
        *,
        duration_ms: float = 10000.0,
        dt_ms: float = 0.1,
        state: MovementState = "quiet",
    ) -> SimulationResult:
        rate_key = "spontaneous_Hz_movement" if state == "movement" else "spontaneous_Hz_quiet"
        rate = midpoint(self.params[rate_key], 5.6)
        spikes = regular_spikes(rate, duration_ms)
        times = make_times(duration_ms, dt_ms)
        return SimulationResult(
            dt_ms=dt_ms,
            times_ms=times,
            voltage_mV=voltage_trace(
                times_ms=times,
                spike_times_ms=spikes,
                resting_mV=-52.9,
                dt_ms=dt_ms,
                oscillation_hz=min(rate, 12.0),
                oscillation_amplitude_mV=1.0,
            ),
            spike_times_ms=spikes,
            labels={state: spikes},
        )


M1PyramidalNeuron = M1Layer5PyramidalNeuron

