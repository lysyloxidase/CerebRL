"""Cerebellar granule cell model."""

from __future__ import annotations

from collections.abc import Mapping

from .core import (
    BioNeuron,
    SimulationResult,
    clamp,
    default_backend_spec,
    make_times,
    midpoint,
    regular_spikes,
    voltage_trace,
)
from .params import CELL_PARAMS

GRANULE_LIF_EQUATIONS = """
dv/dt = (-(v - E_L) + R_m * I_syn) / tau_m : 1 (unless refractory)
dI_syn/dt = -I_syn / tau_syn : amp
g_nmda : siemens
refractory_timer : second
"""


class GranuleCell(BioNeuron):
    """Tiny high-resistance granule neuron with sparse rest activity and fast bursts."""

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        merged_params = dict(CELL_PARAMS["granule"])
        if params is not None:
            merged_params.update(params)
        backend = default_backend_spec(
            cell_name="granule",
            params=merged_params,
            model_class="LIF_or_Izhikevich",
            brian2_equations=GRANULE_LIF_EQUATIONS,
            nest_model="iaf_psc_exp",
            spikingjelly_node="LIFNode",
            tau_default_ms=3.15,
            v_rest_default_mV=-64.0,
            v_thresh_default_mV=-35.0,
            v_reset_mV=-64.0,
            extra_nest_params={"C_m": 3.0, "t_ref": 1.0, "Rin_GOhm": 2.2},
            extra_spikingjelly_kwargs={"step_mode": "s", "decay_input": True},
        )
        super().__init__(cell_name="granule", params=merged_params, backend_implementation=backend)

    @property
    def membrane_capacitance_pF(self) -> float:
        return midpoint(self.params["Cm_pF"], 3.0)

    @property
    def input_resistance_GOhm(self) -> float:
        return midpoint(self.params["Rin_GOhm"], 2.2)

    def simulate(
        self,
        *,
        duration_ms: float = 1000.0,
        dt_ms: float = 0.05,
        mossy_fiber_current_pA: float = 0.0,
    ) -> SimulationResult:
        if mossy_fiber_current_pA <= 0.0:
            rate = midpoint(self.params["spontaneous_Hz"], 0.0)
        else:
            rate = clamp(200.0 + 40.0 * mossy_fiber_current_pA, 0.0, 1000.0)
        spikes = regular_spikes(rate, duration_ms)
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
            labels={"burst": spikes if mossy_fiber_current_pA > 0.0 else ()},
        )


GranuleNeuron = GranuleCell

