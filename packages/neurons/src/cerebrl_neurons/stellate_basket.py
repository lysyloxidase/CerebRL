"""Molecular layer interneuron models: stellate and basket cells."""

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

Subtype = Literal["stellate", "basket"]

MLI_EQUATIONS = """
dv/dt = (-(v - E_L) + R_m * (I_pf - I_adapt)) / tau_m : 1 (unless refractory)
dI_pf/dt = -I_pf / tau_ampa : amp
dI_adapt/dt = -I_adapt / tau_adapt : amp
refractory_timer : second
"""


class StellateBasketCell(BioNeuron):
    """Fast GABAergic MLI implementing PF-driven feedforward inhibition."""

    def __init__(
        self,
        *,
        subtype: Subtype = "stellate",
        params: Mapping[str, object] | None = None,
    ) -> None:
        merged_params = dict(CELL_PARAMS["stellate_basket"])
        merged_params["subtype"] = subtype
        if params is not None:
            merged_params.update(params)
        backend = default_backend_spec(
            cell_name=f"{subtype}_basket_mli" if subtype == "basket" else "stellate_mli",
            params=merged_params,
            model_class="LIF_or_AdEx",
            brian2_equations=MLI_EQUATIONS,
            nest_model="iaf_cond_alpha",
            spikingjelly_node="LIFNode",
            tau_default_ms=12.0,
            v_rest_default_mV=-58.0,
            v_thresh_default_mV=-45.0,
            v_reset_mV=-62.0,
            extra_nest_params={"tau_syn_ex": 2.0, "tau_syn_in": 6.0},
            extra_spikingjelly_kwargs={"decay_input": True},
        )
        super().__init__(
            cell_name="stellate_basket",
            params=merged_params,
            backend_implementation=backend,
        )
        self.subtype = subtype

    def simulate(
        self,
        *,
        duration_ms: float = 1000.0,
        dt_ms: float = 0.1,
        parallel_fiber_drive: float = 0.0,
    ) -> SimulationResult:
        baseline = midpoint(self.params["spontaneous_Hz"], 18.0)
        rate = min(80.0, baseline + 45.0 * parallel_fiber_drive)
        spikes = regular_spikes(rate, duration_ms)
        times = make_times(duration_ms, dt_ms)
        return SimulationResult(
            dt_ms=dt_ms,
            times_ms=times,
            voltage_mV=voltage_trace(
                times_ms=times,
                spike_times_ms=spikes,
                resting_mV=-58.0,
                dt_ms=dt_ms,
            ),
            spike_times_ms=spikes,
            labels={self.subtype: spikes},
        )


MolecularLayerInterneuron = StellateBasketCell

