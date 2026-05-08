"""Inferior olive neuron model with STOs and connexin-36 gap junction coupling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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

IO_HH_EQUATIONS = """
dv/dt = (
    -I_Na - I_K - I_leak - I_h - I_CaT + I_gap + I_syn
) / C_m : 1
dm/dt = (m_inf - m) / tau_m_Na : 1
dh/dt = (h_inf - h) / tau_h_Na : 1
dn/dt = (n_inf - n) / tau_n_K : 1
dr_CaT/dt = (r_inf - r_CaT) / tau_r_CaT : 1
dq_h/dt = (q_inf - q_h) / tau_q_h : 1
I_gap : amp
I_syn : amp
refractory_timer : second
"""


class InferiorOliveNeuron(BioNeuron):
    """Olivary source of climbing fibers and RPE-like teaching signals."""

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        merged_params = dict(CELL_PARAMS["inferior_olive"])
        if params is not None:
            merged_params.update(params)
        backend = default_backend_spec(
            cell_name="inferior_olive",
            params=merged_params,
            model_class="HH_with_Ih_CaT_gap_junctions",
            brian2_equations=IO_HH_EQUATIONS,
            nest_model="hh_cond_exp_traub",
            spikingjelly_node="MultiStepLIFNode",
            tau_default_ms=20.0,
            v_rest_default_mV=-65.0,
            v_thresh_default_mV=-50.0,
            v_reset_mV=-68.0,
            extra_nest_params={"gap_junction": "connexin_36", "STO_Hz": 8.0, "C_m": 220.0},
            extra_spikingjelly_kwargs={"gap_junction": True, "sto_frequency_hz": 8.0},
        )
        super().__init__(
            cell_name="inferior_olive",
            params=merged_params,
            backend_implementation=backend,
        )

    @property
    def sto_frequency_hz(self) -> float:
        return midpoint(self.params["STO_Hz"], 8.0)

    def gap_junction_current_pA(
        self,
        *,
        own_voltage_mV: float,
        neighbor_voltage_mV: float,
        coupling_nS: float = 0.6,
    ) -> float:
        return coupling_nS * (neighbor_voltage_mV - own_voltage_mV)

    def simulate(self, *, duration_ms: float = 1000.0, dt_ms: float = 0.1) -> SimulationResult:
        firing_rate = midpoint(self.params["firing_Hz"], 1.0)
        spikes = regular_spikes(firing_rate, duration_ms)
        times = make_times(duration_ms, dt_ms)
        return SimulationResult(
            dt_ms=dt_ms,
            times_ms=times,
            voltage_mV=voltage_trace(
                times_ms=times,
                spike_times_ms=spikes,
                resting_mV=-65.0,
                dt_ms=dt_ms,
                oscillation_hz=self.sto_frequency_hz,
                oscillation_amplitude_mV=4.0,
            ),
            spike_times_ms=spikes,
            labels={"climbing_fiber": spikes, "sto_frequency_hz": (self.sto_frequency_hz,)},
        )


@dataclass(frozen=True)
class InferiorOliveNetwork:
    """Small IO coupling helper for microzone synchrony experiments."""

    neurons: Sequence[InferiorOliveNeuron]
    coupling_nS: float = 0.6

    def gap_currents_pA(self, voltages_mV: Sequence[float]) -> tuple[float, ...]:
        if len(voltages_mV) != len(self.neurons):
            msg = "voltages_mV must match neuron count"
            raise ValueError(msg)
        currents: list[float] = []
        for index, voltage in enumerate(voltages_mV):
            total = 0.0
            for neighbor_index, neighbor_voltage in enumerate(voltages_mV):
                if neighbor_index != index:
                    total += self.coupling_nS * (neighbor_voltage - voltage)
            currents.append(round(total, 6))
        return tuple(currents)


InferiorOliveCell = InferiorOliveNeuron

