"""AMPA, NMDA, and GABA receptor kinetics used by CerebRL cell models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class ReceptorKinetics:
    name: str
    tau_rise_ms: float
    tau_decay_ms: float
    reversal_mV: float
    conductance_nS: float
    magnesium_sensitive: bool = False

    def conductance(self, t_ms: float) -> float:
        if t_ms < 0.0:
            return 0.0
        rise = exp(-t_ms / self.tau_rise_ms)
        decay = exp(-t_ms / self.tau_decay_ms)
        return self.conductance_nS * max(0.0, decay - rise)

    def current_pA(self, *, t_ms: float, voltage_mV: float, magnesium_mM: float = 1.0) -> float:
        block = 1.0
        if self.magnesium_sensitive:
            block = 1.0 / (1.0 + magnesium_mM * exp(-0.062 * voltage_mV) / 3.57)
        return self.conductance(t_ms) * block * (self.reversal_mV - voltage_mV)


RECEPTOR_PARAMS: Mapping[str, ReceptorKinetics] = {
    "AMPA": ReceptorKinetics(
        name="AMPA",
        tau_rise_ms=0.2,
        tau_decay_ms=2.0,
        reversal_mV=0.0,
        conductance_nS=0.7,
    ),
    "NMDA": ReceptorKinetics(
        name="NMDA",
        tau_rise_ms=2.0,
        tau_decay_ms=80.0,
        reversal_mV=0.0,
        conductance_nS=0.45,
        magnesium_sensitive=True,
    ),
    "GABA_A": ReceptorKinetics(
        name="GABA_A",
        tau_rise_ms=0.5,
        tau_decay_ms=10.0,
        reversal_mV=-75.0,
        conductance_nS=1.2,
    ),
    "GABA_B": ReceptorKinetics(
        name="GABA_B",
        tau_rise_ms=10.0,
        tau_decay_ms=120.0,
        reversal_mV=-95.0,
        conductance_nS=0.3,
    ),
}


def receptor_current_pA(
    receptor: str,
    *,
    t_ms: float,
    voltage_mV: float,
    magnesium_mM: float = 1.0,
) -> float:
    return RECEPTOR_PARAMS[receptor].current_pA(
        t_ms=t_ms,
        voltage_mV=voltage_mV,
        magnesium_mM=magnesium_mM,
    )

