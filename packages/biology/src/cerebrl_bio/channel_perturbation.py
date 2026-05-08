"""Ion-channel and receptor perturbation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelPerturbation:
    cell: str
    target: str
    fractional_change: float
    mechanism: str


@dataclass(frozen=True)
class PerturbationEffect:
    perturbations: tuple[ChannelPerturbation, ...]
    conductance_burden: float
    synaptic_burden: float
    excitability_shift: float
    ltd_gain: float
    inhibition_gain: float

    @property
    def total_impairment(self) -> float:
        return _clamp(
            0.45 * self.conductance_burden
            + 0.3 * self.synaptic_burden
            + 0.15 * abs(self.excitability_shift)
            + 0.1 * self.ltd_gain,
            0.0,
            1.0,
        )


def summarize_perturbations(perturbations: Sequence[ChannelPerturbation]) -> PerturbationEffect:
    conductance = 0.0
    synaptic = 0.0
    excitability = 0.0
    ltd = 0.0
    inhibition = 0.0
    for item in perturbations:
        magnitude = abs(item.fractional_change)
        target = item.target.lower()
        if "cav" in target or "nav" in target or "kv" in target or "sk" in target:
            conductance += magnitude
        if "mgur" in target or "nmda" in target or "gaba" in target or "ip3" in target:
            synaptic += magnitude
        if item.fractional_change > 0.0 and ("excit" in item.mechanism.lower() or "kv" in target):
            excitability += item.fractional_change
        if item.fractional_change < 0.0 and ("cav" in target or "nav" in target):
            excitability += item.fractional_change
        if "ltd" in target.lower() or "pkc" in target.lower() or "mgur" in target:
            ltd += max(0.0, item.fractional_change)
        if "gaba" in target:
            inhibition += max(0.0, item.fractional_change)
    return PerturbationEffect(
        perturbations=tuple(perturbations),
        conductance_burden=_clamp(conductance, 0.0, 1.5),
        synaptic_burden=_clamp(synaptic, 0.0, 1.5),
        excitability_shift=_clamp(excitability, -1.0, 1.0),
        ltd_gain=_clamp(ltd, 0.0, 1.5),
        inhibition_gain=_clamp(inhibition, 0.0, 1.5),
    )


def perturbation_from_mapping(mapping: Mapping[str, object]) -> tuple[ChannelPerturbation, ...]:
    cell = str(mapping["cell"])
    primary = ChannelPerturbation(
        cell=cell,
        target=str(mapping["param"]),
        fractional_change=_as_float(mapping["change"]),
        mechanism=str(mapping.get("mechanism", "disease perturbation")),
    )
    if "param2" not in mapping:
        return (primary,)
    secondary = ChannelPerturbation(
        cell=cell,
        target=str(mapping["param2"]),
        fractional_change=_as_float(mapping["change2"]),
        mechanism=str(mapping.get("mechanism2", "secondary disease perturbation")),
    )
    return (primary, secondary)


def _as_float(value: object) -> float:
    if isinstance(value, (int, float, str, bytes)):
        return float(value)
    msg = f"Expected numeric perturbation value, got {type(value).__name__}"
    raise TypeError(msg)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
