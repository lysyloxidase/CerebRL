"""Motor-deficit phenotype prediction for in silico cerebellar perturbations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotorPhenotype:
    performance: float
    performance_drop: float
    dysmetria: float
    tremor_frequency_hz: float
    tremor_power: float
    hypermetria: float
    gait_irregularity: float
    pf_pc_weight_mean: float
    sara_score: float
    ataxic_gait: bool

    @property
    def severity(self) -> float:
        return min(1.0, self.sara_score / 40.0)


def predict_motor_phenotype(
    *,
    baseline_performance: float = 1.0,
    impairment: float,
    ltd_burden: float = 0.0,
    oscillation_hz: float = 0.0,
    oscillation_power: float = 0.0,
    gait_irregularity: float = 0.0,
) -> MotorPhenotype:
    impairment = _clamp(impairment, 0.0, 1.0)
    ltd_burden = _clamp(ltd_burden, 0.0, 1.0)
    performance = _clamp(baseline_performance * (1.0 - impairment), 0.0, 1.0)
    performance_drop = _clamp(baseline_performance - performance, 0.0, 1.0)
    dysmetria = _clamp(0.15 + 0.75 * impairment, 0.0, 1.0)
    hypermetria = _clamp(0.1 + 0.55 * impairment + 0.2 * ltd_burden, 0.0, 1.0)
    pf_pc_weight_mean = _clamp(1.0 - max(impairment * 0.8, ltd_burden), 0.0, 1.0)
    sara_score = _clamp(
        40.0 * (0.55 * impairment + 0.2 * dysmetria + 0.25 * gait_irregularity),
        0.0,
        40.0,
    )
    return MotorPhenotype(
        performance=performance,
        performance_drop=performance_drop,
        dysmetria=dysmetria,
        tremor_frequency_hz=oscillation_hz,
        tremor_power=_clamp(oscillation_power, 0.0, 1.0),
        hypermetria=hypermetria,
        gait_irregularity=_clamp(gait_irregularity, 0.0, 1.0),
        pf_pc_weight_mean=pf_pc_weight_mean,
        sara_score=sara_score,
        ataxic_gait=performance_drop >= 0.25 or gait_irregularity >= 0.35,
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
