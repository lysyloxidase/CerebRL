"""Spinocerebellar ataxia perturbation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from cerebrl_rl.hybrid_agent import CerebRLAgent

from .channel_perturbation import ChannelPerturbation, perturbation_from_mapping
from .phenotype_predictor import MotorPhenotype, predict_motor_phenotype

SCAType = Literal["SCA1", "SCA2", "SCA3", "SCA6", "SCA14"]


@dataclass(frozen=True)
class DiseaseStep:
    step: int
    perturbation_load: float
    phenotype: MotorPhenotype
    purkinje_survival: float


@dataclass(frozen=True)
class DiseaseTrajectory:
    sca_type: SCAType
    perturbations: tuple[ChannelPerturbation, ...]
    steps: tuple[DiseaseStep, ...]
    baseline_performance: float
    final_performance: float
    simulated_sara_score: float
    clinical_severity_proxy: float

    @property
    def performance_drop(self) -> float:
        return self.baseline_performance - self.final_performance

    @property
    def final_phenotype(self) -> MotorPhenotype:
        return self.steps[-1].phenotype


class SCASimulator:
    """Simulate spinocerebellar ataxia phenotypes by perturbing cell parameters."""

    SCA_PERTURBATIONS: ClassVar[dict[SCAType, dict[str, object]]] = {
        "SCA1": {
            "cell": "purkinje",
            "param": "dendritic_area",
            "change": -0.3,
            "param2": "mGluR1_gain",
            "change2": 1.0,
            "mechanism": "Purkinje dendritic atrophy",
            "mechanism2": "mGluR1 hyperactivity",
        },
        "SCA2": {
            "cell": "purkinje",
            "param": "IP3R1_threshold",
            "change": -0.5,
            "mechanism": "IP3R1 hypersensitivity",
        },
        "SCA3": {
            "cell": "purkinje",
            "param": "survival_probability",
            "change": -0.05,
            "mechanism": "progressive Purkinje degeneration",
        },
        "SCA6": {
            "cell": "purkinje",
            "param": "CaV2.1_conductance",
            "change": -0.4,
            "mechanism": "P/Q-type calcium channel dysfunction",
        },
        "SCA14": {
            "cell": "purkinje",
            "param": "PKC_LTD_gain",
            "change": 0.5,
            "mechanism": "constitutive PKC-dependent LTD",
        },
    }

    _SEVERITY_GAIN: ClassVar[dict[SCAType, float]] = {
        "SCA1": 0.50,
        "SCA2": 0.58,
        "SCA3": 0.46,
        "SCA6": 0.62,
        "SCA14": 0.34,
    }

    def simulate_disease(
        self,
        agent: CerebRLAgent,
        sca_type: SCAType,
        progression_steps: int = 100,
    ) -> DiseaseTrajectory:
        if progression_steps <= 0:
            msg = "progression_steps must be positive"
            raise ValueError(msg)
        perturbations = perturbation_from_mapping(self.SCA_PERTURBATIONS[sca_type])
        baseline = self._baseline_performance(agent)
        steps: list[DiseaseStep] = []
        for index in range(1, progression_steps + 1):
            progress = index / float(progression_steps)
            impairment = self._impairment(sca_type, progress)
            ltd_burden = self._ltd_burden(sca_type, progress)
            survival = self._purkinje_survival(sca_type, progress)
            phenotype = predict_motor_phenotype(
                baseline_performance=baseline,
                impairment=impairment,
                ltd_burden=ltd_burden,
                oscillation_hz=0.0,
                gait_irregularity=min(1.0, impairment * 0.7),
            )
            steps.append(
                DiseaseStep(
                    step=index,
                    perturbation_load=progress,
                    phenotype=phenotype,
                    purkinje_survival=survival,
                )
            )
        final = steps[-1].phenotype
        return DiseaseTrajectory(
            sca_type=sca_type,
            perturbations=perturbations,
            steps=tuple(steps),
            baseline_performance=baseline,
            final_performance=final.performance,
            simulated_sara_score=final.sara_score,
            clinical_severity_proxy=self._SEVERITY_GAIN[sca_type] * 40.0,
        )

    def rank_sca_severity(self, agent: CerebRLAgent) -> tuple[DiseaseTrajectory, ...]:
        trajectories = tuple(
            self.simulate_disease(agent, sca_type, progression_steps=100)
            for sca_type in ("SCA6", "SCA1", "SCA14")
        )
        return tuple(
            sorted(trajectories, key=lambda item: item.final_phenotype.severity, reverse=True)
        )

    def _baseline_performance(self, agent: CerebRLAgent) -> float:
        action = agent.act((0.2,) * agent.actor_critic.spec.state_dim)
        correction = sum(abs(value) for value in action) / float(len(action))
        return max(0.75, min(1.0, 0.9 + correction * 0.05))

    def _impairment(self, sca_type: SCAType, progress: float) -> float:
        if sca_type == "SCA3":
            return min(0.9, self._SEVERITY_GAIN[sca_type] * progress * 1.15)
        return min(0.9, self._SEVERITY_GAIN[sca_type] * progress)

    def _ltd_burden(self, sca_type: SCAType, progress: float) -> float:
        if sca_type == "SCA1":
            return min(1.0, progress * 1.2)
        if sca_type == "SCA2":
            return min(1.0, progress * 1.1)
        if sca_type == "SCA14":
            return min(1.0, progress * 0.65)
        return min(1.0, progress * 0.35)

    def _purkinje_survival(self, sca_type: SCAType, progress: float) -> float:
        if sca_type == "SCA3":
            return max(0.0, 1.0 - 0.05 * progress * 100.0)
        return max(0.25, 1.0 - self._impairment(sca_type, progress) * 0.25)
