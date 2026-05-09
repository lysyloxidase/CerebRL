"""In silico cerebellar pharmacology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cerebrl_rl.hybrid_agent import CerebRLAgent

from .sca_models import SCASimulator, SCAType

DrugMechanism = Literal[
    "persistent_Na_block",
    "SK_opener",
    "Kv_blocker",
    "carbonic_anhydrase_inhibitor",
    "CaV2.1_modulator",
    "mGluR1_NAM",
    "IP3R_antagonist",
    "GABA_A_agonist",
]


@dataclass(frozen=True)
class DrugCompound:
    name: str
    mechanism: DrugMechanism
    target: str
    potency: float
    concentration_uM: float


@dataclass(frozen=True)
class DrugScreenResult:
    compound: DrugCompound
    disease: SCAType
    baseline_performance: float
    diseased_performance: float
    treated_performance: float
    recovery_fraction: float
    p_value: float
    rank_score: float


class CerebellarDrugScreen:
    """Screen compounds against the spiking cerebellar model."""

    def __init__(self, simulator: SCASimulator | None = None) -> None:
        self.simulator = simulator or SCASimulator()

    def screen(
        self,
        agent: CerebRLAgent,
        disease: SCAType,
        compounds: list[DrugCompound],
    ) -> list[DrugScreenResult]:
        trajectory = self.simulator.simulate_disease(agent, disease)
        baseline = trajectory.baseline_performance
        diseased = trajectory.final_performance
        results = tuple(
            self._score_compound(
                compound,
                disease=disease,
                baseline=baseline,
                diseased=diseased,
            )
            for compound in compounds
        )
        return list(sorted(results, key=lambda item: item.rank_score, reverse=True))

    def known_compounds(self) -> tuple[DrugCompound, ...]:
        return (
            DrugCompound(
                name="riluzole",
                mechanism="persistent_Na_block",
                target="Purkinje persistent sodium",
                potency=0.45,
                concentration_uM=10.0,
            ),
            DrugCompound(
                name="chlorzoxazone",
                mechanism="SK_opener",
                target="Purkinje SK channel",
                potency=0.35,
                concentration_uM=30.0,
            ),
            DrugCompound(
                name="4-AP",
                mechanism="Kv_blocker",
                target="Purkinje Kv channel",
                potency=0.32,
                concentration_uM=5.0,
            ),
            DrugCompound(
                name="acetazolamide",
                mechanism="carbonic_anhydrase_inhibitor",
                target="episodic ataxia excitability",
                potency=0.25,
                concentration_uM=50.0,
            ),
        )

    def _score_compound(
        self,
        compound: DrugCompound,
        *,
        disease: SCAType,
        baseline: float,
        diseased: float,
    ) -> DrugScreenResult:
        disease_match = self._mechanism_match(compound, disease)
        rescue = min(0.95, compound.potency * disease_match)
        treated = min(baseline, diseased + (baseline - diseased) * rescue)
        recovery = 0.0 if baseline <= diseased else (treated - diseased) / (baseline - diseased)
        return DrugScreenResult(
            compound=compound,
            disease=disease,
            baseline_performance=baseline,
            diseased_performance=diseased,
            treated_performance=treated,
            recovery_fraction=recovery,
            p_value=max(0.001, 0.05 * (1.0 - recovery)),
            rank_score=recovery * compound.potency,
        )

    def _mechanism_match(self, compound: DrugCompound, disease: SCAType) -> float:
        if disease == "SCA6" and compound.mechanism in {
            "CaV2.1_modulator",
            "persistent_Na_block",
            "SK_opener",
        }:
            return 0.75 if compound.mechanism == "persistent_Na_block" else 1.0
        if disease == "SCA1" and compound.mechanism == "mGluR1_NAM":
            return 1.0
        if disease == "SCA2" and compound.mechanism == "IP3R_antagonist":
            return 1.0
        if disease == "SCA14" and compound.mechanism in {"SK_opener", "GABA_A_agonist"}:
            return 0.65
        return 0.25
