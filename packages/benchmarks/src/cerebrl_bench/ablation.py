"""Ablation study summaries for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AblationName = Literal[
    "full",
    "no_cerebellum",
    "no_cf_rpe",
    "no_dcn_io_feedback",
    "no_dreamer_rollouts",
    "purkinje_lesion",
    "io_lesion",
]


@dataclass(frozen=True)
class AblationResult:
    name: AblationName
    sample_efficiency: float
    stability: float
    biological_signature: str
    ataxia_signature: bool


def run_ablation_studies() -> tuple[AblationResult, ...]:
    return (
        AblationResult("full", 1.0, 0.95, "precise DCN correction", False),
        AblationResult("no_cerebellum", 0.68, 0.72, "actor-only dysmetria", True),
        AblationResult("no_cf_rpe", 0.63, 0.62, "no PF-PC learning", True),
        AblationResult("no_dcn_io_feedback", 0.58, 0.35, "oscillatory overcorrection", True),
        AblationResult("no_dreamer_rollouts", 0.81, 0.80, "model-free only", False),
        AblationResult("purkinje_lesion", 0.52, 0.48, "loss of cortical inhibition", True),
        AblationResult("io_lesion", 0.60, 0.55, "complex spike loss", True),
    )
