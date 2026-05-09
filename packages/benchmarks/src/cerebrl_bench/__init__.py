"""Benchmark suite for CerebRL."""

from __future__ import annotations

from .ablation import AblationResult, run_ablation_studies
from .neuromorphic import DeploymentPlan, NIRExport, deployment_plan, export_nir
from .predictions import FalsifiablePrediction, falsifiable_predictions
from .suite import (
    BASELINES,
    ENVIRONMENTS,
    BenchmarkMetric,
    BenchmarkReport,
    BenchmarkScore,
    DreamerComparison,
    EyeblinkAcquisition,
    compare_dreamer_dmc,
    eyeblink_acquisition_curve,
    run_benchmark_suite,
)
from .visualization_quality import (
    ViewerQuality,
    cerebellar_viewer_quality,
    loop_diagram_quality,
)

__all__ = [
    "BASELINES",
    "ENVIRONMENTS",
    "AblationResult",
    "BenchmarkMetric",
    "BenchmarkReport",
    "BenchmarkScore",
    "DeploymentPlan",
    "DreamerComparison",
    "EyeblinkAcquisition",
    "FalsifiablePrediction",
    "NIRExport",
    "ViewerQuality",
    "cerebellar_viewer_quality",
    "compare_dreamer_dmc",
    "deployment_plan",
    "export_nir",
    "eyeblink_acquisition_curve",
    "falsifiable_predictions",
    "loop_diagram_quality",
    "run_ablation_studies",
    "run_benchmark_suite",
]
