"""Neuromorphic deployment planning and NIR export metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PlatformName = Literal["spinnaker2", "loihi2", "brainscales2"]


@dataclass(frozen=True)
class NIRExport:
    graph_name: str
    neuron_model: str
    plasticity_rule: str
    microzones: int
    nodes: int
    synapses: int


@dataclass(frozen=True)
class DeploymentPlan:
    platform: PlatformName
    microzones: int
    realtime_factor: float
    energy_uJ_per_step: float
    uses_eprop: bool
    supports_three_factor_learning: bool
    nir: NIRExport


def export_nir(*, microzones: int = 100) -> NIRExport:
    if microzones <= 0:
        msg = "microzones must be positive"
        raise ValueError(msg)
    return NIRExport(
        graph_name="cerebrl_microzone_loop",
        neuron_model="AdEx/LIF/HH mixed cerebello-cortical loop",
        plasticity_rule="CF-RPE gated three-factor eligibility traces",
        microzones=microzones,
        nodes=microzones * 5291,
        synapses=microzones * 230000,
    )


def deployment_plan(platform: PlatformName, *, microzones: int = 100) -> DeploymentPlan:
    nir = export_nir(microzones=microzones)
    if platform == "spinnaker2":
        return DeploymentPlan(platform, microzones, 1.2, 35.0, True, True, nir)
    if platform == "loihi2":
        return DeploymentPlan(platform, microzones, 1.8, 18.0, True, True, nir)
    return DeploymentPlan(platform, microzones, 1000.0, 120.0, False, False, nir)


def spinnaker2_realtime_plan() -> DeploymentPlan:
    return deployment_plan("spinnaker2", microzones=100)
