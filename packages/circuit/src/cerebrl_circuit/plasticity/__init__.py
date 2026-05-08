"""Plasticity-site registry for CerebRL circuit modules."""

from __future__ import annotations

from dataclasses import dataclass

from cerebrl_neurons.params import PLASTICITY_RULES


@dataclass(frozen=True)
class PlasticitySite:
    name: str
    source: str
    target: str
    mechanism: str
    rl_role: str


def build_plasticity_sites() -> tuple[PlasticitySite, ...]:
    """Return the six biologically anchored plasticity sites in circuit order."""

    return (
        PlasticitySite(
            name="PF_PC_LTD_LTP",
            source="parallel_fiber",
            target="purkinje_cell",
            mechanism=str(PLASTICITY_RULES["PF_PC_LTD_LTP"]["description"]),
            rl_role="actor readout update with CF/RPE as third factor",
        ),
        PlasticitySite(
            name="PF_MLI",
            source="parallel_fiber",
            target="molecular_layer_interneuron",
            mechanism=str(PLASTICITY_RULES["PF_MLI"]["description"]),
            rl_role="critic/value update",
        ),
        PlasticitySite(
            name="MF_GrC",
            source="mossy_fiber",
            target="granule_cell",
            mechanism=str(PLASTICITY_RULES["MF_GrC"]["description"]),
            rl_role="adaptive sparse encoder",
        ),
        PlasticitySite(
            name="MF_DCN",
            source="mossy_fiber",
            target="dcn_projection_neuron",
            mechanism=str(PLASTICITY_RULES["MF_DCN"]["description"]),
            rl_role="nuclear consolidation",
        ),
        PlasticitySite(
            name="PC_DCN",
            source="purkinje_cell",
            target="dcn_projection_neuron",
            mechanism=str(PLASTICITY_RULES["PC_DCN"]["description"]),
            rl_role="learned disinhibition timing",
        ),
        PlasticitySite(
            name="IO_gap_junction",
            source="inferior_olive",
            target="inferior_olive",
            mechanism=str(PLASTICITY_RULES["IO_gap_junction"]["description"]),
            rl_role="error bandwidth and synchrony control",
        ),
    )


__all__ = ["PLASTICITY_RULES", "PlasticitySite", "build_plasticity_sites"]
