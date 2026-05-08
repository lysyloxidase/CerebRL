"""Biologically grounded cell models for the CerebRL cerebello-cortical loop."""

from __future__ import annotations

from .core import (
    BackendImplementation,
    BackendName,
    BioNeuron,
    Brian2Spec,
    NESTSpec,
    SimulationResult,
    SpikingJellySpec,
)
from .dcn import DCNInhibitoryNeuron, DCNNeuron, DCNProjectionNeuron
from .golgi import GolgiCell, GolgiNeuron
from .granule import GranuleCell, GranuleNeuron
from .inferior_olive import InferiorOliveCell, InferiorOliveNetwork, InferiorOliveNeuron
from .m1_pyramidal import M1Layer5PyramidalNeuron, M1PyramidalNeuron
from .params import CELL_PARAMS, CONNECTIVITY, MICROZONE_SCALE, PLASTICITY_RULES
from .pontine import PontineNeuron, PontineRelayNeuron
from .purkinje import PurkinjeCell, PurkinjeNeuron
from .receptors import RECEPTOR_PARAMS, ReceptorKinetics, receptor_current_pA
from .stellate_basket import MolecularLayerInterneuron, StellateBasketCell
from .thalamic_vl import ThalamicVLNeuron, VLThalamicRelayNeuron
from .ubc import UBCNeuron, UnipolarBrushCell

__all__ = [
    "CELL_PARAMS",
    "CONNECTIVITY",
    "MICROZONE_SCALE",
    "PLASTICITY_RULES",
    "RECEPTOR_PARAMS",
    "BackendImplementation",
    "BackendName",
    "BioNeuron",
    "Brian2Spec",
    "DCNInhibitoryNeuron",
    "DCNNeuron",
    "DCNProjectionNeuron",
    "GolgiCell",
    "GolgiNeuron",
    "GranuleCell",
    "GranuleNeuron",
    "InferiorOliveCell",
    "InferiorOliveNetwork",
    "InferiorOliveNeuron",
    "M1Layer5PyramidalNeuron",
    "M1PyramidalNeuron",
    "MolecularLayerInterneuron",
    "NESTSpec",
    "PontineNeuron",
    "PontineRelayNeuron",
    "PurkinjeCell",
    "PurkinjeNeuron",
    "ReceptorKinetics",
    "SimulationResult",
    "SpikingJellySpec",
    "StellateBasketCell",
    "ThalamicVLNeuron",
    "UBCNeuron",
    "UnipolarBrushCell",
    "VLThalamicRelayNeuron",
    "create_cell",
    "receptor_current_pA",
]


def create_cell(cell_type: str) -> BioNeuron:
    """Factory for all atlas cell types."""

    factories = {
        "purkinje": PurkinjeCell,
        "granule": GranuleCell,
        "golgi": GolgiCell,
        "stellate_basket": StellateBasketCell,
        "ubc": UnipolarBrushCell,
        "dcn_projection": DCNProjectionNeuron,
        "dcn_inhibitory": DCNInhibitoryNeuron,
        "inferior_olive": InferiorOliveNeuron,
        "m1_layer5_pyramidal": M1Layer5PyramidalNeuron,
        "pontine_relay": PontineRelayNeuron,
        "vl_thalamic_relay": VLThalamicRelayNeuron,
    }
    try:
        return factories[cell_type]()
    except KeyError as exc:
        available = ", ".join(sorted(factories))
        msg = f"Unknown cell_type {cell_type!r}. Available: {available}"
        raise ValueError(msg) from exc

