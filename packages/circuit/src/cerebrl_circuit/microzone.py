"""Single cerebellar microzone assembly."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cerebrl_neurons import MICROZONE_SCALE

from .dcn_module import DCNModule, DCNOutput
from .granular_layer import GranularLayer, GranularLayerOutput
from .inferior_olive_module import InferiorOliveModule, InferiorOliveOutput
from .molecular_layer import MolecularLayer, MolecularLayerOutput
from .plasticity import PlasticitySite, build_plasticity_sites
from .purkinje_layer import PurkinjeLayer, PurkinjeLayerOutput


@dataclass(frozen=True)
class MicrozoneScale:
    granule: int
    golgi: int
    purkinje: int
    mli: int
    dcn_projection: int
    dcn_inhibitory: int
    inferior_olive: int
    m1_l5: int
    pontine: int
    vl: int

    @property
    def total(self) -> int:
        return (
            self.granule
            + self.golgi
            + self.purkinje
            + self.mli
            + self.dcn_projection
            + self.dcn_inhibitory
            + self.inferior_olive
            + self.m1_l5
            + self.pontine
            + self.vl
        )


@dataclass(frozen=True)
class MicrozoneStepResult:
    granular: GranularLayerOutput
    molecular: MolecularLayerOutput
    purkinje: PurkinjeLayerOutput
    dcn: DCNOutput
    inferior_olive: InferiorOliveOutput


def minimal_microzone() -> MicrozoneScale:
    return MicrozoneScale(
        granule=5000,
        golgi=50,
        purkinje=32,
        mli=20,
        dcn_projection=10,
        dcn_inhibitory=5,
        inferior_olive=4,
        m1_l5=100,
        pontine=50,
        vl=20,
    )


def scale_from_atlas(scale: str) -> MicrozoneScale:
    try:
        config = MICROZONE_SCALE[scale]
    except KeyError as exc:
        available = ", ".join(sorted(MICROZONE_SCALE))
        msg = f"Unknown microzone scale {scale!r}. Available: {available}"
        raise ValueError(msg) from exc
    return MicrozoneScale(
        granule=int(config["GrC"]),
        golgi=int(config["GoC"]),
        purkinje=int(config["PC"]),
        mli=int(config["MLI"]),
        dcn_projection=int(config["DCN_proj"]),
        dcn_inhibitory=int(config["DCN_inh"]),
        inferior_olive=int(config["IO"]),
        m1_l5=int(config["M1_L5"]),
        pontine=int(config["pontine"]),
        vl=int(config["VL"]),
    )


class CerebellarMicrozone:
    """Complete microzone: granular + molecular + Purkinje + DCN + IO.

    Information flow:
    1. Mossy fibers carry state and efference copy into the granular layer.
    2. GrC sparse expansion recodes 4-5 MF samples into a high-dimensional PF code.
    3. PFs excite PC dendrites and MLIs.
    4. PCs, the sole cortical output, inhibit DCN.
    5. DCN projection neurons emit the cerebellar motor command.
    6. IO climbing fibers broadcast an error/RPE signal that drives complex spikes.
    """

    def __init__(self, scale: str = "standard") -> None:
        self.scale_name = scale
        self.scale = scale_from_atlas(scale)
        self.granular = GranularLayer(self.scale.granule, self.scale.golgi)
        self.molecular = MolecularLayer(self.scale.mli)
        self.purkinje = PurkinjeLayer(self.scale.purkinje)
        self.dcn = DCNModule(self.scale.dcn_projection, self.scale.dcn_inhibitory)
        self.io = InferiorOliveModule(self.scale.inferior_olive)
        self.plasticity_sites = build_plasticity_sites()

    def step(
        self,
        mossy_fiber_rates_hz: Sequence[float],
        *,
        reward_prediction_error: float = 0.0,
        climbing_fiber_events_ms: Sequence[float] = (),
        duration_ms: float = 1000.0,
    ) -> MicrozoneStepResult:
        granular = self.granular.encode(mossy_fiber_rates_hz)
        preliminary_io = self.io.broadcast_error(
            reward_prediction_error,
            dcn_inhibitory_rate_hz=0.0,
            duration_ms=duration_ms,
        )
        cf_spikes = (
            tuple(float(event) for event in climbing_fiber_events_ms)
            if climbing_fiber_events_ms
            else preliminary_io.climbing_fiber_spikes_ms
        )
        molecular = self.molecular.process_parallel_fibers(
            granular,
            climbing_fiber_error=reward_prediction_error,
        )
        purkinje = self.purkinje.readout(
            granular,
            molecular,
            climbing_fiber_spikes_ms=cf_spikes,
            duration_ms=duration_ms,
        )
        dcn = self.dcn.integrate(
            purkinje,
            mf_collateral_drive=granular.parallel_fiber_drive / 0.05,
            cf_collateral_drive=min(1.0, abs(reward_prediction_error)),
            duration_ms=duration_ms,
        )
        gated_io = self.io.broadcast_error(
            reward_prediction_error,
            dcn_inhibitory_rate_hz=dcn.io_feedback_rate_hz,
            duration_ms=duration_ms,
        )
        return MicrozoneStepResult(
            granular=granular,
            molecular=molecular,
            purkinje=purkinje,
            dcn=dcn,
            inferior_olive=gated_io,
        )

    def plasticity_site_names(self) -> tuple[str, ...]:
        return tuple(site.name for site in self.plasticity_sites)

    def plasticity_site_registry(self) -> tuple[PlasticitySite, ...]:
        return self.plasticity_sites
