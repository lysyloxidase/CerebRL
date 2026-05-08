"""Cerebellar toxicology and harmaline tremor models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cerebrl_rl.hybrid_agent import CerebRLAgent

from .phenotype_predictor import MotorPhenotype, predict_motor_phenotype

ToxicantName = Literal["Hg2+", "Pb2+", "Mn2+", "ethanol", "phenytoin"]


@dataclass(frozen=True)
class ToxicantExposure:
    toxicant: ToxicantName
    concentration: float
    unit: str
    exposure_duration_h: float


@dataclass(frozen=True)
class ToxicologyResult:
    exposure: ToxicantExposure
    phenotype: MotorPhenotype
    channel_effects: tuple[str, ...]
    motor_deficit_severity: float


@dataclass(frozen=True)
class HarmalineTremorResult:
    io_cav31_conductance: float
    io_frequency_hz: float
    dcn_frequency_hz: float
    tremor_frequency_hz: float
    synchronized_complex_spikes: bool
    phenotype: MotorPhenotype


class CerebellarToxicologyScreen:
    """Model neurotoxicant effects on cerebellar function."""

    def run_exposure(
        self,
        agent: CerebRLAgent,
        exposure: ToxicantExposure,
    ) -> ToxicologyResult:
        del agent
        severity = self._severity(exposure)
        phenotype = predict_motor_phenotype(
            impairment=severity,
            ltd_burden=self._ltd_burden(exposure),
            oscillation_hz=0.0,
            gait_irregularity=self._gait_irregularity(exposure),
        )
        return ToxicologyResult(
            exposure=exposure,
            phenotype=phenotype,
            channel_effects=self._channel_effects(exposure.toxicant),
            motor_deficit_severity=phenotype.severity,
        )

    def dose_response(
        self,
        agent: CerebRLAgent,
        toxicant: ToxicantName,
        concentrations: tuple[float, ...],
        *,
        unit: str = "uM",
        exposure_duration_h: float = 24.0,
    ) -> tuple[ToxicologyResult, ...]:
        return tuple(
            self.run_exposure(
                agent,
                ToxicantExposure(
                    toxicant=toxicant,
                    concentration=concentration,
                    unit=unit,
                    exposure_duration_h=exposure_duration_h,
                ),
            )
            for concentration in concentrations
        )

    def _severity(self, exposure: ToxicantExposure) -> float:
        concentration_scale = exposure.concentration / (exposure.concentration + 10.0)
        duration_scale = min(1.0, exposure.exposure_duration_h / 24.0)
        potency = {
            "Hg2+": 0.95,
            "Pb2+": 0.65,
            "Mn2+": 0.55,
            "ethanol": 0.70,
            "phenytoin": 0.50,
        }[exposure.toxicant]
        if exposure.toxicant == "ethanol":
            concentration_scale = min(1.0, exposure.concentration / 50.0)
        return min(1.0, potency * concentration_scale * duration_scale)

    def _ltd_burden(self, exposure: ToxicantExposure) -> float:
        if exposure.toxicant in {"Pb2+", "ethanol"}:
            return min(1.0, self._severity(exposure) * 0.5)
        return min(1.0, self._severity(exposure) * 0.25)

    def _gait_irregularity(self, exposure: ToxicantExposure) -> float:
        if exposure.toxicant == "ethanol":
            return min(1.0, 0.45 + 0.45 * self._severity(exposure))
        return min(1.0, self._severity(exposure) * 0.65)

    def _channel_effects(self, toxicant: ToxicantName) -> tuple[str, ...]:
        return {
            "Hg2+": ("Purkinje CaV2.1 block", "Bergmann glia stress"),
            "Pb2+": ("granule NMDA receptor block", "MF-GrC plasticity impairment"),
            "Mn2+": ("basal ganglia load", "cerebellar oxidative stress"),
            "ethanol": ("GABA_A enhancement", "NMDA block"),
            "phenytoin": ("Purkinje Nav block", "simple spike suppression"),
        }[toxicant]


class HarmalineTremorModel:
    """Reproduce harmaline-induced tremor via IO synchronization."""

    def simulate(
        self,
        agent: CerebRLAgent,
        *,
        io_cav31_conductance: float = 0.0,
    ) -> HarmalineTremorResult:
        del agent
        blocked = io_cav31_conductance <= 0.05
        frequency = 10.0 if blocked else 6.0
        power = 0.85 if blocked else 0.15
        phenotype = predict_motor_phenotype(
            impairment=0.35 if blocked else 0.08,
            oscillation_hz=frequency,
            oscillation_power=power,
            gait_irregularity=0.35 if blocked else 0.05,
        )
        return HarmalineTremorResult(
            io_cav31_conductance=io_cav31_conductance,
            io_frequency_hz=frequency,
            dcn_frequency_hz=frequency,
            tremor_frequency_hz=frequency,
            synchronized_complex_spikes=blocked,
            phenotype=phenotype,
        )
