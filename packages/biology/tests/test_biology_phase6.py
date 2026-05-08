from __future__ import annotations

from cerebrl_rl.environments import mountain_car_env
from cerebrl_rl.hybrid_agent import CerebRLAgent

from cerebrl_bio import (
    CerebellarDrugScreen,
    CerebellarToxicologyScreen,
    DrugCompound,
    HarmalineTremorModel,
    SCASimulator,
    ToxicantExposure,
)


def _agent() -> CerebRLAgent:
    return CerebRLAgent(mountain_car_env(), cerebellar_scale="minimal")


def test_sca6_cav21_reduction_causes_measurable_reaching_performance_drop() -> None:
    trajectory = SCASimulator().simulate_disease(_agent(), "SCA6", progression_steps=100)

    assert trajectory.performance_drop > 0.25
    assert trajectory.final_phenotype.dysmetria > 0.45
    assert trajectory.perturbations[0].target == "CaV2.1_conductance"


def test_sca1_mglur1_gain_drives_excessive_ltd_pf_pc_weights_near_zero() -> None:
    trajectory = SCASimulator().simulate_disease(_agent(), "SCA1", progression_steps=100)

    assert trajectory.final_phenotype.pf_pc_weight_mean <= 0.05
    assert trajectory.final_phenotype.hypermetria > 0.5


def test_drug_screen_riluzole_partially_rescues_sca6_phenotype() -> None:
    screen = CerebellarDrugScreen()
    riluzole = DrugCompound(
        name="riluzole",
        mechanism="persistent_Na_block",
        target="Purkinje persistent sodium",
        potency=0.45,
        concentration_uM=10.0,
    )
    result = screen.screen(_agent(), "SCA6", [riluzole])[0]

    assert result.compound.name == "riluzole"
    assert result.treated_performance > result.diseased_performance
    assert result.treated_performance < result.baseline_performance
    assert 0.0 < result.recovery_fraction < 1.0


def test_ethanol_fifty_millimolar_produces_gaba_nmda_ataxic_gait_pattern() -> None:
    screen = CerebellarToxicologyScreen()
    result = screen.run_exposure(
        _agent(),
        ToxicantExposure(
            toxicant="ethanol",
            concentration=50.0,
            unit="mM",
            exposure_duration_h=4.0,
        ),
    )

    assert "GABA_A enhancement" in result.channel_effects
    assert "NMDA block" in result.channel_effects
    assert result.phenotype.ataxic_gait
    assert result.phenotype.gait_irregularity >= 0.45


def test_harmaline_io_cav31_block_emits_eight_to_twelve_hz_tremor() -> None:
    result = HarmalineTremorModel().simulate(_agent(), io_cav31_conductance=0.0)

    assert result.synchronized_complex_spikes
    assert 8.0 <= result.tremor_frequency_hz <= 12.0
    assert result.dcn_frequency_hz == result.tremor_frequency_hz


def test_mercury_dose_response_is_progressive_from_one_to_one_hundred_micromolar() -> None:
    results = CerebellarToxicologyScreen().dose_response(
        _agent(),
        "Hg2+",
        (1.0, 10.0, 100.0),
        unit="uM",
    )
    severities = tuple(result.motor_deficit_severity for result in results)

    assert severities[0] < severities[1] < severities[2]


def test_sca_ranking_matches_sca6_greater_than_sca1_greater_than_sca14() -> None:
    ranking = SCASimulator().rank_sca_severity(_agent())
    names = tuple(item.sca_type for item in ranking)

    assert names == ("SCA6", "SCA1", "SCA14")

