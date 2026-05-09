from __future__ import annotations

from cerebrl_bio import CerebellarDrugScreen, HarmalineTremorModel, SCASimulator
from cerebrl_neurons import CELL_PARAMS
from cerebrl_rl.environments import mountain_car_env
from cerebrl_rl.hybrid_agent import CerebRLAgent
from cerebrl_rl.surrogate_train import SpikingPlasticitySystem

from cerebrl_bench import (
    cerebellar_viewer_quality,
    compare_dreamer_dmc,
    deployment_plan,
    eyeblink_acquisition_curve,
    falsifiable_predictions,
    loop_diagram_quality,
    run_ablation_studies,
    run_benchmark_suite,
)


def _agent() -> CerebRLAgent:
    return CerebRLAgent(mountain_car_env(), cerebellar_scale="minimal")


def test_mountain_car_cerebrl_reaches_ninety_percent_reward_thirty_percent_faster() -> None:
    report = run_benchmark_suite()
    cerebrl = report.score("cerebrl", "mountain-car")
    sac = report.score("sac", "mountain-car")

    assert cerebrl.episodes_to_90_reward <= sac.episodes_to_90_reward * 0.70
    assert cerebrl.sample_efficiency_vs_sac >= 0.30


def test_eyeblink_acquisition_matches_medina_mauk_time_course() -> None:
    curve = eyeblink_acquisition_curve()

    assert curve.max_abs_error <= 0.05
    assert curve.cr_probability[-1] >= 0.80


def test_reaching_trajectory_smoothness_improves_twofold_over_actor_only() -> None:
    report = run_benchmark_suite()
    actor_only = report.score("sac", "reaching")
    cerebrl = report.score("cerebrl", "reaching")

    assert actor_only.smoothness / cerebrl.smoothness >= 2.0


def test_sca6_cav21_drop_increases_pc_isi_cv_within_clinical_window() -> None:
    trajectory = SCASimulator().simulate_disease(_agent(), "SCA6", progression_steps=100)
    prediction = next(
        item
        for item in falsifiable_predictions()
        if item.name == "SCA6 PC irregularity"
    )

    assert trajectory.performance_drop > 0.25
    assert 0.20 <= prediction.measured_effect <= 0.50


def test_harmaline_block_emits_eight_to_twelve_hz_tremor_in_dcn() -> None:
    result = HarmalineTremorModel().simulate(_agent(), io_cav31_conductance=0.0)

    assert 8.0 <= result.dcn_frequency_hz <= 12.0
    assert result.synchronized_complex_spikes


def test_riluzole_partially_rescues_sca6_with_significant_effect() -> None:
    screen = CerebellarDrugScreen()
    riluzole = next(
        compound
        for compound in screen.known_compounds()
        if compound.name == "riluzole"
    )
    result = screen.screen(_agent(), "SCA6", [riluzole])[0]

    assert result.treated_performance > result.diseased_performance
    assert result.p_value < 0.05


def test_no_cf_ablation_stops_cerebellar_learning_with_ataxia_signature() -> None:
    plasticity = SpikingPlasticitySystem(cf_enabled=False)
    update = plasticity.apply_pf_pc(
        pf_activity=0.8,
        pc_activity=0.6,
        delta=1.0,
        cf_fired=True,
    )
    ablations = {item.name: item for item in run_ablation_studies()}

    assert update.weight_delta == 0.0
    assert update.ataxia_signature
    assert ablations["no_cf_rpe"].ataxia_signature


def test_3d_viewer_contract_renders_all_microzone_cell_types_at_thirty_fps() -> None:
    quality = cerebellar_viewer_quality()

    assert quality.target_fps >= 30
    assert quality.renders_all_microzone_cell_types


def test_loop_diagram_contract_animates_signal_propagation_latencies() -> None:
    quality = loop_diagram_quality()

    assert quality.target_fps >= 30
    assert quality.animated_latencies


def test_all_cell_type_parameters_have_literature_provenance() -> None:
    missing = tuple(
        name
        for name, params in CELL_PARAMS.items()
        if not isinstance(params.get("source"), str) or not params["source"]
    )

    assert missing == ()


def test_four_falsifiable_predictions_have_executable_thresholds() -> None:
    predictions = falsifiable_predictions()

    assert len(predictions) == 4
    assert all(item.passed and item.measured_effect >= item.threshold for item in predictions)


def test_dreamer_comparison_matches_on_at_least_five_dmc_tasks() -> None:
    comparison = compare_dreamer_dmc()

    assert len(comparison.tasks) == 12
    assert comparison.matched_tasks >= 5


def test_spinnaker2_plan_runs_one_hundred_microzones_in_realtime() -> None:
    plan = deployment_plan("spinnaker2", microzones=100)

    assert plan.realtime_factor >= 1.0
    assert plan.uses_eprop
    assert plan.supports_three_factor_learning
    assert plan.nir.microzones == 100
