from __future__ import annotations

from cerebrl_rl.cf_rpe import CFRPEComputer
from cerebrl_rl.environments import full_hybrid_mountain_car_gain
from cerebrl_rl.eprop import EPropM1Trainer
from cerebrl_rl.surrogate_train import (
    SpikingPlasticitySystem,
    SurrogateTrainer,
    make_linear_dynamics_transitions,
)


def test_surrogate_training_forward_model_mse_decreases_over_episodes() -> None:
    transitions = make_linear_dynamics_transitions(count=24)
    trainer = SurrogateTrainer()
    result = trainer.train(transitions, episodes=1000)

    assert result.final_mse < result.initial_mse
    assert result.final_mse < 0.1
    assert result.mse_history[-1] < result.mse_history[0]
    assert result.prediction_gain > 0.9


def test_three_factor_pf_pc_weight_delta_changes_sign_with_cf_delta() -> None:
    plasticity = SpikingPlasticitySystem()
    ltd = plasticity.apply_pf_pc(
        pf_activity=1.0,
        pc_activity=1.0,
        delta=1.0,
        cf_fired=True,
    )
    ltp = plasticity.apply_pf_pc(
        pf_activity=1.0,
        pc_activity=1.0,
        delta=-1.0,
        cf_fired=True,
    )

    assert ltd.weight_delta < 0.0
    assert ltp.weight_delta > 0.0


def test_cf_rpe_positive_delta_routes_to_io_complex_spike_and_ltd() -> None:
    router = CFRPEComputer(alpha=0.8)
    routed = router.compute_and_route(
        state=(0.1, 0.0),
        next_state=(0.2, 0.1),
        predicted_next_state=(0.1, 0.0),
        reward=1.0,
        sto_phase_ms=42.0,
    )
    plasticity = SpikingPlasticitySystem()
    pf_pc = plasticity.apply_pf_pc(
        pf_activity=1.0,
        pc_activity=1.0,
        delta=routed.io_current,
        cf_fired=routed.cf_fired,
    )

    assert routed.cf_fired
    assert routed.complex_spike
    assert routed.io_rate_hz > 1.0
    assert routed.plasticity_direction == "LTD"
    assert pf_pc.weight_delta < 0.0


def test_dcn_to_io_feedback_reduces_io_rate_as_training_progresses() -> None:
    router = CFRPEComputer(alpha=0.8)
    rates = tuple(
        router.compute_and_route(
            state=(0.1, 0.0),
            next_state=(0.2, 0.1),
            predicted_next_state=(0.1, 0.0),
            reward=1.0,
            dcn_inhibitory_rate_hz=feedback,
            sto_phase_ms=42.0,
        ).io_rate_hz
        for feedback in (0.0, 20.0, 60.0)
    )

    assert rates[0] > rates[1] > rates[2]


def test_eprop_m1_learns_non_trivial_cartpole_policy() -> None:
    trainer = EPropM1Trainer()
    result = trainer.train_cartpole_policy(episodes=200)

    assert result.final_score > result.initial_score
    assert result.final_score >= 0.75
    assert result.policy_entropy > 0.0


def test_ablation_removing_cf_stops_cerebellar_learning_ataxia_signature() -> None:
    plasticity = SpikingPlasticitySystem(cf_enabled=False)
    update = plasticity.apply_pf_pc(
        pf_activity=1.0,
        pc_activity=1.0,
        delta=1.0,
        cf_fired=True,
    )

    assert update.weight_delta == 0.0
    assert update.ataxia_signature


def test_full_hybrid_beats_actor_only_mountain_car_by_thirty_percent() -> None:
    result = full_hybrid_mountain_car_gain()

    assert result.solved
    assert result.sample_efficiency_gain >= 0.30

