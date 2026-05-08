from __future__ import annotations

from cerebrl_rl.environments import benchmark_sample_efficiency, cartpole_env, mountain_car_env
from cerebrl_rl.hybrid_agent import CerebRLAgent


def test_hybrid_agent_outputs_action_with_cerebellar_correction() -> None:
    agent = CerebRLAgent(mountain_car_env(), cerebellar_scale="minimal")
    output = agent.act_with_trace((0.1, -0.2))

    assert output.corrected_action != output.raw_action
    assert output.cerebellar_correction != (0.0,)
    assert output.predicted_next_state != (0.1, -0.2)


def test_positive_rpe_injection_decreases_pf_pc_weights_ltd() -> None:
    agent = CerebRLAgent(mountain_car_env(), cerebellar_scale="minimal")
    loop_output = agent.inject_rpe(1.0)

    assert loop_output.rpe > 0.0
    assert loop_output.pf_pc_plasticity == "LTD"
    assert loop_output.pf_pc_weight_delta < 0.0
    assert loop_output.complex_spike_count > 0


def test_dreamer_rollout_uses_cerebellar_forward_model_for_next_state() -> None:
    agent = CerebRLAgent(cartpole_env(), actor="ppo", cerebellar_scale="minimal")
    rollout = agent.imagined_rollout((0.1, 0.0, -0.1, 0.2), n_steps=5)

    assert len(rollout.steps) == 5
    assert rollout.steps[0].predicted_next_state != rollout.steps[0].state
    assert rollout.final_state == rollout.steps[-1].predicted_next_state


def test_disabling_cerebellum_drops_to_classical_actor_output() -> None:
    agent = CerebRLAgent(mountain_car_env(), cerebellar_scale="minimal")
    state = (0.2, -0.1)
    classical = agent.classical_action(state)
    disabled = agent.act_with_trace(state, cerebellum_enabled=False)
    enabled = agent.act_with_trace(state, cerebellum_enabled=True)

    assert disabled.corrected_action == classical
    assert disabled.cerebellar_correction == (0.0,)
    assert enabled.corrected_action != classical


def test_mountain_car_cerebrl_solves_within_two_x_sac_budget() -> None:
    result = benchmark_sample_efficiency("mountain-car", agent="cerebrl")
    standard = benchmark_sample_efficiency("mountain-car", agent="classical")

    assert result.solved
    assert result.sample_budget <= 2 * standard.sample_budget


def test_cartpole_cerebrl_matches_ppo_within_fifty_k_steps() -> None:
    result = benchmark_sample_efficiency("cartpole", agent="cerebrl")

    assert result.solved
    assert result.sample_budget <= 50000
