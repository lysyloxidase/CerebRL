from __future__ import annotations

from cerebrl_rl import ClimbingFiberRPE


def test_positive_delta_maps_to_climbing_fiber_ltd() -> None:
    rpe = ClimbingFiberRPE(gamma=0.9)
    result = rpe.compute(state=(0.1, 0.2), next_state=(0.2, 0.3), reward=1.0)

    assert result.gated_delta > 0.0
    assert result.climbing_fiber_rate_hz > 1.0
    assert result.plasticity_direction == "LTD"
    assert result.pf_pc_weight_delta < 0.0


def test_negative_delta_maps_to_climbing_fiber_silence_ltp() -> None:
    rpe = ClimbingFiberRPE(gamma=0.9)
    result = rpe.compute(state=(0.8, 0.7), next_state=(0.4, 0.3), reward=-1.0)

    assert result.gated_delta < 0.0
    assert result.climbing_fiber_rate_hz == 0.0
    assert result.plasticity_direction == "LTP"
    assert result.pf_pc_weight_delta > 0.0


def test_dcn_feedback_attenuates_delta_gain() -> None:
    rpe = ClimbingFiberRPE(gamma=0.9)
    open_loop = rpe.compute(state=(0.1,), next_state=(0.2,), reward=1.0)
    gated = rpe.compute(
        state=(0.1,),
        next_state=(0.2,),
        reward=1.0,
        dcn_inhibitory_rate_hz=100.0,
    )

    assert gated.feedback_gain < open_loop.feedback_gain
    assert gated.gated_delta < open_loop.gated_delta
    assert gated.climbing_fiber_rate_hz < open_loop.climbing_fiber_rate_hz

