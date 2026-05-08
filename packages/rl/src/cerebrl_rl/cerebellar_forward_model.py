"""Spiking cerebello-cortical loop as a forward model and action refiner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cerebrl_circuit import CerebelloCorticalLoop, LoopOutput

Vector = tuple[float, ...]


@dataclass(frozen=True)
class ForwardModelOutput:
    raw_action: Vector
    corrected_action: Vector
    correction: Vector
    predicted_next_state: Vector
    loop_output: LoopOutput


class CerebellarForwardModel:
    """Wrap the closed cerebello-cortical loop as a world model."""

    def __init__(self, *, scale: str = "standard") -> None:
        self.loop = CerebelloCorticalLoop(microzone_scale=scale)

    def refine_action(
        self,
        state: Sequence[float],
        action: Sequence[float],
        *,
        reward_prediction_error: float = 0.0,
        cerebellum_enabled: bool = True,
    ) -> ForwardModelOutput:
        raw_action = tuple(float(value) for value in action)
        loop_output = self.loop.step(
            state,
            raw_action,
            reward_prediction_error,
            cerebellum_enabled=cerebellum_enabled,
        )
        return ForwardModelOutput(
            raw_action=raw_action,
            corrected_action=loop_output.corrected_action,
            correction=loop_output.cerebellar_correction,
            predicted_next_state=loop_output.predicted_next_state,
            loop_output=loop_output,
        )

    def predict_next_state(
        self,
        state: Sequence[float],
        action: Sequence[float],
        *,
        reward_prediction_error: float = 0.0,
    ) -> Vector:
        return self.refine_action(
            state,
            action,
            reward_prediction_error=reward_prediction_error,
        ).predicted_next_state

    def inject_rpe(self, delta: float) -> LoopOutput:
        zero_state = (0.0, 0.0, 0.0, 0.0)
        zero_action = (0.0,)
        return self.loop.step(zero_state, zero_action, reward=delta)
