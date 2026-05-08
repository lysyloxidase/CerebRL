"""Dreamer-style imagined rollouts using the spiking cerebellar forward model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .actor_critic import ClassicalActorCritic
from .cerebellar_forward_model import CerebellarForwardModel

Vector = tuple[float, ...]


@dataclass(frozen=True)
class ImaginedStep:
    state: Vector
    action: Vector
    predicted_next_state: Vector
    value: float


@dataclass(frozen=True)
class DreamerRollout:
    steps: tuple[ImaginedStep, ...]
    bootstrap_value: float

    @property
    def final_state(self) -> Vector:
        if not self.steps:
            return ()
        return self.steps[-1].predicted_next_state


class CerebRLDreamer:
    """Replace Dreamer-V3's RSSM with the spiking cerebellar forward model."""

    def __init__(
        self,
        actor_critic: ClassicalActorCritic,
        forward_model: CerebellarForwardModel,
    ) -> None:
        self.actor_critic = actor_critic
        self.forward_model = forward_model

    def imagined_rollout(self, state: Sequence[float], *, n_steps: int = 5) -> DreamerRollout:
        if n_steps <= 0:
            msg = "n_steps must be positive"
            raise ValueError(msg)
        current_state = tuple(float(value) for value in state)
        steps: list[ImaginedStep] = []
        for _ in range(n_steps):
            action = self.actor_critic.actor(current_state)
            predicted_next = self.forward_model.predict_next_state(current_state, action)
            value = self.actor_critic.value(predicted_next)
            steps.append(
                ImaginedStep(
                    state=current_state,
                    action=action,
                    predicted_next_state=predicted_next,
                    value=value,
                )
            )
            current_state = predicted_next
        return DreamerRollout(
            steps=tuple(steps),
            bootstrap_value=self.actor_critic.value(current_state),
        )
