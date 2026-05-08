"""E-prop training for the M1 spiking module."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp

Vector = tuple[float, ...]


@dataclass(frozen=True)
class EPropUpdate:
    eligibility: Vector
    learning_signal: float
    weight_delta: Vector
    new_weights: Vector


@dataclass(frozen=True)
class EPropTrainingResult:
    episodes: int
    initial_score: float
    final_score: float
    policy_entropy: float
    weights: Vector


class EPropM1Trainer:
    """Biologically plausible online policy adaptation for M1."""

    def __init__(
        self,
        *,
        state_dim: int = 4,
        learning_rate: float = 0.04,
        tau_ms: float = 120.0,
    ) -> None:
        if state_dim <= 0:
            msg = "state_dim must be positive"
            raise ValueError(msg)
        if learning_rate <= 0.0:
            msg = "learning_rate must be positive"
            raise ValueError(msg)
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.tau_ms = tau_ms
        self.weights = tuple(-0.02 if index == 2 else 0.0 for index in range(state_dim))
        self.eligibility = tuple(0.0 for _ in range(state_dim))

    def policy(self, state: Sequence[float]) -> float:
        state_vector = _resize(tuple(float(value) for value in state), self.state_dim)
        drive = sum(state_vector[index] * self.weights[index] for index in range(self.state_dim))
        return 1.0 if drive >= 0.0 else -1.0

    def update(
        self,
        state: Sequence[float],
        *,
        learning_signal: float,
        dt_ms: float = 1.0,
    ) -> EPropUpdate:
        state_vector = _resize(tuple(float(value) for value in state), self.state_dim)
        decay = exp(-dt_ms / self.tau_ms)
        self.eligibility = tuple(
            self.eligibility[index] * decay + state_vector[index]
            for index in range(self.state_dim)
        )
        weight_delta = tuple(
            self.learning_rate * self.eligibility[index] * learning_signal
            for index in range(self.state_dim)
        )
        self.weights = tuple(
            self.weights[index] + weight_delta[index]
            for index in range(self.state_dim)
        )
        return EPropUpdate(
            eligibility=self.eligibility,
            learning_signal=learning_signal,
            weight_delta=weight_delta,
            new_weights=self.weights,
        )

    def train_cartpole_policy(self, *, episodes: int = 200) -> EPropTrainingResult:
        if episodes <= 0:
            msg = "episodes must be positive"
            raise ValueError(msg)
        initial_score = self._cartpole_score()
        for index in range(episodes):
            state = self._cartpole_state(index)
            target = 1.0 if state[2] >= 0.0 else -1.0
            prediction = self.policy(state)
            learning_signal = target - prediction
            if learning_signal == 0.0:
                learning_signal = 0.05 * target
            self.update(state, learning_signal=learning_signal)
        final_score = self._cartpole_score()
        return EPropTrainingResult(
            episodes=episodes,
            initial_score=initial_score,
            final_score=final_score,
            policy_entropy=self._policy_entropy(),
            weights=self.weights,
        )

    def _cartpole_score(self) -> float:
        states = tuple(self._cartpole_state(index) for index in range(32))
        correct = 0
        for state in states:
            target = 1.0 if state[2] >= 0.0 else -1.0
            if self.policy(state) == target:
                correct += 1
        return correct / float(len(states))

    def _policy_entropy(self) -> float:
        states = tuple(self._cartpole_state(index) for index in range(32))
        positive = sum(1 for state in states if self.policy(state) > 0.0)
        p = positive / float(len(states))
        return 2.0 * min(p, 1.0 - p)

    def _cartpole_state(self, index: int) -> Vector:
        centered = (index - 16.0) / 16.0
        return (0.1 * centered, 0.05 * centered, centered, -0.02 * centered)


def _resize(values: Vector, size: int) -> Vector:
    if not values:
        msg = "values must be non-empty"
        raise ValueError(msg)
    return tuple(values[index % len(values)] for index in range(size))
