"""Small deterministic Actor-Critic backbones for CerebRL integration tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

ActorKind = Literal["sac", "ppo", "td3"]
Vector = tuple[float, ...]


class SupportsSpace(Protocol):
    shape: tuple[int, ...]


class SupportsEnvSpec(Protocol):
    observation_space: SupportsSpace
    action_space: SupportsSpace


@dataclass(frozen=True)
class EnvironmentSpec:
    state_dim: int
    action_dim: int
    continuous_actions: bool = True


@dataclass(frozen=True)
class ActorCriticUpdate:
    td_error: float
    critic_loss: float
    actor_loss: float
    algorithm: ActorKind


class ClassicalActorCritic:
    """Dependency-light SAC/PPO/TD3-style facade.

    This is not a deep-learning implementation yet. It is a deterministic,
    strictly typed control surface with the methods the hybrid cerebellar agent
    needs: actor inference, TD-error computation, and a tiny online update.
    """

    def __init__(
        self,
        spec: EnvironmentSpec,
        *,
        algorithm: ActorKind = "sac",
        gamma: float = 0.97,
        learning_rate: float = 0.05,
    ) -> None:
        if spec.state_dim <= 0:
            msg = "state_dim must be positive"
            raise ValueError(msg)
        if spec.action_dim <= 0:
            msg = "action_dim must be positive"
            raise ValueError(msg)
        if not 0.0 <= gamma <= 1.0:
            msg = "gamma must be in [0, 1]"
            raise ValueError(msg)
        if learning_rate <= 0.0:
            msg = "learning_rate must be positive"
            raise ValueError(msg)
        self.spec = spec
        self.algorithm: ActorKind = algorithm
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.actor_bias = tuple(0.02 * (index + 1) for index in range(spec.action_dim))
        self.critic_weights = tuple(1.0 / (index + 1) for index in range(spec.state_dim))
        self.critic_bias = 0.0
        self.update_count = 0
        self.last_update: ActorCriticUpdate | None = None

    def actor(self, state: Sequence[float]) -> Vector:
        state_vector = _as_vector(state, expected_dim=self.spec.state_dim, name="state")
        state_mean = sum(state_vector) / float(len(state_vector))
        state_energy = sum(abs(value) for value in state_vector) / float(len(state_vector))
        entropy_bonus = 0.03 if self.algorithm == "sac" else 0.0
        td3_smoothing = -0.01 if self.algorithm == "td3" else 0.0
        return tuple(
            _clamp(
                self.actor_bias[index]
                + 0.25 * state_vector[index % len(state_vector)]
                + 0.1 * state_mean
                + entropy_bonus
                + td3_smoothing * state_energy,
                -1.0,
                1.0,
            )
            for index in range(self.spec.action_dim)
        )

    def value(self, state: Sequence[float]) -> float:
        state_vector = _as_vector(state, expected_dim=self.spec.state_dim, name="state")
        weighted = sum(
            state_vector[index] * self.critic_weights[index]
            for index in range(self.spec.state_dim)
        )
        return weighted / float(self.spec.state_dim) + self.critic_bias

    def compute_td_error(
        self,
        state: Sequence[float],
        reward: float,
        next_state: Sequence[float],
        done: bool,
    ) -> float:
        bootstrap = 0.0 if done else self.gamma * self.value(next_state)
        return reward + bootstrap - self.value(state)

    def update(
        self,
        state: Sequence[float],
        action: Sequence[float],
        reward: float,
        next_state: Sequence[float],
        done: bool,
    ) -> ActorCriticUpdate:
        td_error = self.compute_td_error(state, reward, next_state, done)
        action_vector = _as_vector(action, expected_dim=self.spec.action_dim, name="action")
        self.critic_bias += self.learning_rate * td_error
        action_pressure = sum(action_vector) / float(len(action_vector))
        self.actor_bias = tuple(
            _clamp(
                bias + self.learning_rate * 0.1 * td_error * (1.0 + action_pressure),
                -1.0,
                1.0,
            )
            for bias in self.actor_bias
        )
        self.update_count += 1
        update = ActorCriticUpdate(
            td_error=td_error,
            critic_loss=0.5 * td_error * td_error,
            actor_loss=-td_error * action_pressure,
            algorithm=self.algorithm,
        )
        self.last_update = update
        return update


def create_actor_critic(env: object | None, actor: ActorKind = "sac") -> ClassicalActorCritic:
    return ClassicalActorCritic(infer_environment_spec(env), algorithm=actor)


def infer_environment_spec(env: object | None) -> EnvironmentSpec:
    if env is None:
        return EnvironmentSpec(state_dim=4, action_dim=1)
    state_dim = _read_dim(env, "state_dim")
    action_dim = _read_dim(env, "action_dim")
    if state_dim is not None and action_dim is not None:
        return EnvironmentSpec(state_dim=state_dim, action_dim=action_dim)
    obs_dim = _space_dim(getattr(env, "observation_space", None))
    act_dim = _space_dim(getattr(env, "action_space", None))
    return EnvironmentSpec(state_dim=obs_dim or 4, action_dim=act_dim or 1)


def _read_dim(env: object, attr: str) -> int | None:
    value = getattr(env, attr, None)
    if isinstance(value, int) and value > 0:
        return value
    return None


def _space_dim(space: object | None) -> int | None:
    if space is None:
        return None
    shape = getattr(space, "shape", None)
    if isinstance(shape, tuple) and shape and isinstance(shape[0], int):
        return max(1, shape[0])
    n = getattr(space, "n", None)
    if isinstance(n, int) and n > 0:
        return 1
    return None


def _as_vector(values: Sequence[float], *, expected_dim: int, name: str) -> Vector:
    vector = tuple(float(value) for value in values)
    if not vector:
        msg = f"{name} must contain at least one element"
        raise ValueError(msg)
    if len(vector) == expected_dim:
        return vector
    return tuple(vector[index % len(vector)] for index in range(expected_dim))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
