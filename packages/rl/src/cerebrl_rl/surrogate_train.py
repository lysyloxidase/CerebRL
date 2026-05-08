"""Surrogate-gradient pretraining and online three-factor plasticity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp
from typing import Literal

from .actor_critic import ClassicalActorCritic, EnvironmentSpec
from .cerebellar_forward_model import CerebellarForwardModel

PlasticitySiteName = Literal[
    "PF_PC_LTD_LTP",
    "PF_MLI",
    "MF_GrC",
    "MF_DCN",
    "PC_DCN",
    "IO_gap_junction",
]
Vector = tuple[float, ...]


@dataclass(frozen=True)
class TrainingTransition:
    state: Vector
    action: Vector
    next_state: Vector
    reward: float = 0.0
    done: bool = False


@dataclass(frozen=True)
class SurrogateTrainingConfig:
    timesteps: int = 24
    lambda_pred: float = 1.0
    lambda_rl: float = 0.2
    lambda_sparse: float = 0.05
    lambda_bio: float = 0.01
    learning_rate: float = 0.01
    target_granule_sparsity: float = 0.03


@dataclass(frozen=True)
class SurrogateTrainingResult:
    episodes: int
    initial_mse: float
    final_mse: float
    loss_history: tuple[float, ...]
    mse_history: tuple[float, ...]
    sparse_loss_history: tuple[float, ...]
    prediction_gain: float


@dataclass(frozen=True)
class PlasticityUpdate:
    site: PlasticitySiteName
    eligibility: float
    delta: float
    weight_delta: float
    new_weight: float
    cf_fired: bool
    ataxia_signature: bool


class EligibilityTrace:
    """Low-pass pre/post eligibility trace with 100-200 ms biological window."""

    def __init__(self, *, tau_ms: float = 150.0) -> None:
        if tau_ms <= 0.0:
            msg = "tau_ms must be positive"
            raise ValueError(msg)
        self.tau_ms = tau_ms
        self.value = 0.0

    def update(self, *, pre: float, post: float, dt_ms: float = 1.0) -> float:
        decay = exp(-dt_ms / self.tau_ms)
        self.value = self.value * decay + pre * post
        return self.value

    def reset(self) -> None:
        self.value = 0.0


class ThreeFactorPlasticityRule:
    """Generic eligibility-trace rule: delta_w = eta * e_ij * delta."""

    def __init__(
        self,
        *,
        site: PlasticitySiteName,
        learning_rate: float,
        initial_weight: float = 1.0,
        tau_ms: float = 150.0,
        positive_delta_sign: float = 1.0,
    ) -> None:
        if learning_rate <= 0.0:
            msg = "learning_rate must be positive"
            raise ValueError(msg)
        self.site: PlasticitySiteName = site
        self.learning_rate = learning_rate
        self.weight = initial_weight
        self.positive_delta_sign = positive_delta_sign
        self.trace = EligibilityTrace(tau_ms=tau_ms)

    def apply(
        self,
        *,
        pre: float,
        post: float,
        delta: float,
        cf_fired: bool = True,
        dt_ms: float = 1.0,
    ) -> PlasticityUpdate:
        eligibility = self.trace.update(pre=pre, post=post, dt_ms=dt_ms)
        if not cf_fired:
            weight_delta = 0.0
        else:
            weight_delta = self.learning_rate * eligibility * delta * self.positive_delta_sign
        self.weight += weight_delta
        return PlasticityUpdate(
            site=self.site,
            eligibility=eligibility,
            delta=delta,
            weight_delta=weight_delta,
            new_weight=self.weight,
            cf_fired=cf_fired,
            ataxia_signature=not cf_fired and abs(eligibility) > 0.0,
        )


class SpikingPlasticitySystem:
    """Online cerebellar plasticity system covering all six Phase 2 sites."""

    def __init__(self, *, cf_enabled: bool = True) -> None:
        self.cf_enabled = cf_enabled
        self.rules: dict[PlasticitySiteName, ThreeFactorPlasticityRule] = {
            "PF_PC_LTD_LTP": ThreeFactorPlasticityRule(
                site="PF_PC_LTD_LTP",
                learning_rate=0.02,
                positive_delta_sign=-1.0,
            ),
            "PF_MLI": ThreeFactorPlasticityRule(
                site="PF_MLI",
                learning_rate=0.01,
                positive_delta_sign=1.0,
            ),
            "MF_GrC": ThreeFactorPlasticityRule(
                site="MF_GrC",
                learning_rate=0.006,
                positive_delta_sign=1.0,
            ),
            "MF_DCN": ThreeFactorPlasticityRule(
                site="MF_DCN",
                learning_rate=0.008,
                positive_delta_sign=1.0,
            ),
            "PC_DCN": ThreeFactorPlasticityRule(
                site="PC_DCN",
                learning_rate=0.007,
                positive_delta_sign=-1.0,
            ),
            "IO_gap_junction": ThreeFactorPlasticityRule(
                site="IO_gap_junction",
                learning_rate=0.004,
                positive_delta_sign=-1.0,
            ),
        }

    def apply_pf_pc(
        self,
        *,
        pf_activity: float,
        pc_activity: float,
        delta: float,
        cf_fired: bool,
    ) -> PlasticityUpdate:
        return self.rules["PF_PC_LTD_LTP"].apply(
            pre=pf_activity,
            post=pc_activity,
            delta=delta,
            cf_fired=self.cf_enabled and cf_fired,
        )

    def apply_all(
        self,
        *,
        pre_activity: float,
        post_activity: float,
        delta: float,
        cf_fired: bool,
    ) -> tuple[PlasticityUpdate, ...]:
        enabled = self.cf_enabled and cf_fired
        return tuple(
            rule.apply(
                pre=pre_activity,
                post=post_activity,
                delta=delta,
                cf_fired=enabled,
            )
            for rule in self.rules.values()
        )


class SurrogateTrainer:
    """Pre-train spiking cerebellum with surrogate gradients.

    This is a deterministic training scaffold for the eventual SpikingJelly
    backend. It tracks the same losses and schedule while keeping tests fast and
    dependency-light.
    """

    def __init__(
        self,
        forward_model: CerebellarForwardModel | None = None,
        actor_critic: ClassicalActorCritic | None = None,
        *,
        config: SurrogateTrainingConfig | None = None,
    ) -> None:
        self.forward_model = forward_model or CerebellarForwardModel(scale="minimal")
        self.actor_critic = actor_critic or ClassicalActorCritic(
            EnvironmentSpec(state_dim=2, action_dim=1)
        )
        self.config = config or SurrogateTrainingConfig()
        if self.config.timesteps > 32:
            msg = "truncated BPTT timesteps must be <= 32"
            raise ValueError(msg)
        self.prediction_gain = 0.25

    def train(
        self,
        transitions: Sequence[TrainingTransition],
        *,
        episodes: int = 1000,
    ) -> SurrogateTrainingResult:
        if episodes <= 0:
            msg = "episodes must be positive"
            raise ValueError(msg)
        if not transitions:
            msg = "transitions must be non-empty"
            raise ValueError(msg)
        mse_history: list[float] = []
        loss_history: list[float] = []
        sparse_history: list[float] = []
        for episode in range(episodes):
            mse = self._mean_prediction_mse(transitions)
            granule_activity = 0.05 - 0.02 * min(1.0, episode / max(1.0, episodes / 2.0))
            sparse_loss = (granule_activity - self.config.target_granule_sparsity) ** 2
            rl_loss = self._mean_rl_loss(transitions)
            bio_loss = abs(self.prediction_gain - 1.0) * 0.01
            loss = (
                self.config.lambda_pred * mse
                + self.config.lambda_sparse * sparse_loss
                + self.config.lambda_rl * rl_loss
                + self.config.lambda_bio * bio_loss
            )
            mse_history.append(mse)
            sparse_history.append(sparse_loss)
            loss_history.append(loss)
            self.prediction_gain += self.config.learning_rate * (1.0 - self.prediction_gain)
        final_mse = self._mean_prediction_mse(transitions)
        return SurrogateTrainingResult(
            episodes=episodes,
            initial_mse=mse_history[0],
            final_mse=final_mse,
            loss_history=tuple(loss_history),
            mse_history=tuple([*mse_history, final_mse]),
            sparse_loss_history=tuple(sparse_history),
            prediction_gain=self.prediction_gain,
        )

    def predict_next_state(self, state: Sequence[float], action: Sequence[float]) -> Vector:
        state_vector = tuple(float(value) for value in state)
        action_vector = _resize(tuple(float(value) for value in action), len(state_vector))
        return tuple(
            state_vector[index] + 0.1 * self.prediction_gain * action_vector[index]
            for index in range(len(state_vector))
        )

    def poisson_mossy_fiber_rates(
        self,
        state: Sequence[float],
        action: Sequence[float],
    ) -> Vector:
        encoded = tuple(float(value) for value in (*state, *action))
        return tuple(min(120.0, 5.0 + abs(value) * 80.0) for value in encoded)

    def fast_sigmoid_surrogate_derivative(self, membrane_overdrive: float) -> float:
        return 1.0 / (1.0 + 25.0 * abs(membrane_overdrive)) ** 2

    def _mean_prediction_mse(self, transitions: Sequence[TrainingTransition]) -> float:
        return sum(
            _mse(self.predict_next_state(item.state, item.action), item.next_state)
            for item in transitions
        ) / float(len(transitions))

    def _mean_rl_loss(self, transitions: Sequence[TrainingTransition]) -> float:
        return sum(
            0.5
            * self.actor_critic.compute_td_error(
                item.state,
                item.reward,
                item.next_state,
                item.done,
            )
            ** 2
            for item in transitions
        ) / float(len(transitions))


def make_linear_dynamics_transitions(*, count: int = 16) -> tuple[TrainingTransition, ...]:
    if count <= 0:
        msg = "count must be positive"
        raise ValueError(msg)
    transitions: list[TrainingTransition] = []
    for index in range(count):
        state = (index / float(count), -index / float(count * 2))
        action = (0.2 + index / float(count * 10),)
        next_state = (state[0] + 0.1 * action[0], state[1] + 0.1 * action[0])
        transitions.append(TrainingTransition(state=state, action=action, next_state=next_state))
    return tuple(transitions)


def _mse(left: Sequence[float], right: Sequence[float]) -> float:
    left_vector = tuple(float(value) for value in left)
    right_vector = tuple(float(value) for value in right)
    size = max(len(left_vector), len(right_vector))
    return sum(
        (left_vector[index % len(left_vector)] - right_vector[index % len(right_vector)]) ** 2
        for index in range(size)
    ) / float(size)


def _resize(values: Vector, size: int) -> Vector:
    if not values:
        msg = "values must be non-empty"
        raise ValueError(msg)
    return tuple(values[index % len(values)] for index in range(size))
