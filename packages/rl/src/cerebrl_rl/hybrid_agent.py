"""Hybrid classical Actor-Critic plus spiking cerebellar forward model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cerebrl_circuit import LoopOutput

from .actor_critic import ActorKind, create_actor_critic
from .cerebellar_forward_model import CerebellarForwardModel, ForwardModelOutput
from .dreamer_integration import CerebRLDreamer, DreamerRollout

Vector = tuple[float, ...]


@dataclass(frozen=True)
class AgentActOutput:
    raw_action: Vector
    corrected_action: Vector
    predicted_next_state: Vector
    cerebellar_correction: Vector
    forward_model: ForwardModelOutput


@dataclass(frozen=True)
class AgentLearnOutput:
    td_error: float
    pf_pc_weight_delta: float
    pf_pc_plasticity: str
    actor_critic_updates: int
    imagined_rollout: DreamerRollout | None


class CerebRLAgent:
    """Hybrid RL agent: classical Actor-Critic + spiking cerebellar forward model."""

    def __init__(
        self,
        env: object | None,
        actor: ActorKind = "sac",
        cerebellar_scale: str = "standard",
        use_dreamer_rollouts: bool = True,
        n_microzones: int = 1,
    ) -> None:
        if n_microzones <= 0:
            msg = "n_microzones must be positive"
            raise ValueError(msg)
        self.env = env
        self.n_microzones = n_microzones
        self.actor_critic = create_actor_critic(env, actor)
        self.forward_model = CerebellarForwardModel(scale=cerebellar_scale)
        self.loop = self.forward_model.loop
        self.use_dreamer = use_dreamer_rollouts
        self.dreamer = CerebRLDreamer(self.actor_critic, self.forward_model)
        self.last_act_output: AgentActOutput | None = None
        self.last_learn_output: AgentLearnOutput | None = None

    def act(self, state: Sequence[float], *, cerebellum_enabled: bool = True) -> Vector:
        return self.act_with_trace(state, cerebellum_enabled=cerebellum_enabled).corrected_action

    def act_with_trace(
        self,
        state: Sequence[float],
        *,
        cerebellum_enabled: bool = True,
    ) -> AgentActOutput:
        raw_action = self.actor_critic.actor(state)
        forward_output = self.forward_model.refine_action(
            state,
            raw_action,
            reward_prediction_error=0.0,
            cerebellum_enabled=cerebellum_enabled,
        )
        output = AgentActOutput(
            raw_action=raw_action,
            corrected_action=forward_output.corrected_action,
            predicted_next_state=forward_output.predicted_next_state,
            cerebellar_correction=forward_output.correction,
            forward_model=forward_output,
        )
        self.last_act_output = output
        return output

    def learn(
        self,
        state: Sequence[float],
        action: Sequence[float],
        reward: float,
        next_state: Sequence[float],
        done: bool,
    ) -> AgentLearnOutput:
        delta = self.actor_critic.compute_td_error(state, reward, next_state, done)
        rpe_loop = self.inject_rpe(delta)
        update = self.actor_critic.update(state, action, reward, next_state, done)
        rollout = self._imagined_rollout(state, n_steps=5) if self.use_dreamer else None
        output = AgentLearnOutput(
            td_error=update.td_error,
            pf_pc_weight_delta=rpe_loop.pf_pc_weight_delta,
            pf_pc_plasticity=rpe_loop.pf_pc_plasticity,
            actor_critic_updates=self.actor_critic.update_count,
            imagined_rollout=rollout,
        )
        self.last_learn_output = output
        return output

    def inject_rpe(self, delta: float) -> LoopOutput:
        return self.loop.step(
            (0.0,) * self.actor_critic.spec.state_dim,
            (0.0,) * self.actor_critic.spec.action_dim,
            reward=delta,
        )

    def imagined_rollout(self, state: Sequence[float], *, n_steps: int = 5) -> DreamerRollout:
        return self._imagined_rollout(state, n_steps=n_steps)

    def _imagined_rollout(self, state: Sequence[float], *, n_steps: int = 5) -> DreamerRollout:
        return self.dreamer.imagined_rollout(state, n_steps=n_steps)

    def classical_action(self, state: Sequence[float]) -> Vector:
        return self.actor_critic.actor(state)
