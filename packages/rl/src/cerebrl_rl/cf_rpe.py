"""Climbing-fiber reward-prediction-error computation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

PlasticityDirection = Literal["LTD", "LTP", "maintenance"]


@dataclass(frozen=True)
class ClimbingFiberRPEResult:
    """TD error and biological IO/CF interpretation."""

    raw_delta: float
    feedback_gain: float
    gated_delta: float
    climbing_fiber_rate_hz: float
    plasticity_direction: PlasticityDirection
    pf_pc_weight_delta: float


class ClimbingFiberRPE:
    """Compute the TD/RPE signal carried by climbing fibers.

    The scalar TD error is:

    delta(t) = r(t) + gamma * V(s_{t+1}) - V(s_t)

    Positive delta drives IO firing, CF spikes, PC complex spikes, and LTD at
    active PF-PC synapses. Negative delta suppresses IO below baseline and
    yields LTP at active PF-PC synapses. DCN->IO inhibition attenuates the gain.
    """

    def __init__(
        self,
        *,
        gamma: float = 0.97,
        value_weights: Sequence[float] | None = None,
        learning_rate: float = 0.01,
        feedback_sensitivity: float = 0.025,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            msg = "gamma must be in [0, 1]"
            raise ValueError(msg)
        if learning_rate <= 0.0:
            msg = "learning_rate must be positive"
            raise ValueError(msg)
        if feedback_sensitivity < 0.0:
            msg = "feedback_sensitivity must be non-negative"
            raise ValueError(msg)
        self.gamma = gamma
        self.value_weights = tuple(float(weight) for weight in value_weights or ())
        self.learning_rate = learning_rate
        self.feedback_sensitivity = feedback_sensitivity

    def value(self, state: Sequence[float]) -> float:
        vector = tuple(float(item) for item in state)
        if not vector:
            msg = "state must contain at least one element"
            raise ValueError(msg)
        if self.value_weights:
            weighted = sum(
                value * self.value_weights[index % len(self.value_weights)]
                for index, value in enumerate(vector)
            )
            return weighted / float(len(vector))
        return sum(vector) / float(len(vector))

    def compute(
        self,
        *,
        state: Sequence[float],
        next_state: Sequence[float],
        reward: float,
        dcn_inhibitory_rate_hz: float = 0.0,
    ) -> ClimbingFiberRPEResult:
        raw_delta = reward + self.gamma * self.value(next_state) - self.value(state)
        feedback_gain = 1.0 / (
            1.0 + self.feedback_sensitivity * max(0.0, dcn_inhibitory_rate_hz)
        )
        gated_delta = raw_delta * feedback_gain
        plasticity_direction = self._plasticity_direction(gated_delta)
        cf_rate = self._climbing_fiber_rate(gated_delta)
        weight_delta = self._pf_pc_weight_delta(gated_delta, plasticity_direction)
        return ClimbingFiberRPEResult(
            raw_delta=raw_delta,
            feedback_gain=feedback_gain,
            gated_delta=gated_delta,
            climbing_fiber_rate_hz=cf_rate,
            plasticity_direction=plasticity_direction,
            pf_pc_weight_delta=weight_delta,
        )

    def _plasticity_direction(self, gated_delta: float) -> PlasticityDirection:
        if gated_delta > 1e-9:
            return "LTD"
        if gated_delta < -1e-9:
            return "LTP"
        return "maintenance"

    def _climbing_fiber_rate(self, gated_delta: float) -> float:
        if gated_delta > 0.0:
            return min(8.0, 1.0 + 3.0 * gated_delta)
        if gated_delta < 0.0:
            return 0.0
        return 1.0

    def _pf_pc_weight_delta(
        self,
        gated_delta: float,
        plasticity_direction: PlasticityDirection,
    ) -> float:
        if plasticity_direction == "LTD":
            return -self.learning_rate * gated_delta
        if plasticity_direction == "LTP":
            return self.learning_rate * abs(gated_delta)
        return 0.0
