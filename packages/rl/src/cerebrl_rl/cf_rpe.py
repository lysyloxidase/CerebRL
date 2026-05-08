"""Climbing-fiber reward-prediction-error computation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import pi, sin
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


@dataclass(frozen=True)
class CFRPERoutingResult:
    """CF-RPE mixture after IO gating and DCN feedback."""

    td_error: float
    sensory_prediction_error: float
    combined_delta: float
    io_current: float
    sto_gate: float
    feedback_attenuation: float
    io_rate_hz: float
    cf_fired: bool
    complex_spike: bool
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


class CFRPEComputer:
    """Compute and route the climbing-fiber RPE signal.

    The CF signal combines TD error with sensory prediction error, then routes
    the result through IO STO gating and DCN->IO inhibition.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.7,
        gamma: float = 0.97,
        beta_feedback: float = 0.025,
        sto_frequency_hz: float = 6.0,
        io_threshold: float = 0.05,
        learning_rate: float = 0.01,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            msg = "alpha must be in [0, 1]"
            raise ValueError(msg)
        if beta_feedback < 0.0:
            msg = "beta_feedback must be non-negative"
            raise ValueError(msg)
        if sto_frequency_hz <= 0.0:
            msg = "sto_frequency_hz must be positive"
            raise ValueError(msg)
        self.alpha = alpha
        self.gamma = gamma
        self.beta_feedback = beta_feedback
        self.sto_frequency_hz = sto_frequency_hz
        self.io_threshold = io_threshold
        self.learning_rate = learning_rate
        self._td = ClimbingFiberRPE(gamma=gamma, learning_rate=learning_rate)

    def compute_and_route(
        self,
        *,
        state: Sequence[float],
        next_state: Sequence[float],
        predicted_next_state: Sequence[float],
        reward: float,
        dcn_inhibitory_rate_hz: float = 0.0,
        sto_phase_ms: float = 42.0,
    ) -> CFRPERoutingResult:
        td_result = self._td.compute(
            state=state,
            next_state=next_state,
            reward=reward,
            dcn_inhibitory_rate_hz=0.0,
        )
        pred_error = _mean_abs_difference(next_state, predicted_next_state)
        combined_delta = self.alpha * td_result.raw_delta + (1.0 - self.alpha) * pred_error
        feedback = self.beta_feedback * max(0.0, dcn_inhibitory_rate_hz)
        io_current = combined_delta - feedback
        sto_gate = max(
            0.0,
            sin(2.0 * pi * self.sto_frequency_hz * sto_phase_ms / 1000.0),
        )
        gated_current = io_current * sto_gate
        cf_fired = gated_current > self.io_threshold
        if cf_fired:
            io_rate = min(8.0, 1.0 + 3.0 * gated_current)
        elif io_current < 0.0:
            io_rate = 0.0
        else:
            io_rate = min(1.0, max(0.05, 0.2 + gated_current))
        plasticity = _plasticity_direction(gated_current)
        weight_delta = _pf_pc_weight_delta(
            gated_current,
            plasticity,
            learning_rate=self.learning_rate,
        )
        return CFRPERoutingResult(
            td_error=td_result.raw_delta,
            sensory_prediction_error=pred_error,
            combined_delta=combined_delta,
            io_current=io_current,
            sto_gate=sto_gate,
            feedback_attenuation=feedback,
            io_rate_hz=io_rate,
            cf_fired=cf_fired,
            complex_spike=cf_fired,
            plasticity_direction=plasticity,
            pf_pc_weight_delta=weight_delta,
        )


def _mean_abs_difference(left: Sequence[float], right: Sequence[float]) -> float:
    left_vector = tuple(float(value) for value in left)
    right_vector = tuple(float(value) for value in right)
    if not left_vector or not right_vector:
        msg = "state vectors must be non-empty"
        raise ValueError(msg)
    size = max(len(left_vector), len(right_vector))
    return sum(
        abs(left_vector[index % len(left_vector)] - right_vector[index % len(right_vector)])
        for index in range(size)
    ) / float(size)


def _plasticity_direction(gated_delta: float) -> PlasticityDirection:
    if gated_delta > 1e-9:
        return "LTD"
    if gated_delta < -1e-9:
        return "LTP"
    return "maintenance"


def _pf_pc_weight_delta(
    gated_delta: float,
    plasticity_direction: PlasticityDirection,
    *,
    learning_rate: float,
) -> float:
    if plasticity_direction == "LTD":
        return -learning_rate * gated_delta
    if plasticity_direction == "LTP":
        return learning_rate * abs(gated_delta)
    return 0.0
