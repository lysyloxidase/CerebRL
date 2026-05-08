"""Closed M1 -> pons -> cerebellum -> DCN -> VL -> M1 loop."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cerebrl_neurons import M1Layer5PyramidalNeuron, PontineRelayNeuron, VLThalamicRelayNeuron
from cerebrl_rl import ClimbingFiberRPE, ClimbingFiberRPEResult

from .microzone import CerebellarMicrozone, MicrozoneStepResult

Vector = tuple[float, ...]


@dataclass(frozen=True)
class M1Output:
    action: Vector
    cortical_policy: Vector
    corticopontine_spikes_ms: tuple[float, ...]
    corticopontine_rate_hz: float
    corticospinal_rate_hz: float


@dataclass(frozen=True)
class PontineOutput:
    mossy_fiber_rates_hz: Vector
    relay_spikes_ms: tuple[float, ...]
    latency_ms: float


@dataclass(frozen=True)
class VLThalamicOutput:
    relay_spikes_ms: tuple[float, ...]
    correction: Vector
    thalamic_drive_hz: float
    latency_ms: float


@dataclass(frozen=True)
class ParallelMotorOutput:
    red_nucleus_drive: float
    rubrospinal_command: Vector
    striatal_disynaptic_signal: float


@dataclass(frozen=True)
class LoopOutput:
    corrected_action: Vector
    uncorrected_action: Vector
    cerebellar_correction: Vector
    predicted_next_state: Vector
    rpe: float
    rpe_result: ClimbingFiberRPEResult
    loop_latency_ms: float
    latency_by_segment_ms: dict[str, float]
    path: tuple[str, ...]
    m1_output: M1Output
    pontine_output: PontineOutput
    microzone: MicrozoneStepResult | None
    vl_output: VLThalamicOutput
    parallel_motor_output: ParallelMotorOutput
    m1_feedback_rate_hz: float
    pf_pc_plasticity: str
    pf_pc_weight_delta: float
    io_firing_rate_hz: float
    complex_spike_count: int
    dysmetria_index: float


class M1Module:
    """Layer 5 M1 command source with corticospinal and corticopontine branches."""

    def __init__(self, neuron_count: int = 100) -> None:
        if neuron_count <= 0:
            msg = "neuron_count must be positive"
            raise ValueError(msg)
        self.neuron_count = neuron_count
        self._representative = M1Layer5PyramidalNeuron()

    def issue_command(self, state: Sequence[float], action: Sequence[float]) -> M1Output:
        state_vector = _as_vector(state, name="state")
        action_vector = _as_vector(action, name="action")
        cortical_policy = _resize_vector(
            tuple(0.05 * value for value in state_vector),
            len(action_vector),
        )
        action_magnitude = _mean_abs(action_vector)
        movement_state = "movement" if action_magnitude > 0.05 else "quiet"
        trace = self._representative.simulate(duration_ms=10000.0, state=movement_state)
        corticopontine_rate = trace.firing_rate_hz(label=movement_state)
        corticospinal_rate = corticopontine_rate * (1.0 + min(1.0, action_magnitude))
        return M1Output(
            action=action_vector,
            cortical_policy=cortical_policy,
            corticopontine_spikes_ms=(0.0,),
            corticopontine_rate_hz=corticopontine_rate,
            corticospinal_rate_hz=corticospinal_rate,
        )


class PontineRelay:
    """Pontine relay from corticopontine M1 activity to mossy fibers."""

    def __init__(self, neuron_count: int = 50) -> None:
        if neuron_count <= 0:
            msg = "neuron_count must be positive"
            raise ValueError(msg)
        self.neuron_count = neuron_count
        self._relay = PontineRelayNeuron()

    @property
    def latency_ms(self) -> float:
        return self._relay.latency_ms

    def relay(self, m1_output: M1Output, state: Sequence[float]) -> PontineOutput:
        state_vector = _as_vector(state, name="state")
        context_rates = tuple(min(120.0, 5.0 + abs(value) * 45.0) for value in state_vector)
        action_rates = tuple(min(120.0, 10.0 + abs(value) * 70.0) for value in m1_output.action)
        policy_rates = tuple(
            min(120.0, 10.0 + abs(value) * 60.0)
            for value in m1_output.cortical_policy
        )
        relay_spikes = self._relay.relay(m1_output.corticopontine_spikes_ms)
        return PontineOutput(
            mossy_fiber_rates_hz=context_rates + action_rates + policy_rates,
            relay_spikes_ms=relay_spikes,
            latency_ms=self.latency_ms,
        )


class VLThalamicRelay:
    """VL relay that returns DCN corrections to M1."""

    def __init__(self, neuron_count: int = 20) -> None:
        if neuron_count <= 0:
            msg = "neuron_count must be positive"
            raise ValueError(msg)
        self.neuron_count = neuron_count
        self._relay = VLThalamicRelayNeuron()

    @property
    def latency_ms(self) -> float:
        return self._relay.latency_ms

    def relay_correction(
        self,
        *,
        dcn_motor_command: float,
        rpe: float,
        action_dimensions: int,
        dcn_event_time_ms: float,
    ) -> VLThalamicOutput:
        relay_spikes = self._relay.relay_to_m1((dcn_event_time_ms,))
        sign = 1.0 if rpe >= 0.0 else -1.0
        correction_scalar = sign * min(0.5, dcn_motor_command * 0.25)
        correction = tuple(correction_scalar for _ in range(action_dimensions))
        return VLThalamicOutput(
            relay_spikes_ms=relay_spikes,
            correction=correction,
            thalamic_drive_hz=20.0 + dcn_motor_command * 80.0,
            latency_ms=self.latency_ms,
        )


class CerebelloCorticalLoop:
    """Complete closed loop: M1 -> cerebellum -> M1.

    The loop propagates an M1 corticopontine spike through pontine mossy fibers,
    the microzone, DCN, VL thalamus, and back into M1. The signal path is kept in
    the biological 10-25 ms online-correction range while the microzone's
    learning traces can still be evaluated over a longer CF window.
    """

    def __init__(self, microzone_scale: str = "standard") -> None:
        self.m1 = M1Module()
        self.pontine = PontineRelay()
        self.cerebellum = CerebellarMicrozone(scale=microzone_scale)
        self.vl_thalamus = VLThalamicRelay()
        self.rpe = ClimbingFiberRPE()
        self.io_feedback_adaptation_hz = 0.0

    def step(
        self,
        state: Sequence[float],
        action: Sequence[float],
        reward: float,
        dt_ms: float = 1.0,
        *,
        cerebellum_enabled: bool = True,
    ) -> LoopOutput:
        """Run one timestep of the full correction circuit."""

        state_vector = _as_vector(state, name="state")
        action_vector = _as_vector(action, name="action")
        if dt_ms <= 0.0:
            msg = "dt_ms must be positive"
            raise ValueError(msg)

        m1_output = self.m1.issue_command(state_vector, action_vector)
        pontine_output = self.pontine.relay(m1_output, state_vector)
        uncorrected_next_state = _predict_next_state(state_vector, action_vector, dt_ms=dt_ms)
        rpe_result = self.rpe.compute(
            state=state_vector,
            next_state=uncorrected_next_state,
            reward=reward,
            dcn_inhibitory_rate_hz=self.io_feedback_adaptation_hz,
        )
        latency = self._latency_by_segment()
        loop_latency_ms = sum(latency.values())

        if cerebellum_enabled:
            microzone_output = self.cerebellum.step(
                pontine_output.mossy_fiber_rates_hz,
                reward_prediction_error=rpe_result.gated_delta,
                duration_ms=1000.0,
            )
            vl_output = self.vl_thalamus.relay_correction(
                dcn_motor_command=microzone_output.dcn.motor_command,
                rpe=rpe_result.gated_delta,
                action_dimensions=len(action_vector),
                dcn_event_time_ms=loop_latency_ms - self.vl_thalamus.latency_ms,
            )
            corrected_action = _add_vectors(action_vector, vl_output.correction)
            self._update_io_feedback(microzone_output.dcn.io_feedback_rate_hz)
            io_firing_rate = microzone_output.inferior_olive.firing_rate_hz
            complex_spike_count = microzone_output.purkinje.complex_spike_count
            motor_command = microzone_output.dcn.motor_command
        else:
            microzone_output = None
            vl_output = VLThalamicOutput(
                relay_spikes_ms=(),
                correction=tuple(0.0 for _ in action_vector),
                thalamic_drive_hz=0.0,
                latency_ms=0.0,
            )
            corrected_action = action_vector
            io_firing_rate = 0.0
            complex_spike_count = 0
            motor_command = 0.0

        predicted_next_state = _predict_next_state(state_vector, corrected_action, dt_ms=dt_ms)
        parallel_output = _parallel_motor_output(
            motor_command=motor_command,
            action_dimensions=len(action_vector),
            rpe=rpe_result.gated_delta,
        )
        dysmetria_index = _mean_abs(_subtract_vectors(corrected_action, action_vector))
        return LoopOutput(
            corrected_action=corrected_action,
            uncorrected_action=action_vector,
            cerebellar_correction=vl_output.correction,
            predicted_next_state=predicted_next_state,
            rpe=rpe_result.gated_delta,
            rpe_result=rpe_result,
            loop_latency_ms=loop_latency_ms,
            latency_by_segment_ms=latency,
            path=(
                "M1_L5",
                "pontine_nuclei",
                "mossy_fibers",
                "granular_layer",
                "parallel_fibers",
                "purkinje_cells",
                "dcn",
                "vl_thalamus",
                "M1_L5",
            ),
            m1_output=m1_output,
            pontine_output=pontine_output,
            microzone=microzone_output,
            vl_output=vl_output,
            parallel_motor_output=parallel_output,
            m1_feedback_rate_hz=m1_output.corticospinal_rate_hz + vl_output.thalamic_drive_hz,
            pf_pc_plasticity=rpe_result.plasticity_direction,
            pf_pc_weight_delta=rpe_result.pf_pc_weight_delta,
            io_firing_rate_hz=io_firing_rate,
            complex_spike_count=complex_spike_count,
            dysmetria_index=dysmetria_index,
        )

    def reset_io_feedback(self) -> None:
        self.io_feedback_adaptation_hz = 0.0

    def _update_io_feedback(self, dcn_feedback_rate_hz: float) -> None:
        self.io_feedback_adaptation_hz = min(
            200.0,
            self.io_feedback_adaptation_hz + max(0.0, dcn_feedback_rate_hz) * 0.25,
        )

    def _latency_by_segment(self) -> dict[str, float]:
        return {
            "m1_to_pontine": self.pontine.latency_ms,
            "pontine_to_mossy_fiber": 1.5,
            "mossy_granule_parallel_fiber_to_purkinje": 2.0,
            "purkinje_to_dcn": 3.0,
            "dcn_to_vl_to_m1": self.vl_thalamus.latency_ms,
        }


def _as_vector(values: Sequence[float], *, name: str) -> Vector:
    vector = tuple(float(value) for value in values)
    if not vector:
        msg = f"{name} must contain at least one element"
        raise ValueError(msg)
    return vector


def _resize_vector(values: Vector, size: int) -> Vector:
    if size <= 0:
        msg = "size must be positive"
        raise ValueError(msg)
    if len(values) == size:
        return values
    return tuple(values[index % len(values)] for index in range(size))


def _add_vectors(left: Vector, right: Vector) -> Vector:
    resized_right = _resize_vector(right, len(left))
    return tuple(left[index] + resized_right[index] for index in range(len(left)))


def _subtract_vectors(left: Vector, right: Vector) -> Vector:
    resized_right = _resize_vector(right, len(left))
    return tuple(left[index] - resized_right[index] for index in range(len(left)))


def _mean_abs(values: Sequence[float]) -> float:
    vector = tuple(float(value) for value in values)
    if not vector:
        return 0.0
    return sum(abs(value) for value in vector) / float(len(vector))


def _predict_next_state(state: Vector, action: Vector, *, dt_ms: float) -> Vector:
    action_for_state = _resize_vector(action, len(state))
    dt_scale = dt_ms / 10.0
    return tuple(
        state[index] + action_for_state[index] * dt_scale
        for index in range(len(state))
    )


def _parallel_motor_output(
    *,
    motor_command: float,
    action_dimensions: int,
    rpe: float,
) -> ParallelMotorOutput:
    sign = 1.0 if rpe >= 0.0 else -1.0
    red_nucleus_drive = motor_command * 0.65
    rubrospinal = tuple(sign * red_nucleus_drive * 0.5 for _ in range(action_dimensions))
    return ParallelMotorOutput(
        red_nucleus_drive=red_nucleus_drive,
        rubrospinal_command=rubrospinal,
        striatal_disynaptic_signal=motor_command * 0.2,
    )
