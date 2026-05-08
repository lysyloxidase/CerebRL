"""Shared simulation and backend-specification primitives for CerebRL neurons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import pi, sin
from typing import Literal, cast

BackendName = Literal["brian2", "nest", "spikingjelly"]
ParameterMap = Mapping[str, object]


def _empty_labels() -> dict[str, tuple[float, ...]]:
    return {}


@dataclass(frozen=True)
class Brian2Spec:
    """Minimal Brian2 implementation contract for a cell model."""

    equations: str
    threshold: str
    reset: str
    method: str
    namespace: Mapping[str, float]


@dataclass(frozen=True)
class NESTSpec:
    """NEST model contract with parameter names matching NEST conventions."""

    model: str
    params: Mapping[str, float | str | bool]
    receptor_ports: Mapping[str, int]


@dataclass(frozen=True)
class SpikingJellySpec:
    """SpikingJelly implementation contract for torch-based integration."""

    node_class: str
    kwargs: Mapping[str, float | bool | str]
    surrogate: str


@dataclass(frozen=True)
class BackendImplementation:
    """All supported backend implementations for one biological cell type."""

    brian2: Brian2Spec
    nest: NESTSpec
    spikingjelly: SpikingJellySpec

    def get(self, backend: BackendName) -> Brian2Spec | NESTSpec | SpikingJellySpec:
        if backend == "brian2":
            return self.brian2
        if backend == "nest":
            return self.nest
        return self.spikingjelly

    def validate(self) -> None:
        """Raise early if a backend spec is incomplete."""

        if not self.brian2.equations.strip():
            msg = "Brian2 equations must be non-empty"
            raise ValueError(msg)
        if "dv/dt" not in self.brian2.equations and "v/dt" not in self.brian2.equations:
            msg = "Brian2 equations must expose membrane-voltage dynamics"
            raise ValueError(msg)
        if not self.nest.model:
            msg = "NEST model name must be non-empty"
            raise ValueError(msg)
        if "V_th" not in self.nest.params:
            msg = "NEST parameters must include V_th"
            raise ValueError(msg)
        if not self.spikingjelly.node_class:
            msg = "SpikingJelly node class must be non-empty"
            raise ValueError(msg)


@dataclass(frozen=True)
class SimulationResult:
    """Pure-Python trace returned by every cell model."""

    dt_ms: float
    times_ms: tuple[float, ...]
    voltage_mV: tuple[float, ...]
    spike_times_ms: tuple[float, ...]
    labels: Mapping[str, tuple[float, ...]] = field(default_factory=_empty_labels)

    def firing_rate_hz(
        self,
        *,
        start_ms: float = 0.0,
        end_ms: float | None = None,
        label: str | None = None,
    ) -> float:
        """Return the event rate in Hz over a half-open time window."""

        end = self.times_ms[-1] if end_ms is None else end_ms
        if end <= start_ms:
            msg = "end_ms must be greater than start_ms"
            raise ValueError(msg)
        events = self.labels[label] if label is not None else self.spike_times_ms
        count = sum(1 for spike in events if start_ms < spike <= end)
        return count / ((end - start_ms) / 1000.0)

    def first_spike_after(self, time_ms: float, *, label: str | None = None) -> float | None:
        events = self.labels[label] if label is not None else self.spike_times_ms
        for spike in events:
            if spike > time_ms:
                return spike
        return None


class BioNeuron:
    """Base class for typed cell models with backend exports."""

    def __init__(
        self,
        *,
        cell_name: str,
        params: ParameterMap,
        backend_implementation: BackendImplementation,
    ) -> None:
        self.cell_name = cell_name
        self.params = params
        self.backend_implementation = backend_implementation
        self.backend_implementation.validate()

    def backend(self, backend: BackendName) -> Brian2Spec | NESTSpec | SpikingJellySpec:
        return self.backend_implementation.get(backend)


def midpoint(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and value:
        sequence = cast(Sequence[object], value)
        numeric = [float(item) for item in sequence if isinstance(item, (int, float))]
        if numeric:
            return sum(numeric) / float(len(numeric))
    return default


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def make_times(duration_ms: float, dt_ms: float) -> tuple[float, ...]:
    steps = round(duration_ms / dt_ms)
    return tuple(round(step * dt_ms, 10) for step in range(steps + 1))


def regular_spikes(
    rate_hz: float,
    duration_ms: float,
    *,
    start_ms: float = 0.0,
    end_ms: float | None = None,
) -> tuple[float, ...]:
    """Schedule deterministic spikes at the requested rate."""

    if rate_hz <= 0.0:
        return ()
    end = duration_ms if end_ms is None else min(end_ms, duration_ms)
    interval_ms = 1000.0 / rate_hz
    spikes: list[float] = []
    t_ms = start_ms + interval_ms
    while t_ms <= end + 1e-9:
        spikes.append(round(t_ms, 10))
        t_ms += interval_ms
    return tuple(spikes)


def merge_spikes(*trains: Sequence[float]) -> tuple[float, ...]:
    return tuple(sorted({round(spike, 10) for train in trains for spike in train}))


def voltage_trace(
    *,
    times_ms: Sequence[float],
    spike_times_ms: Sequence[float],
    resting_mV: float,
    dt_ms: float,
    oscillation_hz: float = 0.0,
    oscillation_amplitude_mV: float = 0.0,
    inhibition_windows: Sequence[tuple[float, float]] = (),
    hyperpolarized_mV: float = -78.0,
) -> tuple[float, ...]:
    """Create a compact membrane trace with spikes and optional oscillation."""

    spike_lookup = {round(spike, 10) for spike in spike_times_ms}
    values: list[float] = []
    for time_ms in times_ms:
        voltage = resting_mV
        if oscillation_hz > 0.0:
            voltage += oscillation_amplitude_mV * sin(2.0 * pi * oscillation_hz * time_ms / 1000.0)
        if any(start <= time_ms <= end for start, end in inhibition_windows):
            voltage = hyperpolarized_mV
        if round(time_ms, 10) in spike_lookup:
            voltage = 30.0
        elif round(time_ms - dt_ms, 10) in spike_lookup:
            voltage = min(resting_mV - 3.0, -70.0)
        values.append(round(voltage, 6))
    return tuple(values)


def default_backend_spec(
    *,
    cell_name: str,
    params: ParameterMap,
    model_class: str,
    brian2_equations: str,
    nest_model: str,
    spikingjelly_node: str,
    tau_default_ms: float,
    v_rest_default_mV: float,
    v_thresh_default_mV: float,
    v_reset_mV: float = -70.0,
    extra_nest_params: Mapping[str, float | str | bool] | None = None,
    extra_spikingjelly_kwargs: Mapping[str, float | str | bool] | None = None,
) -> BackendImplementation:
    tau_m = midpoint(params.get("tau_m_ms"), tau_default_ms)
    v_rest = midpoint(params.get("V_rest_mV"), v_rest_default_mV)
    v_thresh = midpoint(params.get("V_thresh_mV"), v_thresh_default_mV)
    nest_params: dict[str, float | str | bool] = {
        "E_L": v_rest,
        "V_reset": v_reset_mV,
        "V_th": v_thresh,
        "tau_m": tau_m,
        "biological_cell": cell_name,
        "model_class": model_class,
    }
    if extra_nest_params is not None:
        nest_params.update(extra_nest_params)
    sj_kwargs: dict[str, float | str | bool] = {
        "tau": tau_m,
        "v_threshold": v_thresh,
        "v_reset": v_reset_mV,
        "biological_cell": cell_name,
    }
    if extra_spikingjelly_kwargs is not None:
        sj_kwargs.update(extra_spikingjelly_kwargs)
    return BackendImplementation(
        brian2=Brian2Spec(
            equations=brian2_equations.strip(),
            threshold="v > V_thresh",
            reset="v = V_reset; refractory_timer = refractory_ms",
            method="exponential_euler",
            namespace={
                "tau_m": tau_m,
                "E_L": v_rest,
                "V_thresh": v_thresh,
                "V_reset": v_reset_mV,
            },
        ),
        nest=NESTSpec(
            model=nest_model,
            params=nest_params,
            receptor_ports={"AMPA": 1, "NMDA": 2, "GABA_A": 3, "GABA_B": 4},
        ),
        spikingjelly=SpikingJellySpec(
            node_class=spikingjelly_node,
            kwargs=sj_kwargs,
            surrogate="Sigmoid(alpha=4.0)",
        ),
    )
