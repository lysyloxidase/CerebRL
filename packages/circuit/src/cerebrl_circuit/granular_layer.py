"""Granular layer sparse-expansion encoder."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import gcd


@dataclass(frozen=True)
class GranularLayerOutput:
    """Population code emitted by GrC axons as parallel fibers."""

    mossy_fiber_rates_hz: tuple[float, ...]
    active_granule_indices: tuple[int, ...]
    active_fraction: float
    granule_spike_rate_hz: float
    golgi_feedback_hz: float
    parallel_fiber_drive: float
    expansion_ratio: float

    @property
    def active_count(self) -> int:
        return len(self.active_granule_indices)


class GranularLayer:
    """Sparse expansion encoder: MF -> GrC.

    Each granule cell samples 4-5 mossy fibers in a glomerulus. The layer expands
    a low-dimensional MF pattern into a high-dimensional, sparse PF code while
    Golgi feedback keeps instantaneous activity near 1-5%.
    """

    def __init__(
        self,
        granule_count: int,
        golgi_count: int,
        *,
        mossy_fiber_count: int = 512,
        seed: int = 17,
    ) -> None:
        if granule_count <= 0:
            msg = "granule_count must be positive"
            raise ValueError(msg)
        if golgi_count <= 0:
            msg = "golgi_count must be positive"
            raise ValueError(msg)
        if mossy_fiber_count < 5:
            msg = "mossy_fiber_count must be at least 5"
            raise ValueError(msg)
        self.granule_count = granule_count
        self.golgi_count = golgi_count
        self.mossy_fiber_count = mossy_fiber_count
        self.seed = seed

    def mossy_inputs_for_granule(self, granule_index: int) -> tuple[int, ...]:
        """Return the deterministic 4-5 MF glomerular sample for one GrC."""

        if granule_index < 0 or granule_index >= self.granule_count:
            msg = "granule_index out of range"
            raise IndexError(msg)
        input_count = 4 + (granule_index % 2)
        return tuple(
            (granule_index * 37 + offset * 101 + self.seed) % self.mossy_fiber_count
            for offset in range(input_count)
        )

    def encode(self, mossy_fiber_rates_hz: Sequence[float]) -> GranularLayerOutput:
        """Encode MF rates into a deterministic sparse GrC/PF activity pattern."""

        rates = tuple(max(0.0, float(rate)) for rate in mossy_fiber_rates_hz)
        if not rates:
            msg = "mossy_fiber_rates_hz must contain at least one rate"
            raise ValueError(msg)

        mean_rate = sum(rates) / float(len(rates))
        peak_rate = max(rates)
        normalized_drive = min(1.0, (0.7 * mean_rate + 0.3 * peak_rate) / 100.0)
        golgi_feedback_hz = 3.0 + 5.0 * normalized_drive
        target_sparsity = min(0.05, max(0.01, 0.01 + 0.035 * normalized_drive))
        active_count = max(1, round(self.granule_count * target_sparsity))

        weighted_sum = sum((index + 1) * rate for index, rate in enumerate(rates))
        start = round(weighted_sum * 10.0) % self.granule_count
        step = 7919
        while gcd(step, self.granule_count) != 1:
            step += 2
        active = tuple((start + index * step) % self.granule_count for index in range(active_count))

        active_fraction = active_count / float(self.granule_count)
        return GranularLayerOutput(
            mossy_fiber_rates_hz=rates,
            active_granule_indices=active,
            active_fraction=active_fraction,
            granule_spike_rate_hz=800.0 * normalized_drive,
            golgi_feedback_hz=golgi_feedback_hz,
            parallel_fiber_drive=active_fraction,
            expansion_ratio=self.granule_count / float(len(rates)),
        )
