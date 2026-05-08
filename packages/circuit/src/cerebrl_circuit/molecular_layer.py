"""Molecular layer feedforward inhibition."""

from __future__ import annotations

from dataclasses import dataclass

from cerebrl_neurons import StellateBasketCell

from .granular_layer import GranularLayerOutput


@dataclass(frozen=True)
class MolecularLayerOutput:
    stellate_rate_hz: float
    basket_rate_hz: float
    dendritic_gain: float
    somatic_inhibition: float
    value_estimate: float


class MolecularLayer:
    """Feedforward inhibition by stellate/basket molecular layer interneurons.

    Stellate cells inhibit PC dendrites and basket cells inhibit PC somata. Both
    receive PF excitation, forming an inhibitory surround and a value-like critic
    population for PF->MLI plasticity.
    """

    def __init__(self, interneuron_count: int) -> None:
        if interneuron_count <= 0:
            msg = "interneuron_count must be positive"
            raise ValueError(msg)
        self.interneuron_count = interneuron_count
        self.stellate_count = interneuron_count // 2
        self.basket_count = interneuron_count - self.stellate_count
        self._stellate = StellateBasketCell(subtype="stellate")
        self._basket = StellateBasketCell(subtype="basket")

    def process_parallel_fibers(
        self,
        granular_output: GranularLayerOutput,
        *,
        climbing_fiber_error: float = 0.0,
    ) -> MolecularLayerOutput:
        pf_drive = granular_output.parallel_fiber_drive
        stellate_trace = self._stellate.simulate(parallel_fiber_drive=pf_drive)
        basket_trace = self._basket.simulate(parallel_fiber_drive=pf_drive * 1.15)
        stellate_rate = stellate_trace.firing_rate_hz(label="stellate")
        basket_rate = basket_trace.firing_rate_hz(label="basket")
        dendritic_gain = max(0.65, 1.0 - stellate_rate / 250.0)
        somatic_inhibition = min(0.35, basket_rate / 220.0)
        value_estimate = pf_drive * (1.0 + climbing_fiber_error) - somatic_inhibition * 0.1
        return MolecularLayerOutput(
            stellate_rate_hz=stellate_rate,
            basket_rate_hz=basket_rate,
            dendritic_gain=dendritic_gain,
            somatic_inhibition=somatic_inhibition,
            value_estimate=value_estimate,
        )
