"""Plasticity rules for CerebRL circuit modules."""

from __future__ import annotations

try:
    from cerebrl_neurons.params import PLASTICITY_RULES
except ModuleNotFoundError:  # pragma: no cover
    PLASTICITY_RULES: dict[str, dict[str, object]] = {}

__all__ = ["PLASTICITY_RULES"]

