"""Connectivity rules shared by circuit assemblies."""

from __future__ import annotations

try:
    from cerebrl_neurons.params import CONNECTIVITY
except ModuleNotFoundError:  # pragma: no cover - editable install convenience
    CONNECTIVITY: dict[str, dict[str, object]] = {}

__all__ = ["CONNECTIVITY"]

