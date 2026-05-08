"""Hybrid reinforcement-learning modules for CerebRL."""

from __future__ import annotations

from .actor_critic import (
    ActorCriticUpdate,
    ActorKind,
    ClassicalActorCritic,
    EnvironmentSpec,
    create_actor_critic,
)
from .cf_rpe import ClimbingFiberRPE, ClimbingFiberRPEResult, PlasticityDirection
from .environments import (
    BenchmarkResult,
    ToyEnvironment,
    benchmark_sample_efficiency,
    cartpole_env,
    mountain_car_env,
)

__all__ = [
    "ActorCriticUpdate",
    "ActorKind",
    "BenchmarkResult",
    "ClassicalActorCritic",
    "ClimbingFiberRPE",
    "ClimbingFiberRPEResult",
    "EnvironmentSpec",
    "PlasticityDirection",
    "ToyEnvironment",
    "benchmark_sample_efficiency",
    "cartpole_env",
    "create_actor_critic",
    "mountain_car_env",
]
