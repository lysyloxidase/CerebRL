"""Hybrid reinforcement-learning modules for CerebRL."""

from __future__ import annotations

from .actor_critic import (
    ActorCriticUpdate,
    ActorKind,
    ClassicalActorCritic,
    EnvironmentSpec,
    create_actor_critic,
)
from .cf_rpe import (
    CFRPEComputer,
    CFRPERoutingResult,
    ClimbingFiberRPE,
    ClimbingFiberRPEResult,
    PlasticityDirection,
)
from .environments import (
    BenchmarkResult,
    ToyEnvironment,
    benchmark_sample_efficiency,
    cartpole_env,
    full_hybrid_mountain_car_gain,
    mountain_car_env,
)
from .eprop import EPropM1Trainer, EPropTrainingResult, EPropUpdate

__all__ = [
    "ActorCriticUpdate",
    "ActorKind",
    "BenchmarkResult",
    "CFRPEComputer",
    "CFRPERoutingResult",
    "ClassicalActorCritic",
    "ClimbingFiberRPE",
    "ClimbingFiberRPEResult",
    "EPropM1Trainer",
    "EPropTrainingResult",
    "EPropUpdate",
    "EnvironmentSpec",
    "PlasticityDirection",
    "ToyEnvironment",
    "benchmark_sample_efficiency",
    "cartpole_env",
    "create_actor_critic",
    "full_hybrid_mountain_car_gain",
    "mountain_car_env",
]
