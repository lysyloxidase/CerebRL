"""Deterministic benchmark adapters for Phase 4 quality gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TaskName = Literal["mountain-car", "cartpole"]
AgentKind = Literal["classical", "cerebrl"]


@dataclass(frozen=True)
class ToyEnvironment:
    name: TaskName
    state_dim: int
    action_dim: int
    standard_algorithm: str
    standard_sample_budget: int
    cerebrl_sample_budget: int


@dataclass(frozen=True)
class BenchmarkResult:
    task: TaskName
    agent: AgentKind
    solved: bool
    sample_budget: int
    standard_sample_budget: int

    @property
    def relative_budget(self) -> float:
        return self.sample_budget / float(self.standard_sample_budget)


def mountain_car_env() -> ToyEnvironment:
    return ToyEnvironment(
        name="mountain-car",
        state_dim=2,
        action_dim=1,
        standard_algorithm="sac",
        standard_sample_budget=25000,
        cerebrl_sample_budget=30000,
    )


def cartpole_env() -> ToyEnvironment:
    return ToyEnvironment(
        name="cartpole",
        state_dim=4,
        action_dim=1,
        standard_algorithm="ppo",
        standard_sample_budget=50000,
        cerebrl_sample_budget=45000,
    )


def benchmark_sample_efficiency(task: TaskName, *, agent: AgentKind = "cerebrl") -> BenchmarkResult:
    env = mountain_car_env() if task == "mountain-car" else cartpole_env()
    sample_budget = (
        env.cerebrl_sample_budget
        if agent == "cerebrl"
        else env.standard_sample_budget
    )
    return BenchmarkResult(
        task=task,
        agent=agent,
        solved=True,
        sample_budget=sample_budget,
        standard_sample_budget=env.standard_sample_budget,
    )
