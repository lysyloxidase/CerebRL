# CerebRL

[![Python](https://github.com/lysyloxidase/CerebRL/actions/workflows/ci-python.yml/badge.svg)](https://github.com/lysyloxidase/CerebRL/actions/workflows/ci-python.yml)
[![Rust](https://github.com/lysyloxidase/CerebRL/actions/workflows/ci-rust.yml/badge.svg)](https://github.com/lysyloxidase/CerebRL/actions/workflows/ci-rust.yml)
[![Frontend](https://github.com/lysyloxidase/CerebRL/actions/workflows/ci-frontend.yml/badge.svg)](https://github.com/lysyloxidase/CerebRL/actions/workflows/ci-frontend.yml)

RL that thinks like a cerebellum: a biologically grounded hybrid reinforcement
learning research platform mapped onto the cerebello-cortical loop.

Public repository: https://github.com/lysyloxidase/CerebRL

CerebRL combines a classical actor-critic backbone with a spiking cerebellar
forward model. The inner loop models M1 -> pontine nuclei -> mossy fibers ->
granular layer -> molecular layer -> Purkinje cells -> DCN -> VL thalamus ->
M1, targeting the 10-25 ms correction circuit used for online motor control.

This repository is intentionally dependency-light today. The core packages use
deterministic pure-Python simulators and quality gates that make the biological
contracts testable before heavier Brian2, NEST, SpikingJelly, Stable-Baselines3,
or neuromorphic backends are plugged in.

## Architecture

```text
Classical RL outer loop
  state -> actor -> raw action -> environment
             ^                         |
             |                         v
          critic <- reward, next state, TD error

Spiking cerebellar inner loop
  M1 L5 -> pontine relay -> mossy fibers -> granule sparse expansion
      -> parallel fibers -> Purkinje linear readout -> DCN correction
      -> VL thalamus -> M1 updated action

Teaching signal
  critic TD/RPE -> inferior olive -> climbing fibers -> complex spikes
  DCN inhibitory feedback -> IO gain control
```

The key modeling idea is that the granular layer acts like a sparse expansion
encoder: low-dimensional state/action context is expanded into a high-dimensional
parallel-fiber representation. Purkinje cells then learn a biologically gated
linear readout using climbing-fiber reward prediction error.

## What Is Implemented

- Cell-type faithful neuron models for the cerebello-cortical loop:
  Purkinje, granule, Golgi, stellate/basket, UBC, DCN projection/inhibitory,
  inferior olive, M1 layer 5, pontine relay, and VL thalamic relay neurons.
- Complete parameter atlas with literature provenance in
  `packages/neurons/src/cerebrl_neurons/params.py`.
- Microzone circuit assembly covering granular, molecular, Purkinje, DCN, and
  inferior olive modules.
- Closed cerebello-cortical loop with CF-RPE routing.
- Hybrid RL agent with SAC/PPO/TD3-style actor-critic scaffold, cerebellar
  action refinement, and Dreamer-style imagined rollouts.
- Training pipeline for surrogate-gradient pretraining, online three-factor
  cerebellar plasticity, and e-prop-style M1 learning.
- Biological applications for SCA modeling, drug screening, toxicology, and
  harmaline tremor.
- Benchmark package with deterministic gates for sample efficiency, ablations,
  visualizer contracts, neuromorphic deployment metadata, and falsifiable
  predictions.

## Monorepo Layout

```text
packages/
  neurons/      Cell models, receptor kinetics, parameter atlas
  circuit/      Microzone and full M1-pons-cerebellum-DCN-VL-M1 loop
  rl/           Hybrid actor-critic, CF-RPE, Dreamer and training scaffolds
  biology/      SCA, drug-screen, toxicology, phenotype prediction modules
  benchmarks/   Phase 7 benchmark, ablation, visualization and NIR contracts
  kernels_rs/   Rust sparse connectivity and spike propagation kernels

apps/
  api/          FastAPI health endpoint and API placeholder
  web/          Next.js/React visualization component package
  cli/          Command-line entrypoint scaffold

docs/           Architecture, neuron atlas, plasticity, loop anatomy and theory
data/           Parameter and connectivity JSON assets
```

## Quick Start

```bash
git clone https://github.com/lysyloxidase/CerebRL.git
cd CerebRL
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip pytest ruff pyright
./.venv/bin/python -m pip install -e packages/neurons -e packages/circuit -e packages/rl -e packages/biology -e packages/benchmarks
```

Run the core validation suite:

```bash
./.venv/bin/python -m pytest packages/neurons packages/circuit packages/rl packages/biology packages/benchmarks -q
./.venv/bin/python -m ruff check packages/neurons packages/circuit packages/rl packages/biology packages/benchmarks
```

Run the API health endpoint:

```bash
./.venv/bin/python -m pip install fastapi uvicorn
./.venv/bin/python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
```

Run frontend typecheck:

```bash
cd apps/web
npm install
npm run typecheck
```

## Example Usage

```python
from cerebrl_rl.environments import mountain_car_env
from cerebrl_rl.hybrid_agent import CerebRLAgent

agent = CerebRLAgent(mountain_car_env(), cerebellar_scale="minimal")
state = (0.1, -0.2)

trace = agent.act_with_trace(state)
print(trace.raw_action)
print(trace.cerebellar_correction)
print(trace.corrected_action)
```

```python
from cerebrl_bio import CerebellarDrugScreen
from cerebrl_rl.environments import mountain_car_env
from cerebrl_rl.hybrid_agent import CerebRLAgent

agent = CerebRLAgent(mountain_car_env(), cerebellar_scale="minimal")
screen = CerebellarDrugScreen()
results = screen.screen(agent, "SCA6", list(screen.known_compounds()))

for result in results:
    print(result.compound.name, result.recovery_fraction, result.p_value)
```

## Benchmark Claims

The current benchmark layer is deterministic and intended as a contract suite,
not as final large-scale training evidence. It verifies that the implementation
surfaces the metrics needed for the full experimental program:

- Mountain-Car: CerebRL reaches 90% reward with at least 30% fewer episodes than
  the SAC baseline contract.
- Eyeblink: conditioned-response acquisition follows the Medina/Mauk-style
  reference curve within the configured tolerance.
- Reaching: cerebellar correction improves trajectory smoothness by at least 2x
  over actor-only control.
- Harmaline: IO CaV3.1 block produces an 8-12 Hz DCN tremor signature.
- SCA6: 40% CaV2.1 reduction produces performance drop and Purkinje ISI CV
  increase in the 20-50% target window.

Run the final gates:

```bash
./.venv/bin/python -m pytest packages/benchmarks -q
```

## Six Plasticity Sites

- PF -> PC: CF-gated three-factor LTD/LTP.
- PF -> MLI: CF-gated bidirectional value update.
- MF -> GrC: NMDA-dependent representation tuning.
- MF -> DCN: rebound-driven consolidation.
- PC -> DCN: rebound-gated inhibitory plasticity.
- IO gap junctions: activity-dependent synchrony control.

## Biological Applications

CerebRL preserves cell identity, channel parameters, synaptic mechanisms, and
loop anatomy so perturbations map onto biological hypotheses:

- SCA1, SCA2, SCA3, SCA6, and SCA14 disease trajectories.
- In silico drug screening for riluzole, chlorzoxazone, 4-AP, acetazolamide,
  and mechanism-matched candidates.
- Toxicology dose-response curves for mercury, lead, manganese, ethanol, and
  phenytoin.
- Harmaline tremor through inferior-olive synchrony and DCN oscillation.

## Falsifiable Predictions

1. CF-RPE is necessary for cerebellar RL. Ablating the CF-RPE channel should
   degrade RL performance while preserving supervised adaptation.
2. DCN -> IO negative feedback implements an adaptive learning-rate mechanism.
   Blocking it should cause oscillatory overcorrection.
3. Microzone modularity should support transfer from reaching to tracking faster
   than training a tabula-rasa cerebellum.
4. SCA6 CaV2.1 reduction should increase Purkinje ISI CV by 20-50%, matching
   the clinical electrophysiology target window.

Executable checks live in `packages/benchmarks/src/cerebrl_bench/predictions.py`
and `packages/benchmarks/tests/test_phase7_final_gates.py`.

## Neuromorphic Path

The benchmark package defines NIR-oriented deployment metadata for:

- SpiNNaker-2: target 100 microzones at real-time or better.
- Loihi-2: on-chip three-factor learning and graded spike support.
- BrainScaleS-2: accelerated screening mode for parameter sweeps.

See `packages/benchmarks/src/cerebrl_bench/neuromorphic.py`.

## Status

CerebRL is a research prototype. The biological structure, interfaces, and
quality gates are in place; the next step is replacing deterministic scaffolds
with large-scale Brian2/NEST/SpikingJelly simulations and real RL benchmark
training runs.

## License

MIT. See `LICENSE`.
