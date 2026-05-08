# CerebRL

CerebRL is a research monorepo for a hybrid reinforcement-learning algorithm mapped
onto the human cerebello-cortical loop.

Phase 1 implements biologically grounded cell models for the cerebellar and
cortical loop cell types:

- Purkinje cells
- Granule cells
- Golgi cells
- Stellate/basket molecular layer interneurons
- Unipolar brush cells
- Deep cerebellar nuclei projection and inhibitory neurons
- Inferior olive neurons with subthreshold oscillations and gap junction coupling
- Motor cortex layer 5 pyramidal neurons
- Pontine relay neurons
- Ventrolateral thalamic relay neurons

The implementation is intentionally backend-neutral. Each cell exposes a compact
pure-Python simulator for tests and a validated specification for Brian2, NEST,
and SpikingJelly integration.

## Quickstart

```bash
cd packages/neurons
python -m pytest
python -m compileall src tests
```

Optional quality tools are configured in `packages/neurons/pyproject.toml`:

```bash
ruff check .
pyright
```

