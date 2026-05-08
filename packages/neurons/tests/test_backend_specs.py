from __future__ import annotations

from cerebrl_neurons import CELL_PARAMS, Brian2Spec, NESTSpec, SpikingJellySpec, create_cell


def test_every_cell_exports_brian2_nest_and_spikingjelly_specs() -> None:
    for cell_type in CELL_PARAMS:
        cell = create_cell(cell_type)

        brian2 = cell.backend("brian2")
        nest = cell.backend("nest")
        spikingjelly = cell.backend("spikingjelly")

        assert isinstance(brian2, Brian2Spec)
        assert "dv/dt" in brian2.equations
        assert brian2.threshold
        assert brian2.reset
        assert "V_thresh" in brian2.namespace

        assert isinstance(nest, NESTSpec)
        assert nest.model
        assert nest.params["biological_cell"]
        assert nest.receptor_ports["AMPA"] == 1

        assert isinstance(spikingjelly, SpikingJellySpec)
        assert spikingjelly.node_class
        assert spikingjelly.kwargs["biological_cell"]

