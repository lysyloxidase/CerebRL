from __future__ import annotations

from cerebrl_neurons import CELL_PARAMS, CONNECTIVITY, MICROZONE_SCALE, PLASTICITY_RULES
from cerebrl_neurons.receptors import receptor_current_pA


def test_parameter_atlas_contains_loop_cell_types() -> None:
    expected = {
        "purkinje",
        "granule",
        "golgi",
        "stellate_basket",
        "ubc",
        "dcn_projection",
        "dcn_inhibitory",
        "inferior_olive",
        "m1_layer5_pyramidal",
        "pontine_relay",
        "vl_thalamic_relay",
    }
    assert expected.issubset(CELL_PARAMS)
    assert CELL_PARAMS["purkinje"]["PF_synapses"] == 175000
    assert CELL_PARAMS["granule"]["Cm_pF"] == 3.0
    assert CELL_PARAMS["inferior_olive"]["gap_junction"] == "connexin_36"


def test_plasticity_connectivity_and_microzone_tables_are_present() -> None:
    assert set(PLASTICITY_RULES) == {
        "PF_PC_LTD_LTP",
        "PF_MLI",
        "MF_GrC",
        "MF_DCN",
        "PC_DCN",
        "IO_gap_junction",
    }
    assert CONNECTIVITY["GrC_to_PC_via_PF"]["synapses_per_PC"] == 175000
    assert MICROZONE_SCALE["minimal"]["PC"] == 32


def test_receptor_current_directions_and_nmda_voltage_block() -> None:
    assert receptor_current_pA("AMPA", t_ms=1.0, voltage_mV=-65.0) > 0.0
    assert receptor_current_pA("GABA_A", t_ms=3.0, voltage_mV=-65.0) < 0.0

    nmda_hyperpolarized = receptor_current_pA("NMDA", t_ms=10.0, voltage_mV=-80.0)
    nmda_depolarized = receptor_current_pA("NMDA", t_ms=10.0, voltage_mV=-40.0)
    assert nmda_depolarized > nmda_hyperpolarized

