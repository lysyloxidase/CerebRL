"""Complete biophysical parameter atlas for the cerebello-cortical loop.

Every parameter is traceable to a primary electrophysiology or anatomy source.
"""

from __future__ import annotations

CELL_PARAMS: dict[str, dict[str, object]] = {
    "purkinje": {
        "Cm_specific_uF_cm2": 0.77,
        "Cm_whole_pF": [700, 1000],
        "Rin_MOhm": [10, 50],
        "tau_m_ms": [50, 80],
        "V_rest_mV": -65,
        "V_thresh_mV": -55,
        "spontaneous_simple_spike_Hz": [30, 70],
        "complex_spike_Hz": 1.0,
        "max_driven_Hz": 200,
        "PF_synapses": 175000,
        "CF_contacts": [250, 1500],
        "neurotransmitter": "GABA",
        "count_human": 15e6,
        "model_class": "EGLIF_multicompartment",
        "source": "Roth & Häusser 2001; Andersen 1992; Napper & Harvey 1988",
    },
    "granule": {
        "Cm_pF": 3.0,
        "Rin_GOhm": [1.0, 3.4],
        "tau_m_ms": 3.15,
        "V_rest_mV": [-58, -70],
        "V_thresh_mV": -35,
        "spontaneous_Hz": [0, 0.31],
        "burst_Hz": [800, 1000],
        "MF_inputs_per_cell": [4, 5],
        "PF_length_mm": [3, 5],
        "neurotransmitter": "glutamate",
        "count_human": 66e9,
        "model_class": "LIF_or_Izhikevich",
        "source": "D'Angelo 1995/2001; Hamann 2002; Azevedo 2009",
    },
    "golgi": {
        "spontaneous_Hz": [3, 8],
        "neurotransmitter": "GABA+glycine",
        "role": "Feedback inhibition of granule cells in glomerulus",
        "model_class": "AdEx_pacemaker",
        "source": "Solinas 2007; Forti 2006; Dieudonné 1998",
    },
    "stellate_basket": {
        "spontaneous_Hz": [1, 35],
        "neurotransmitter": "GABA",
        "role": "Feedforward inhibition of PCs (stellate=dendritic, basket=somatic)",
        "rl_role": "CRITIC in Shinji 2024 actor-critic mapping",
        "model_class": "LIF_or_AdEx",
    },
    "ubc": {
        "spontaneous_Hz": [0.5, 30],
        "subtypes": ["ON_mGluR1", "OFF_mGluR2"],
        "role": "Amplify/delay mossy fiber signal in vestibulocerebellum",
        "model_class": "LIF_slow_synaptic",
        "source": "Borges-Merjane & Bhatt eLife 2019",
    },
    "dcn_projection": {
        "Cm_pF": 184,
        "Rin_MOhm": [31, 46],
        "tau_m_ms": 7,
        "spontaneous_Hz": [20, 100],
        "max_Hz": 160,
        "neurotransmitter": "glutamate",
        "rebound_firing": True,
        "PC_convergence": [50, 860],
        "count_human_dentate": 5e6,
        "model_class": "AdEx_with_rebound",
        "source": "Canto 2016; Raman 2000; Gauck & Jaeger 2000",
    },
    "dcn_inhibitory": {
        "neurotransmitter": "GABA",
        "target": "inferior_olive",
        "role": "Negative feedback: DCN inhibits IO -> reduces CF error signal",
        "source": "Bengtsson & Hesslow 2006",
    },
    "inferior_olive": {
        "Cm_pF": [200, 240],
        "Rin_MOhm": [23, 31],
        "V_rest_mV": -65,
        "firing_Hz": 1.0,
        "STO_Hz": [4, 12],
        "gap_junction": "connexin_36",
        "count_human": 5e5,
        "role": "SOURCE of ALL climbing fibers = TD/RPE ERROR SIGNAL",
        "model_class": "HH_with_Ih_CaT_gap_junctions",
        "source": "Llinás & Yarom 1986; Bazzigaluppi 2012",
    },
    "m1_layer5_pyramidal": {
        "Cm_pF": [150, 250],
        "Rin_MOhm": [50, 150],
        "tau_m_ms": [10, 25],
        "V_rest_mV": -52.9,
        "spontaneous_Hz_quiet": 5.6,
        "spontaneous_Hz_movement": 12.7,
        "subtypes": ["corticospinal", "corticopontine"],
        "model_class": "AdEx_or_Izhikevich",
        "source": "Schiemann 2015 Cell Reports",
    },
    "pontine_relay": {
        "role": "Relay cortical efference copy to cerebellum via mossy fibers",
        "latency_ms": 1,
        "model_class": "LIF_relay",
    },
    "vl_thalamic_relay": {
        "role": "Relay DCN output to M1; cerebellum->M1 leg approximately 5-6 ms",
        "latency_ms": [5, 6],
        "model_class": "LIF_relay_with_burst",
    },
}

PLASTICITY_RULES: dict[str, dict[str, object]] = {
    "PF_PC_LTD_LTP": {
        "description": "THREE-FACTOR rule: CF(RPE) + PF(context) -> LTD at PF-PC synapse",
        "equation": "dw/dt = -eta_LTD*PF(t)*CF(t-Delta)*theta_LTD(Ca) "
        "+ eta_LTP*PF(t)*theta_LTP(Ca)",
        "mechanism": "High Ca -> PKC/CaMKII -> AMPAR endocytosis -> LTD; "
        "Low Ca -> PP2B -> LTP",
        "eligibility_window_ms": [100, 200],
        "rl_interpretation": "CF = RPE delta(t); PF activity in window = eligibility trace",
        "source": "Ito 1982; Antonietti 2016; Gao 2012",
    },
    "PF_MLI": {
        "description": "PF->stellate/basket bidirectional plasticity (CF-gated)",
        "rl_interpretation": "MLI = critic; PF-MLI plasticity updates value estimate",
        "source": "Geminiani 2024 PLOS Comp Biol",
    },
    "MF_GrC": {
        "description": "Mossy fiber -> granule cell LTP/LTD (NMDA-dependent)",
        "source": "Mapelli & D'Angelo 2007",
    },
    "MF_DCN": {
        "description": "Mossy fiber -> DCN LTP (rebound-driven: PC pause -> DCN burst -> CaMKII)",
        "rl_interpretation": "Consolidation: cortical memory -> nuclear memory transfer",
        "source": "Yamazaki & Nagao 2012",
    },
    "PC_DCN": {
        "description": "PC->DCN inhibitory plasticity (rebound-gated)",
        "source": "Aizenman & Bhatt 2005",
    },
    "IO_gap_junction": {
        "description": "Connexin-36 gap junction plasticity (activity-dependent uncoupling)",
        "role": "Shapes microzone synchrony and error-signal bandwidth",
        "source": "Lefler 2014; Turecek 2014",
    },
}

CONNECTIVITY: dict[str, dict[str, object]] = {
    "MF_to_GrC": {
        "inputs_per_GrC": [4, 5],
        "structure": "cerebellar_glomerulus",
        "shared_with": "Golgi cell dendrites in same glomerulus",
    },
    "GrC_to_PC_via_PF": {
        "synapses_per_PC": 175000,
        "PF_length_mm": [3, 5],
        "PC_trees_crossed": 500,
    },
    "CF_to_PC": {"ratio": "1:1", "contacts": [250, 1500], "location": "proximal dendrite"},
    "PC_to_DCN": {"convergence": [50, 860], "type": "inhibitory_GABA"},
    "DCN_to_VL": {"type": "excitatory_glutamate", "latency_ms": [2, 3]},
    "VL_to_M1": {"target": "layer_5_pyramidal", "latency_ms": [2, 3]},
    "M1_to_pontine": {"source": "layer_5_corticopontine", "latency_ms": 1},
    "pontine_to_MF": {"type": "excitatory_glutamate", "latency_ms": [1, 2]},
    "DCN_inhibitory_to_IO": {
        "type": "GABAergic_nucleo_olivary",
        "role": "Negative feedback: reduces CF error signal after learning",
    },
    "total_loop_latency_ms": {"latency_ms": [10, 25]},
}

MICROZONE_SCALE: dict[str, dict[str, int | str]] = {
    "minimal": {
        "GrC": 5000,
        "GoC": 50,
        "PC": 32,
        "MLI": 20,
        "DCN_proj": 10,
        "DCN_inh": 5,
        "IO": 4,
        "M1_L5": 100,
        "pontine": 50,
        "VL": 20,
        "total": "~5,291",
    },
    "standard": {
        "GrC": 100000,
        "GoC": 300,
        "PC": 400,
        "MLI": 200,
        "DCN_proj": 30,
        "DCN_inh": 15,
        "IO": 10,
        "M1_L5": 500,
        "pontine": 200,
        "VL": 100,
        "total": "~101,755",
    },
    "large": {
        "GrC": 1000000,
        "GoC": 3000,
        "PC": 5000,
        "MLI": 2000,
        "DCN_proj": 200,
        "DCN_inh": 100,
        "IO": 50,
        "M1_L5": 2000,
        "pontine": 800,
        "VL": 400,
        "total": "~1,014,550",
    },
}

