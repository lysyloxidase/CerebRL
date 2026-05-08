from __future__ import annotations

from cerebrl_circuit import CerebelloCorticalLoop


def test_full_loop_signal_path_closes_under_twenty_five_ms() -> None:
    loop = CerebelloCorticalLoop(microzone_scale="minimal")
    output = loop.step((0.2, -0.1, 0.4), (0.3, -0.2), reward=1.0)

    assert output.path == (
        "M1_L5",
        "pontine_nuclei",
        "mossy_fibers",
        "granular_layer",
        "parallel_fibers",
        "purkinje_cells",
        "dcn",
        "vl_thalamus",
        "M1_L5",
    )
    assert output.pontine_output.relay_spikes_ms == (1.0,)
    assert output.microzone is not None
    assert output.microzone.granular.active_count > 0
    assert output.vl_output.relay_spikes_ms
    assert output.loop_latency_ms < 25.0


def test_positive_rpe_drives_io_complex_spike_and_pf_pc_ltd() -> None:
    loop = CerebelloCorticalLoop(microzone_scale="minimal")
    output = loop.step((0.1, 0.2), (0.2,), reward=1.0)

    assert output.rpe > 0.0
    assert output.io_firing_rate_hz > 1.0
    assert output.complex_spike_count > 0
    assert output.pf_pc_plasticity == "LTD"
    assert output.pf_pc_weight_delta < 0.0


def test_negative_rpe_silences_io_and_pf_pc_ltp() -> None:
    loop = CerebelloCorticalLoop(microzone_scale="minimal")
    output = loop.step((0.5, 0.4), (0.1,), reward=-1.0)

    assert output.rpe < 0.0
    assert output.io_firing_rate_hz == 0.0
    assert output.complex_spike_count == 0
    assert output.pf_pc_plasticity == "LTP"
    assert output.pf_pc_weight_delta > 0.0


def test_dcn_output_modulates_m1_activity_and_action() -> None:
    loop = CerebelloCorticalLoop(microzone_scale="minimal")
    output = loop.step((0.2, 0.2), (0.25, -0.1), reward=0.8)

    assert output.cerebellar_correction != (0.0, 0.0)
    assert output.corrected_action != output.uncorrected_action
    assert output.m1_feedback_rate_hz > output.m1_output.corticospinal_rate_hz


def test_dcn_to_io_feedback_reduces_io_rate_with_repeated_training() -> None:
    loop = CerebelloCorticalLoop(microzone_scale="minimal")
    first = loop.step((0.2, 0.1), (0.2,), reward=1.0)
    second = loop.step((0.2, 0.1), (0.2,), reward=1.0)
    third = loop.step((0.2, 0.1), (0.2,), reward=1.0)

    assert first.io_firing_rate_hz > second.io_firing_rate_hz > third.io_firing_rate_hz
    assert loop.io_feedback_adaptation_hz > 0.0


def test_loop_latency_matches_biological_estimate() -> None:
    loop = CerebelloCorticalLoop(microzone_scale="minimal")
    output = loop.step((0.1, 0.2), (0.2,), reward=0.0)

    assert 10.0 <= output.loop_latency_ms <= 25.0
    assert output.latency_by_segment_ms["m1_to_pontine"] == 1.0
    assert output.latency_by_segment_ms["dcn_to_vl_to_m1"] == 5.5


def test_removing_cerebellum_leaves_motor_output_uncorrected() -> None:
    loop = CerebelloCorticalLoop(microzone_scale="minimal")
    corrected = loop.step((0.2, -0.1), (0.3, -0.2), reward=1.0)
    uncorrected = loop.step((0.2, -0.1), (0.3, -0.2), reward=1.0, cerebellum_enabled=False)

    assert corrected.corrected_action != corrected.uncorrected_action
    assert uncorrected.corrected_action == uncorrected.uncorrected_action
    assert uncorrected.dysmetria_index == 0.0

