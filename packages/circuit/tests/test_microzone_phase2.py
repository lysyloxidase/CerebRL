from __future__ import annotations

from cerebrl_circuit import CerebellarMicrozone
from cerebrl_circuit.dcn_module import DCNModule
from cerebrl_circuit.granular_layer import GranularLayer
from cerebrl_circuit.inferior_olive_module import InferiorOliveModule


def test_microzone_assembles_at_minimal_scale() -> None:
    microzone = CerebellarMicrozone(scale="minimal")

    assert microzone.scale.granule == 5000
    assert microzone.scale.purkinje == 32
    assert microzone.scale.total == 5291
    assert microzone.plasticity_site_names() == (
        "PF_PC_LTD_LTP",
        "PF_MLI",
        "MF_GrC",
        "MF_DCN",
        "PC_DCN",
        "IO_gap_junction",
    )


def test_granule_population_sparsity_is_one_to_five_percent() -> None:
    granular = GranularLayer(granule_count=5000, golgi_count=50)
    output = granular.encode((25.0, 50.0, 10.0, 80.0, 30.0))

    assert 0.01 <= output.active_fraction <= 0.05
    assert output.active_count == round(5000 * output.active_fraction)
    assert len(granular.mossy_inputs_for_granule(0)) == 4
    assert len(granular.mossy_inputs_for_granule(1)) == 5


def test_pf_drive_produces_biological_purkinje_simple_spikes() -> None:
    microzone = CerebellarMicrozone(scale="minimal")
    result = microzone.step((30.0, 60.0, 20.0, 75.0, 15.0), duration_ms=1000.0)

    assert 30.0 <= result.purkinje.simple_spike_rate_hz <= 70.0


def test_single_climbing_fiber_spike_produces_complex_spike_burst_and_pause() -> None:
    microzone = CerebellarMicrozone(scale="minimal")
    result = microzone.step(
        (30.0, 60.0, 20.0, 75.0, 15.0),
        climbing_fiber_events_ms=(100.0,),
        duration_ms=300.0,
    )

    assert result.purkinje.complex_event_times_ms == (100.0,)
    assert result.purkinje.complex_burst_spikes_ms == (100.0, 101.2, 102.4)
    assert result.purkinje.pause_windows_ms == ((100.0, 115.0),)
    assert all(not 100.0 <= spike <= 115.0 for spike in result.purkinje.simple_spike_times_ms)


def test_tonic_purkinje_inhibition_suppresses_dcn_projection_activity() -> None:
    dcn = DCNModule(projection_count=10, inhibitory_count=5)
    disinhibited = dcn.response_to_pc_rate(0.0, mf_collateral_drive=0.4)
    tonically_inhibited = dcn.response_to_pc_rate(60.0, mf_collateral_drive=0.4)

    assert tonically_inhibited.projection_rate_hz < disinhibited.projection_rate_hz
    assert tonically_inhibited.pc_inhibition_fraction > 0.5


def test_purkinje_pause_releases_dcn_rebound_burst() -> None:
    dcn = DCNModule(projection_count=10, inhibitory_count=5)
    output = dcn.response_to_pc_rate(
        55.0,
        mf_collateral_drive=0.2,
        pc_pause_windows_ms=((100.0, 250.0),),
        duration_ms=400.0,
    )

    assert output.rebound_spikes_ms[0] <= 260.0
    assert output.rebound_rate_hz >= 120.0
    assert output.motor_command > 0.0


def test_io_gap_junctions_synchronize_neighboring_oscillators_at_six_hz() -> None:
    io = InferiorOliveModule(neuron_count=4)
    output = io.oscillate(duration_ms=1000.0)

    assert 5.5 <= output.sto_frequency_hz <= 6.5
    assert output.synchrony_index >= 0.95
    assert output.phase_offsets_rad == (0.0, 0.0, 0.0, 0.0)


def test_dcn_to_io_feedback_reduces_climbing_fiber_firing() -> None:
    io = InferiorOliveModule(neuron_count=4)
    uninhibited = io.broadcast_error(1.0, dcn_inhibitory_rate_hz=0.0)
    inhibited = io.broadcast_error(1.0, dcn_inhibitory_rate_hz=100.0)

    assert inhibited.firing_rate_hz < uninhibited.firing_rate_hz
    assert inhibited.climbing_fiber_spikes_ms != uninhibited.climbing_fiber_spikes_ms

