from __future__ import annotations

from math import isclose

from cerebrl_neurons import (
    DCNProjectionNeuron,
    GolgiCell,
    GranuleCell,
    InferiorOliveNetwork,
    InferiorOliveNeuron,
    M1Layer5PyramidalNeuron,
    PontineRelayNeuron,
    PurkinjeCell,
    StellateBasketCell,
    UnipolarBrushCell,
    VLThalamicRelayNeuron,
)


def test_purkinje_simple_and_complex_spikes() -> None:
    trace = PurkinjeCell().simulate(duration_ms=1000.0, climbing_fiber_times_ms=(500.0,))

    assert 30.0 <= trace.firing_rate_hz(label="simple") <= 70.0
    assert trace.labels["complex"] == (500.0,)
    assert trace.labels["complex_burst_spikes"] == (500.0, 501.2, 502.4)


def test_granule_biophysics_and_burst_capability() -> None:
    cell = GranuleCell()
    assert isclose(cell.membrane_capacitance_pF, 3.0)
    assert cell.input_resistance_GOhm > 1.0

    burst = cell.simulate(duration_ms=10.0, mossy_fiber_current_pA=25.0)
    assert burst.firing_rate_hz(label="burst") >= 800.0


def test_golgi_pacemaker_is_in_biological_range() -> None:
    trace = GolgiCell().simulate(duration_ms=2000.0)
    assert 3.0 <= trace.firing_rate_hz(label="pacemaker") <= 8.0


def test_stellate_basket_spontaneous_range() -> None:
    trace = StellateBasketCell(subtype="basket").simulate(duration_ms=1000.0)
    assert 1.0 <= trace.firing_rate_hz(label="basket") <= 35.0


def test_ubc_slow_mossy_fiber_amplification() -> None:
    trace = UnipolarBrushCell().simulate(duration_ms=400.0, mossy_fiber_event_times_ms=(50.0,))
    assert len(trace.labels["slow_mossy_response"]) > 1
    assert trace.labels["slow_mossy_response"][0] > 50.0


def test_dcn_rebound_after_purkinje_inhibition_release() -> None:
    trace = DCNProjectionNeuron().simulate_pc_release_rebound(
        inhibition_start_ms=100.0,
        inhibition_end_ms=250.0,
        duration_ms=400.0,
    )
    first_rebound = trace.first_spike_after(250.0, label="rebound")
    assert first_rebound is not None
    assert 250.0 < first_rebound <= 260.0
    assert trace.firing_rate_hz(start_ms=250.0, end_ms=285.0, label="rebound") >= 120.0


def test_inferior_olive_sto_and_gap_junctions() -> None:
    io = InferiorOliveNeuron()
    trace = io.simulate(duration_ms=1000.0)

    assert 4.0 <= trace.labels["sto_frequency_hz"][0] <= 12.0
    assert trace.firing_rate_hz(label="climbing_fiber") == 1.0
    assert io.gap_junction_current_pA(own_voltage_mV=-65.0, neighbor_voltage_mV=-60.0) > 0.0

    network = InferiorOliveNetwork(neurons=(io, InferiorOliveNeuron()))
    currents = network.gap_currents_pA((-65.0, -60.0))
    assert currents[0] == -currents[1]


def test_m1_layer5_rates_match_quiet_and_movement_states() -> None:
    m1 = M1Layer5PyramidalNeuron()
    quiet = m1.simulate(duration_ms=10000.0, state="quiet")
    movement = m1.simulate(duration_ms=10000.0, state="movement")

    assert isclose(quiet.firing_rate_hz(label="quiet"), 5.6, abs_tol=0.1)
    assert isclose(movement.firing_rate_hz(label="movement"), 12.7, abs_tol=0.1)


def test_pontine_and_vl_relay_latencies() -> None:
    pontine = PontineRelayNeuron()
    vl = VLThalamicRelayNeuron()

    assert pontine.relay((10.0, 20.0)) == (11.0, 21.0)
    assert vl.relay_to_m1((10.0,)) == (15.5,)

    vl_trace = vl.simulate(duration_ms=60.0, dcn_spike_times_ms=(10.0,), rebound_burst=True)
    assert vl_trace.labels["relay_to_m1"] == (15.5,)
    assert len(vl_trace.labels["burst"]) > 0

