pub fn deliver_with_latency(spikes_ms: &[f64], latency_ms: f64) -> Vec<f64> {
    spikes_ms.iter().map(|spike| spike + latency_ms).collect()
}

