pub mod sparse_connectivity;
pub mod spike_propagation;

#[cfg(test)]
mod tests {
    use crate::spike_propagation::deliver_with_latency;

    #[test]
    fn delivers_spike_with_latency() {
        assert_eq!(deliver_with_latency(&[1.0, 2.5], 0.5), vec![1.5, 3.0]);
    }
}

