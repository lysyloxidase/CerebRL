#[derive(Debug, Clone, PartialEq)]
pub struct CsrConnectivity {
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub weights: Vec<f32>,
}

impl CsrConnectivity {
    pub fn edge_count(&self) -> usize {
        self.indices.len()
    }
}

