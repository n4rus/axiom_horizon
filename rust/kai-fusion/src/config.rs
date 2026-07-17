//! Kai-Fusion model configuration (Phase C, Slice 1: dense RoPE-MHA).

#[derive(Clone)]
pub struct Config {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize, // grouped-query; = n_heads for MHA (Slice 1)
    pub vocab_size: usize,
    pub intermediate: usize,
    pub rope_theta: f32,
    pub max_seq: usize,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }
    pub fn dim_kv(&self) -> usize {
        self.n_kv_heads * self.head_dim()
    }

    /// Estimated f32 memory in GB for all weights (embed + layers + output + norms).
    pub fn estimated_f32_gb(&self) -> f64 {
        let dk = self.dim_kv();
        let per_layer = self.dim * self.dim       // wq
            + dk * self.dim                        // wk
            + dk * self.dim                        // wv
            + self.dim * self.dim                  // wo
            + self.intermediate * self.dim          // w1 (gate)
            + self.dim * self.intermediate          // w2 (down)
            + self.intermediate * self.dim          // w3 (up)
            + self.dim                              // attn_norm
            + self.dim;                             // ffn_norm
        let total = self.vocab_size * self.dim      // embed
            + self.vocab_size * self.dim            // output
            + self.dim                              // final_norm
            + self.n_layers * per_layer;
        total as f64 * 4.0 / 1e9                    // f32 = 4 bytes
    }

    /// Tiny model for the runnable Slice-1 demo (fast, local, no weights needed).
    pub fn tiny() -> Self {
        Self {
            dim: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            vocab_size: 512,
            intermediate: 128,
            rope_theta: 10000.0,
            max_seq: 64,
        }
    }
}
