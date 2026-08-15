//! Mixture of Experts (DeepSeek-V2 style) — routed experts + shared experts.
//! Top-k routing with optional router bias, capacity factor, and auxiliary loss.
#![allow(dead_code)]

use crate::config::{Config, MoEConfig};
use ndarray::{Array1, Array2, ArrayView2, s};

/// Router output for a single token
#[derive(Clone, Debug)]
pub struct RouterOutput {
    pub top_k_indices: Array1<usize>,   // [top_k] expert indices
    pub top_k_weights: Array1<f32>,     // [top_k] softmax weights
    pub all_logits: Array1<f32>,        // [n_experts] pre-softmax logits
    pub aux_loss: f32,                  // load balancing aux loss
}

/// Mixture of Experts layer (SwiGLU experts)
pub struct MoELayer {
    config: MoEConfig,
    dim: usize,
    intermediate: usize,
    n_experts: usize,
    n_shared: usize,
    top_k: usize,
    capacity_factor: f32,

    // Shared experts (always active)
    shared_w1: Array2<f32>,  // [n_shared * intermediate, dim]
    shared_w2: Array2<f32>,  // [dim, n_shared * intermediate]
    shared_w3: Array2<f32>,  // [n_shared * intermediate, dim]

    // Routed experts
    expert_w1: Vec<Array2<f32>>,  // [n_experts] of [intermediate, dim]
    expert_w2: Vec<Array2<f32>>,  // [n_experts] of [dim, intermediate]
    expert_w3: Vec<Array2<f32>>,  // [n_experts] of [intermediate, dim]

    // Router
    router_weight: Array2<f32>,  // [n_experts, dim]
    router_bias: Option<Array1<f32>>,  // [n_experts]

    // Load tracking for aux loss
    expert_counts: Vec<f32>,
    total_tokens: f32,
}

impl MoELayer {
    pub fn new(config: &Config, dim: usize, intermediate: usize) -> Self {
        let moe = &config.moe;
        let n_experts = moe.n_experts;
        let n_shared = moe.n_shared;
        let top_k = moe.top_k;
        let capacity_factor = moe.capacity_factor;
        let router_bias = moe.router_bias;

        // Initialize shared experts
        let shared_w1 = Array2::zeros((n_shared * intermediate, dim));
        let shared_w2 = Array2::zeros((dim, n_shared * intermediate));
        let shared_w3 = Array2::zeros((n_shared * intermediate, dim));

        // Initialize routed experts
        let expert_w1 = vec![Array2::zeros((intermediate, dim)); n_experts];
        let expert_w2 = vec![Array2::zeros((dim, intermediate)); n_experts];
        let expert_w3 = vec![Array2::zeros((intermediate, dim)); n_experts];

        // Router: [n_experts, dim]
        let router_weight = Array2::zeros((n_experts, dim));
        let router_bias = if router_bias {
            Some(Array1::zeros(n_experts))
        } else {
            None
        };

        Self {
            config: moe.clone(),
            dim,
            intermediate,
            n_experts,
            n_shared,
            top_k,
            capacity_factor,
            router_bias,
            shared_w1,
            shared_w2,
            shared_w3,
            expert_w1,
            expert_w2,
            expert_w3,
            router_weight,
            expert_counts: vec![0.0; n_experts],
            total_tokens: 0.0,
        }
    }

    /// Initialize weights from pretrained tensors (for transplant)
    pub fn load_pretrained(&mut self, shared: (Array2<f32>, Array2<f32>, Array2<f32>),
                           experts: Vec<(Array2<f32>, Array2<f32>, Array2<f32>)>,
                           router: (Array2<f32>, Option<Array1<f32>>)) {
        let (w1, w2, w3) = shared;
        self.shared_w1 = w1;
        self.shared_w2 = w2;
        self.shared_w3 = w3;

        for (i, (w1, w2, w3)) in experts.into_iter().enumerate() {
            self.expert_w1[i] = w1;
            self.expert_w2[i] = w2;
            self.expert_w3[i] = w3;
        }

        self.router_weight = router.0;
        self.router_bias = router.1;
    }

    /// Route tokens to experts (top-k)
    pub fn route(&self, x: &ArrayView2<f32>) -> Vec<RouterOutput> {
        // x: [batch, dim]
        let batch = x.nrows();
        let mut outputs = Vec::with_capacity(batch);

        for i in 0..batch {
            let token = x.row(i); // [dim]
            let logits = self.router_weight.dot(&token); // [n_experts]
            let logits = if let Some(bias) = &self.router_bias {
                &logits + bias
            } else {
                logits
            };

            // Softmax
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_logits = logits.mapv(|x| (x - max_logit).exp());
            let sum_exp = exp_logits.sum();
            let probs = exp_logits / sum_exp;

            // Top-k
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_k = &indexed[..self.top_k];

            let sum_top = top_k.iter().map(|(_, w)| *w).sum::<f32>();

            let top_k_indices = Array1::from_iter(top_k.iter().map(|(i, _)| *i));
            let top_k_weights = Array1::from_iter(top_k.iter().map(|(_, w)| *w / sum_top));

            // Aux loss: load balancing (entropy of expert assignment)
            let uniform = 1.0 / self.n_experts as f32;
            let aux_loss = probs.iter().map(|p| p * (p / uniform).ln()).sum::<f32>();

            outputs.push(RouterOutput {
                top_k_indices,
                top_k_weights,
                all_logits: logits.to_owned(),
                aux_loss,
            });
        }

        outputs
    }

    /// Forward pass through MoE layer
    /// x: [batch, dim] -> out: [batch, dim]
    pub fn forward(&mut self, x: &ArrayView2<f32>) -> Array2<f32> {
        let batch = x.nrows();
        let mut out = Array2::zeros((batch, self.dim));

        for i in 0..batch {
            let token = x.row(i);
            let mut token_out = Array1::zeros(self.dim);

            // Shared experts
            for s in 0..self.n_shared {
                let w1 = self.shared_w1.slice(s![s * self.intermediate..(s + 1) * self.intermediate, ..]);
                let w2 = self.shared_w2.slice(s![.., s * self.intermediate..(s + 1) * self.intermediate]);
                let w3 = self.shared_w3.slice(s![s * self.intermediate..(s + 1) * self.intermediate, ..]);

                let gate = w1.dot(&token).mapv(|x| x.max(0.0));
                let up = w3.dot(&token);
                let hidden = &gate * &up;
                let expert_out = w2.dot(&hidden);
                token_out += &expert_out;
            }

            // Router
            let logits = self.router_weight.dot(&token);
            let logits = if let Some(bias) = &self.router_bias {
                &logits + bias
            } else {
                logits
            };

            // Top-k softmax
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_logits = logits.mapv(|x| (x - max_logit).exp());
            let probs = &exp_logits / exp_logits.sum();

            // Top-k
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_k = &indexed[..self.top_k];

            let sum_top = top_k.iter().map(|(_, w)| *w).sum::<f32>();

            for (e, prob) in top_k {
                let weight = prob / sum_top;
                let w1 = &self.expert_w1[*e];
                let w2 = &self.expert_w2[*e];
                let w3 = &self.expert_w3[*e];

                let gate = w1.dot(&token).mapv(|x| x.max(0.0));
                let up = w3.dot(&token);
                let hidden = &gate * &up;
                let expert_out = w2.dot(&hidden);
                token_out += &(&expert_out * weight);
            }

            out.row_mut(i).assign(&token_out);
        }

        out
    }

    /// Compute load balancing auxiliary loss
    pub fn aux_loss(&self) -> f32 {
        if self.total_tokens == 0.0 {
            return 0.0;
        }
        let target = self.total_tokens / (self.n_experts * self.top_k) as f32;
        let mut loss = 0.0;
        for &count in &self.expert_counts {
            let diff = count - target;
            loss += diff * diff;
        }
        loss / self.n_experts as f32
    }

    /// Reset load tracking
    pub fn reset_load(&mut self) {
        self.expert_counts.fill(0.0);
        self.total_tokens = 0.0;
    }
}