//! Kai — Minimal AGI Core
//! 
//! A physics-wired transformer inference engine with VFE adaptive control,
//! streaming layer-by-layer execution, attractor memory, and Darwin evolution.

#![allow(dead_code)]

pub mod config;
pub mod gguf;
pub mod streaming;
pub mod vfe;
pub mod attractor;
pub mod darwin;
pub mod ollama;

// Re-exports
pub use config::{Config, PhysicsParams};
pub use gguf::{read_kv, read_tensors, build_config, Weights, LayerWeights, Tokenizer};
pub use streaming::{generate, generate_with_phys};
pub use vfe::{compute_vfe, sample_top_p, argmax, VFEState};
pub use attractor::Attractor;
pub use darwin::{DarwinConfig, Candidate, darwin_evolve, revert_source};