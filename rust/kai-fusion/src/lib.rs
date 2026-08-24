//! Kai-Fusion library crate.
//!
//! Exposes the VFE controller (`vfe`), its attractor priors (`attractor`),
//! and scalar-curvature wiring (`curvature`) as a reusable library so the
//! external physics simulators (rust/external_tests) import the *real*
//! implementation instead of an inlined copy. The `kai` binary (src/main.rs)
//! is unchanged and still declares its own private module graph.
//!
//! Only the modules needed by external consumers are surfaced here; everything
//! else stays internal to the binary.
pub mod attractor;
pub mod companion;
pub mod curvature;
pub mod vfe;
