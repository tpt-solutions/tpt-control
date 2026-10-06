//! `tpt-ctrl-verify` — mathematical verification harnesses for the
//! `tpt-control` stack.
//!
//! Three layers of verification live here:
//!
//! * [`strategies`] — `proptest` generators for well-posed control inputs:
//!   stable LTI systems, symmetric positive definite matrices, bounded noise.
//! * [`lyapunov`] — numerical checks of Lyapunov conditions
//!   `V(x) > 0`, `V'(x) < 0` (the mathematical contracts from spec.txt).
//! * [`proofs`] — Kani bounded-model-checking harnesses (compile with
//!   `cargo kani` on Linux; gated behind `#[cfg(kani)]` so ordinary builds
//!   never see them).
//!
//! The [`testing`] module provides tolerance-based assertion helpers used by
//! the whole workspace's test suites.

// The `kani` cfg is set by the Kani verifier itself (Linux CI only).
#![allow(unexpected_cfgs)]

pub mod lyapunov;
pub mod proofs;
pub mod strategies;
pub mod testing;
