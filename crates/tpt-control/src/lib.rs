//! `tpt-control` — pure-Rust, AI-native control systems theory stack.
//!
//! Umbrella crate re-exporting the sub-crates behind features:
//!
//! | Feature      | Crate                | Contents                                          |
//! |--------------|----------------------|---------------------------------------------------|
//! | (always)     | [`tpt_ctrl_core`]    | state-space models, dense linalg, discretization  |
//! | `classic`    | [`classic`]          | transfer functions, PID, lead-lag, Bode, margins  |
//! | `optimal`    | [`optimal`]          | CARE/DARE, LQR/DLQR, Kalman, LQG                  |
//! | `estimation` | [`estimation`]       | KF, EKF, UKF                                      |
//! | `mpc`        | [`mpc`]              | condensed MPC, active-set QP, explicit tables     |
//! | `verify`     | [`verify`]           | proptest strategies, Lyapunov checks, Kani proofs |
//!
//! All numerical kernels are implemented in-house: no LAPACK, no external
//! QP solvers, `#![forbid(unsafe_code)]` everywhere, and the core is
//! `no_std + alloc` compatible.
//!
//! # Example
//!
//! Discretize a plant, design an LQR, and check the closed loop:
//!
//! ```
//! use tpt_control::prelude::*;
//!
//! let a = DMat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]);
//! let b = DMat::from_rows(&[&[0.0], &[1.0]]);
//! let q = DMat::identity(2);
//! let r = DMat::from_rows(&[&[1.0]]);
//! let sol = tpt_control::optimal::lqr(&a, &b, &q, &r).unwrap();
//! let cl = &a - &(&b * &sol.k);
//! assert!(tpt_control::tpt_ctrl_core::analysis::is_stable_continuous(&cl, 1e-9).unwrap());
//! ```

#![forbid(unsafe_code)]

/// Foundation: models, linalg, discretization (always available).
pub use tpt_ctrl_core;

/// Core types re-exported at the root for convenience.
pub use tpt_ctrl_core::prelude::*;

/// Classical control tools (feature `classic`).
#[cfg(feature = "classic")]
pub use tpt_ctrl_classic as classic;

/// Optimal control (feature `optimal`).
#[cfg(feature = "optimal")]
pub use tpt_ctrl_optimal as optimal;

/// State estimation (feature `estimation`).
#[cfg(feature = "estimation")]
pub use tpt_ctrl_estimation as estimation;

/// Model predictive control (feature `mpc`).
#[cfg(feature = "mpc")]
pub use tpt_ctrl_mpc as mpc;

/// Verification harnesses (feature `verify`).
#[cfg(feature = "verify")]
pub use tpt_ctrl_verify as verify;

/// Everything you usually want, in one `use`.
pub mod prelude {
    #[cfg(feature = "classic")]
    pub use tpt_ctrl_classic::{freq, leadlag, pid, tf};
    pub use tpt_ctrl_core::prelude::*;
    #[cfg(feature = "estimation")]
    pub use tpt_ctrl_estimation::{ekf, kf, ukf};
    #[cfg(feature = "mpc")]
    pub use tpt_ctrl_mpc::{unroll, ExplicitMpc, Mpc, MpcBuilder, MpcSolution};
    #[cfg(feature = "optimal")]
    pub use tpt_ctrl_optimal::{care, dare, dlqg, dlqr, kalman_gain, lqr, LqrSolution};
}
