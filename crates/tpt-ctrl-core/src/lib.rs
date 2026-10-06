//! `tpt-ctrl-core` — the foundation of the `tpt-control` stack.
//!
//! A pure-Rust, `no_std` + `alloc` compatible control-systems foundation:
//! state-space models with compile-time dimensions, from-scratch dense
//! linear algebra (LU, Cholesky, QR, real Schur, Pade matrix exponential,
//! Lyapunov solvers), and exact continuous-to-discrete conversion.
//! No LAPACK, no external solvers — every numerical kernel in this crate is
//! implemented in-house and `#![forbid(unsafe_code)]` applies.
//!
//! # Layout
//!
//! * [`smat::Mat`] — compile-time dimensioned matrices (dimension mismatches
//!   are compile errors).
//! * [`dmat::DMat`] — runtime-dimensioned workhorse used by the decompositions
//!   and by the solver crates.
//! * [`state_space`] — `StateSpace<N, M, P>`, [`Continuous`], [`Discrete`],
//!   and the runtime-dimensioned [`DynSystem`].
//! * [`discretize`] — exact ZOH and Tustin discretization.
//! * [`lu`], [`cholesky`], [`qr`], [`hessenberg`], [`schur`], [`expm`],
//!   [`lyapunov`] — the numerical kernels.
//! * [`analysis`] — poles, stability, controllability, observability.
//! * [`fd`] — finite-difference Jacobian/gradient hooks for agents.
//! * [`simulate`] — RK4 and discrete propagation.
//!
//! # Example
//!
//! Discretize a continuous plant with the exact zero-order hold:
//!
//! ```
//! use tpt_ctrl_core::smat::Mat;
//! use tpt_ctrl_core::state_space::{Continuous, StateSpace};
//!
//! // Double integrator: x'' = u
//! let sys = Continuous::new(StateSpace::new(
//!     Mat::from_rows_arr(&[[0.0, 1.0], [0.0, 0.0]]),
//!     Mat::from_rows_arr(&[[0.0], [1.0]]),
//!     Mat::from_rows_arr(&[[1.0, 0.0]]),
//!     Mat::from_rows_arr(&[[0.0]]),
//! ));
//! let dt = 0.05_f64;
//! let disc = sys.discretize_zoh(dt).unwrap();
//! let ad = disc.ss.a;
//! assert!((ad[(0, 1)] - dt).abs() < 1e-12); // exact, not approximated
//! ```

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod analysis;
pub mod c64;
pub mod cholesky;
pub mod discretize;
pub mod dmat;
pub mod error;
pub mod expm;
pub mod fd;
pub mod field;
pub mod hessenberg;
pub mod lu;
pub mod lyapunov;
pub mod qr;
pub mod schur;
pub mod simulate;
pub mod smat;
pub mod state_space;

pub mod math;

pub use c64::C64;
pub use dmat::DMat;
pub use error::{LinalgError, Result};
pub use field::Field;
pub use smat::{Mat, Vector};
pub use state_space::{Continuous, Discrete, DynSystem, StateSpace};

/// Re-exports of the most-used types for ergonomic `use tpt_ctrl_core::prelude::*;`.
pub mod prelude {
    pub use crate::c64::C64;
    pub use crate::dmat::DMat;
    pub use crate::error::{LinalgError, Result};
    pub use crate::field::Field;
    pub use crate::smat::{Mat, Vector};
    pub use crate::state_space::{Continuous, Discrete, DynSystem, StateSpace};
}
