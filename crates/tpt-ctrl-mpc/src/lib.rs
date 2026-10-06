//! `tpt-ctrl-mpc` — constrained model predictive control, from scratch.
//!
//! * [`qp`] — strictly convex QP solver (`min ½xᵀGx + gᵀx s.t. Aᵀx ≥ b`)
//!   via a dual-feasibility active-set method with exact KKT sub-solves.
//!   No OSQP / clarabel / Apache-only dependencies.
//! * [`unroll::unroll`] — horizon condensing: predicted trajectory as an affine
//!   function of the input sequence.
//! * [`Mpc`] — condensed linear MPC with input bounds and optional
//!   inequality constraints, solved at each step (receding horizon).
//!
//! # Example
//!
//! ```
//! use tpt_ctrl_core::dmat::DMat;
//! use tpt_ctrl_mpc::MpcBuilder;
//!
//! // Double integrator, inputs bounded to |u| <= 0.5.
//! let a = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
//! let b = DMat::from_rows(&[&[0.005], &[0.1]]);
//! let mpc = MpcBuilder::new(a, b, 10)
//!     .stage_costs(&DMat::identity(2), &DMat::from_rows(&[&[1.0]]))
//!     .terminal_cost(&(&DMat::identity(2) * 5.0))
//!     .input_bounds(vec![-0.5], vec![0.5])
//!     .build();
//! let sol = mpc.solve(&DMat::from_rows(&[&[0.5], &[0.5]])).unwrap();
//! assert!(sol.u0[0].abs() <= 0.5 + 1e-9);
//! ```

#![forbid(unsafe_code)]

pub mod explicit;
pub mod mpc;
pub mod qp;
pub mod unroll;

pub use explicit::{ExplicitError, ExplicitMpc};
pub use mpc::{Mpc, MpcBuilder, MpcSolution};
pub use qp::QpSolution;
pub use unroll::{unroll, CondensedHorizon};
