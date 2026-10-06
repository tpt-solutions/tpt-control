//! `tpt-ctrl-optimal` — infinite-horizon optimal control, from scratch.
//!
//! * [`dare::dare`] — discrete algebraic Riccati equation via the
//!   structure-preserving doubling algorithm (SDA): matrix multiplications
//!   and well-conditioned solves only, quadratic convergence, no Schur
//!   ordering, no external solvers.
//! * [`care::care`] — continuous CARE via Newton-Kleinman iteration, initialized
//!   from the DLQR gain of a fine ZOH discretization (which is provably
//!   stabilizing for the continuous plant), converging quadratically.
//! * [`lqr::lqr`] / [`lqr::dlqr`] — LQR gain computation on top of the solvers.
//! * [`kalman`] — steady-state Kalman gain as the dual DARE, and LQG/dLQG
//!   controller combining LQR with Kalman filtering (separation principle).
//!
//! # Example
//!
//! ```
//! use tpt_ctrl_optimal::lqr;
//! use tpt_ctrl_core::dmat::DMat;
//!
//! // Double integrator x'' = u, minimize int x^T I x + u^2.
//! let a = DMat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]);
//! let b = DMat::from_rows(&[&[0.0], &[1.0]]);
//! let q = DMat::identity(2);
//! let r = DMat::from_rows(&[&[1.0]]);
//! let sol = lqr(&a, &b, &q, &r).unwrap();
//! // Classic closed form: X = [[sqrt(3), 1], [1, sqrt(3)]], K = [1, sqrt(3)].
//! assert!((sol.x[(0, 0)] - 3.0f64.sqrt()).abs() < 1e-6);
//! assert!((sol.k[(0, 1)] - 3.0f64.sqrt()).abs() < 1e-6);
//! ```

#![forbid(unsafe_code)]

pub mod care;
pub mod dare;
pub mod kalman;
pub mod lqr;

pub use care::care;
pub use dare::dare;
pub use kalman::{dlqg, kalman_gain, lqg, LqgController};
pub use lqr::{dlqr, lqr, LqrSolution};
