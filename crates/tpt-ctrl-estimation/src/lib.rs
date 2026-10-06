//! `tpt-ctrl-estimation` — stochastic state estimation, from scratch.
//!
//! * [`kf`] — discrete Kalman filter with Joseph-form covariance update.
//! * [`ekf`] — extended Kalman filter with user-provided Jacobian closures.
//! * [`ukf`] — unscented Kalman filter for strongly nonlinear observation
//!   models.
//!
//! All filters share the same shape: a stateful estimate `(x, P)` advanced
//! by `predict` / `update` steps, with free functions for one-shot use.
//!
//! # Validation plants (todo.md Phase 2 checkpoint)
//!
//! `tpt-vehicle`/`tpt-vehicle-physics` exist but are not usable as test
//! dependencies from here, so the nonlinear EKF/UKF validation uses a
//! synthetic plant: constant-velocity tracking with a **range-only**
//! (radial, nonlinear) measurement — the classic sensor-fusion geometry —
//! plus range-rate (Doppler). This exercises genuinely curved observation
//! models while keeping the ground truth analytic.

#![forbid(unsafe_code)]

pub mod ekf;
pub mod kf;
pub mod ukf;
