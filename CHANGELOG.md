# Changelog

All notable changes to the `tpt-control` workspace are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Every crate now ships its own `README.md` (features, install line, a working example, links to runnable examples) and `CHANGELOG.md`, and sets `readme` in its manifest.
- Runnable examples: `tpt-ctrl-core/discretize_and_simulate`, `tpt-ctrl-classic/pid_step_response`, `tpt-ctrl-classic/bode_margins`, `tpt-ctrl-optimal/lqg_regulator`, `tpt-ctrl-verify/lyapunov_contract`, and the umbrella `tpt-control/quickstart`.
- `tpt_ctrl_optimal::kalman_gain_continuous` (steady-state Kalman gain for continuous-time models, from the dual CARE).

### Fixed
- `tpt_ctrl_optimal::lqg` (continuous-time) took its observer gain from the discrete DARE, which is wrong for a continuous plant and could give an unstable observer. It now uses `kalman_gain_continuous`.

### Changed
- `tpt-ctrl-mpc` keyword `quadratic-programming` (21 characters, over the crates.io limit of 20) is replaced by `optimization`; its category `concurrency` is replaced by `science::robotics`.

## [Unreleased]

### Added
- `tpt-ctrl-optimal::certify`: `certify_continuous` / `certify_discrete` return a
  `Certificate` (relative Riccati residual, closed-loop margin, verdict). Non-finite inputs fail closed.
- `tpt_control::matlab`: MATLAB-named wrappers (`tf2ss`, `c2d`, `lqr`, `dlqr`, `kalman`); cheat sheet in `docs/MATLAB.md`.
- `ExplicitMpc::refine` (exact online-solve fallback), `ExplicitMpc::to_c_header`, `ExplicitMpc::to_rust_source`.
- `tpt-control-cli`: the `tpt` command (`tpt lqr plant.toml`) with CI-friendly exit codes.
- `tpt-control-wasm` and `playground/`: in-browser LQR design (`just playground`).
- `bindings/python`: PyO3 package `tpt_control` (excluded from the Rust workspace).
- CI: `wasm` build job; weekly scheduled Kani run.

### Fixed
- DARE/CARE solvers returned NaN gains for unstabilizable plants instead of an error. They now return `LinalgError::NotStabilizable`.
- `just kani` ran a nonexistent `verify` harness; it now runs the crate's proofs.
- Removed a leftover debug example (`tpt-ctrl-optimal/examples/dbg.rs`).

## [0.1.0] — 2026-10-06

Initial workspace build: Phases 0–4 of `todo.md`.

### Added

#### tpt-ctrl-core (`no_std` + `alloc`)
- Const-generic dimensioned matrices (`Mat<T, R, C>`) and the runtime
  workhorse `DMat<T>` with the full operator set.
- From-scratch factorizations: LU with partial pivoting, Cholesky,
  Householder QR, Hessenberg reduction, real Schur via Francis
  double-shift QR (with robust 2x2 block splitting).
- Pade-13 scaling-and-squaring matrix exponential.
- Lyapunov solvers: continuous (Bartels–Stewart over real Schur) and
  discrete (squaring/doubling).
- `StateSpace<T, N, M, P>` with continuous/discrete wrappers; exact
  zero-order-hold and Tustin discretization; RK4 and discrete simulation;
  poles/stability/controllability/observability analysis.
- Finite-difference Jacobian/gradient hooks for agent integration.

#### tpt-ctrl-classic
- Rational transfer functions with companion-matrix poles/zeros, series /
  parallel / closed-loop algebra, controllable-canonical realization.
- PID with back-calculation anti-windup, derivative-on-measurement and
  derivative filtering.
- Lead / lag / notch compensator constructors (unity DC gain).
- Frequency response, Bode and Nyquist data, gain/phase margins on a
  logarithmic grid with crossover interpolation.

#### tpt-ctrl-optimal
- DARE via the structure-preserving doubling algorithm.
- CARE via Newton–Kleinman with a provably stabilizing initial gain from
  a fine ZOH discretization.
- LQR/DLQR, steady-state Kalman gain (dual DARE), LQG and discrete LQG.

#### tpt-ctrl-estimation
- Discrete Kalman filter with Joseph-form covariance update.
- EKF driven by user-provided model + Jacobian closures.
- UKF with corrected Wan–van der Merwe weights and Cholesky sigma points.

#### tpt-ctrl-mpc
- Strictly convex active-set QP solver (exact KKT sub-solves,
  add-before-drop pivoting, dependency fallback).
- Horizon condensing (`Sx`, `Su`), condensed MPC with input bounds and
  linear state constraints, receding-horizon controller.
- Explicit-MPC nearest-neighbour lookup tables.

#### tpt-ctrl-verify
- proptest strategies: SPD matrices, Hurwitz and Schur matrices, LTI
  systems, bounded noise.
- Lyapunov contract checking over sampled boxes.
- Kani bounded-model-checking harnesses (LU, Cholesky, ZOH, unrolling,
  DARE, QP) under `#[cfg(kani)]`; first Kani use in the tpt-* ecosystem.

#### Workspace
- Umbrella crate `tpt-control` with feature-gated re-exports.
- Worked examples: inverted pendulum LQR, KF sensor fusion, constrained MPC.
- Criterion benchmarks for expm/Schur/solve/Lyapunov/DARE/QP/MPC step.
- CI (fmt, clippy, nextest, doctests, no_std build, cargo-deny) and a
  non-blocking Kani workflow; `justfile`; pinned toolchain; `deny.toml`
  with a permissive-only license allowlist.

[0.1.0]: https://github.com/tpt-solutions/tpt-control/releases/tag/v0.1.0
