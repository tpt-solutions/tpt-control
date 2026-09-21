# tpt-control — Phased Build Tracker

Companion to `spec.txt`. License: dual MIT OR Apache-2.0. Author/org: TPT Solutions.

## Phase 0 — Foundation & Repo Setup

- [ ] Workspace scaffold: root `Cargo.toml` with `[workspace]`, `resolver = "2"`, `[workspace.package]` (`version`, `edition = "2021"`, `authors = ["TPT Solutions"]`, `license = "MIT OR Apache-2.0"`, `repository`)
- [ ] Create `crates/` layout per spec.txt section 3: `tpt-ctrl-core`, `tpt-ctrl-classic`, `tpt-ctrl-optimal`, `tpt-ctrl-estimation`, `tpt-ctrl-mpc`, `tpt-ctrl-verify`, `tpt-control` (umbrella)
- [ ] Add `LICENSE-MIT` and `LICENSE-APACHE` (TPT Solutions boilerplate, mirror `tpt-dsp`); set `license.workspace = true` in every member crate
- [ ] Add `deny.toml` modeled on `tpt-dsp`/`tpt-math` (permissive-license allowlist; deny unknown registry/git; deny Apache-2.0-only crates such as `nalgebra`, `clarabel`, `faer`, `statrs` per ecosystem ADR-0007 precedent)
- [ ] CI: `.github/workflows/ci.yml` (build + test + clippy + fmt + cargo-deny)
- [ ] `rustfmt.toml`, `rust-toolchain.toml` (pin version), `justfile` task runner (mirror `tpt-dsp` conventions)
- [ ] Add `tpt-math` as a path dependency; confirm which primitives are usable today (LU solve/inverse, fixed/dynamic matrices)
- [ ] **Checkpoint:** `tpt-math` currently has no matrix exponential (Padé/scaling-squaring), no eigenvalue solver, no QR/SVD/Cholesky/Schur decomposition. Decide: implement locally in `tpt-ctrl-core`, or contribute upstream to `tpt-math-linalg-dense`/`-complex` first, before ZOH/Tustin discretization can be completed from scratch
- [ ] `tpt-ctrl-core`: `StateSpace<N, M, P>` const-generic type (N states, M inputs, P outputs), continuous + discrete LTI/LTV model structs
- [ ] `tpt-ctrl-core`: `no_std` + `alloc` compatibility
- [ ] `tpt-ctrl-core`: exact Zero-Order Hold (ZOH) discretization from scratch
- [ ] `tpt-ctrl-core`: Tustin (bilinear) discretization from scratch
- [ ] `tpt-ctrl-verify`: scaffold crate, set up `proptest` harness for basic matrix operations
- [ ] `tpt-ctrl-verify`: set up first Kani harness in the tpt-* ecosystem (no local prior art exists yet — expect extra setup time)

## Phase 1 — Classical & Optimal Control

- [ ] `tpt-ctrl-classic`: PID controller with anti-windup (back-calculation)
- [ ] **Checkpoint:** review `tpt-dsp-control` (existing PID/kinematics/motor-control subcrate in `tpt-dsp`) before implementing PID — decide whether to depend on it, fork logic, or explicitly scope `tpt-ctrl-classic`'s PID as state-space-integrated control-theory PID vs. tpt-dsp's embedded/motor-control PID, to avoid duplicated maintenance
- [ ] `tpt-ctrl-classic`: lead-lag compensators, transfer function algebra
- [ ] **Checkpoint:** confirm `tpt-dsp-core`'s FFT/filter API surface is stable enough to depend on before starting frequency-response work
- [ ] `tpt-ctrl-classic`: add `tpt-dsp-core` dependency; Bode/Nyquist plots and gain/phase margins via native Rust FFT
- [ ] `tpt-ctrl-optimal`: from-scratch CARE/DARE solvers (Newton-Kleinman iteration or structure-preserving Schur decomposition) — likely the single largest task in this phase since Schur/eigenvalue solving doesn't exist in `tpt-math` yet
- [ ] `tpt-ctrl-optimal`: LQR gain computation
- [ ] `tpt-ctrl-optimal`: LQG separation principle (LQR + Kalman filter, depends on Phase 2's KF)
- [ ] Integration test: LQR on an inverted-pendulum model, verified via `proptest` for closed-loop stability

## Phase 2 — Estimation & MPC

- [ ] `tpt-ctrl-estimation`: discrete Kalman Filter (KF)
- [ ] `tpt-ctrl-estimation`: Extended Kalman Filter (EKF) with user-provided Jacobian closures
- [ ] `tpt-ctrl-estimation`: Unscented Kalman Filter (UKF) for nonlinear observation models
- [ ] **Checkpoint:** identify a real nonlinear plant model to validate EKF/UKF against — `tpt-vehicle`/`tpt-vehicle-physics` is the closest existing analog to spec.txt's "tpt-multibody-dynamics"; confirm whether it's usable or whether a synthetic test plant is needed instead
- [ ] `tpt-ctrl-mpc`: linear MPC formulation with explicit state/input constraint matrix generation
- [ ] `tpt-ctrl-mpc`: horizon unrolling
- [ ] `tpt-ctrl-mpc`: from-scratch active-set (or interior-point) QP solver tailored to MPC's banded/structured matrices
- [ ] `tpt-ctrl-verify`: Kani harness verifying horizon-unrolling correctness

## Phase 3 — Explicit MPC & Verification Hardening

- [ ] `tpt-ctrl-mpc`: explicit MPC (mpc-MP) lookup table generation for low-latency deployment
- [ ] `tpt-ctrl-verify`: Lyapunov stability helper traits/checks (V(x) > 0, V̇(x) < 0) as runtime debug-assertions and as Kani proofs
- [ ] `tpt-ctrl-verify`: proptest strategies for generating valid/stable LTI systems and bounded noise profiles (broaden beyond Phase 0's basic matrix-op harness)
- [ ] `tpt-ctrl-verify`: Kani proofs of no-panic (division by zero, out-of-bounds) across CARE/DARE, QP, and matrix-exponential solver loops
- [ ] Differentiability hooks: automatic differentiation or finite-difference fallback on cost functions and state transitions (spec.txt "AI-Native Agents" integration point)
- [ ] **Checkpoint:** `tpt-ai`/`tpt-infer`/`tpt-pantheon`/`tpt-archon` are currently spec-only stubs in the ecosystem — treat differentiability hooks as forward-looking API design, not an integration test, until one of those crates is actually scaffolded

## Phase 4 — Umbrella Crate, Docs & Examples

- [ ] `tpt-control` umbrella crate: feature-gated re-exports of all sub-crates (`classic`, `optimal`, `estimation`, `mpc`, `verify` features)
- [ ] Top-level `README.md`; per-crate README/API docs; doctests for public API surface
- [ ] Worked example: inverted pendulum with LQR (reuse Phase 1's integration test)
- [ ] Worked example: KF/EKF sensor-fusion
- [ ] Worked example: linear MPC with constraints
- [ ] Benchmarks (`criterion`, matching `tpt-math`/`tpt-dsp` convention) for solver hot paths (Riccati iteration, QP solve, matrix exponential)

## Phase 5 — Release v1.0

- [ ] Full `cargo-deny` audit pass across all sub-crates
- [ ] `CHANGELOG.md`; align versions across all workspace members
- [ ] Confirm dual LICENSE files and `authors = ["TPT Solutions"]` consistent across every crate (matches `tpt-dsp`/`tpt-cadence`/`tpt-pattern` convention)
- [ ] Decide publish target: crates.io vs. internal/path-dependency-only (confirm which siblings actually publish)
- [ ] Tag and publish v1.0.0
