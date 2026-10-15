# tpt-control — Phased Build Tracker

Companion to `spec.txt`. License: dual MIT OR Apache-2.0. Author/org: TPT Solutions.

## Phase 0 — Foundation & Repo Setup

- [x] Workspace scaffold: root `Cargo.toml` with `[workspace]`, `resolver = "2"`, `[workspace.package]` (`version`, `edition = "2021"`, `authors = ["TPT Solutions"]`, `license = "MIT OR Apache-2.0"`, `repository`)
- [x] Create `crates/` layout per spec.txt section 3: `tpt-ctrl-core`, `tpt-ctrl-classic`, `tpt-ctrl-optimal`, `tpt-ctrl-estimation`, `tpt-ctrl-mpc`, `tpt-ctrl-verify`, `tpt-control` (umbrella)
- [x] Add `LICENSE-MIT` and `LICENSE-APACHE` (TPT Solutions boilerplate, mirror `tpt-dsp`); set `license.workspace = true` in every member crate
- [x] Add `deny.toml` modeled on `tpt-dsp`/`tpt-math` (permissive-license allowlist; deny unknown registry/git; deny Apache-2.0-only crates such as `nalgebra`, `clarabel`, `faer`, `statrs` per ecosystem ADR-0007 precedent)
- [x] CI: `.github/workflows/ci.yml` (build + test + clippy + fmt + cargo-deny) plus a non-blocking `.github/workflows/kani.yml`
- [x] `rustfmt.toml`, `rust-toolchain.toml` (pinned `1.97.1`), `justfile` task runner (mirror `tpt-dsp` conventions)
- [x] Add `tpt-math` as a path dependency; confirm which primitives are usable today (LU solve/inverse, fixed/dynamic matrices)
  - **Outcome:** `tpt-math-linalg-dense` offers DMatrix/DVector arithmetic plus f64-only LU solve/inverse. That is a strict subset of what `tpt-ctrl-core` needs, so the decision was **implement kernels locally in `tpt-ctrl-core`** and keep the build free of cross-workspace path dependencies (a sibling path dep would break fresh-clone CI and `cargo publish`). Revisit contributing expm/Cholesky/Schur upstream to `tpt-math-linalg-*` once APIs stabilize.
- [x] **Checkpoint:** `tpt-math` currently has no matrix exponential (Padé/scaling-squaring), no eigenvalue solver, no QR/SVD/Cholesky/Schur decomposition. Decide: implement locally in `tpt-ctrl-core`, or contribute upstream to `tpt-math-linalg-dense`/`-complex` first, before ZOH/Tustin discretization can be completed from scratch
  - **Decision: implemented locally** (see previous item). All kernels live in `tpt-ctrl-core` with proptest/numerical verification.
- [x] `tpt-ctrl-core`: `StateSpace<N, M, P>` const-generic type (N states, M inputs, P outputs), continuous + discrete LTI/LTV model structs
- [x] `tpt-ctrl-core`: `no_std` + `alloc` compatibility (CI builds `thumbv6m-none-eabi`; float kernels dispatch to `libm` when `std` is off)
- [x] `tpt-ctrl-core`: exact Zero-Order Hold (ZOH) discretization from scratch (Van Loan block exponential)
- [x] `tpt-ctrl-core`: Tustin (bilinear) discretization from scratch
- [x] `tpt-ctrl-verify`: scaffold crate, set up `proptest` harness for basic matrix operations
- [x] `tpt-ctrl-verify`: set up first Kani harness in the tpt-* ecosystem (no local prior art exists yet — expect extra setup time)
  - **Note:** harnesses compile under `#[cfg(kani)]` (invisible to normal builds); executed by the Linux-only `kani.yml` workflow.

## Phase 1 — Classical & Optimal Control

- [x] `tpt-ctrl-classic`: PID controller with anti-windup (back-calculation)
- [x] **Checkpoint:** review `tpt-dsp-control` (existing PID/kinematics/motor-control subcrate in `tpt-dsp`) before implementing PID — decide whether to depend on it, fork logic, or explicitly scope `tpt-ctrl-classic`'s PID as state-space-integrated control-theory PID vs. tpt-dsp's embedded/motor-control PID, to avoid duplicated maintenance
  - **Decision: scoped separately.** `tpt-ctrl-classic::pid::Pid` is the control-theory PID (error-in/out, configurable saturation, anti-windup, derivative filtering, composable with TF algebra); `tpt-dsp-control` keeps the embedded/motor-control PID. Rationale documented in the `pid` module docs.
- [x] `tpt-ctrl-classic`: lead-lag compensators, transfer function algebra (+ notch sections, poles/zeros via companion-matrix eigenvalues)
- [x] **Checkpoint:** confirm `tpt-dsp-core`'s FFT/filter API surface is stable enough to depend on before starting frequency-response work
  - **Outcome:** not needed for the implemented scope — margins are computed by direct evaluation on a log grid (exact for rational TFs), so `tpt-ctrl-classic` has no `tpt-dsp` dependency. Revisit FFT-based impulse-response identification only if measured-data frequency response becomes a requirement.
- [x] `tpt-ctrl-classic`: add `tpt-dsp-core` dependency; Bode/Nyquist plots and gain/phase margins via native Rust FFT
  - **Adjusted scope:** Bode/Nyquist data and margins implemented via direct complex evaluation + phase unwrapping (validated against textbook values: 1/(s(s+1)) PM ≈ 51.8°; 4/(s+1)³ GM = 2). No FFT dependency.
- [x] `tpt-ctrl-optimal`: from-scratch CARE/DARE solvers (Newton-Kleinman iteration or structure-preserving Schur decomposition) — likely the single largest task in this phase since Schur/eigenvalue solving doesn't exist in `tpt-math` yet
  - **Delivered:** DARE via structured doubling (SDA); CARE via Newton–Kleinman initialized from the DLQR gain of a fine ZOH discretization (provably stabilizing); both cross-checked against slow fixed-point iterations and closed forms.
- [x] `tpt-ctrl-optimal`: LQR gain computation (LQR/DLQR)
- [x] `tpt-ctrl-optimal`: LQG separation principle (LQR + Kalman filter, depends on Phase 2's KF)
  - **Delivered:** steady-state Kalman gain as the dual DARE (innovation gain `P Cᵀ(CPCᵀ+R)⁻¹`), `lqg` + `dlqg`.
- [x] Integration test: LQR on an inverted-pendulum model, verified via `proptest` for closed-loop stability (`crates/tpt-ctrl-optimal/tests/inverted_pendulum.rs`)

## Phase 2 — Estimation & MPC

- [x] `tpt-ctrl-estimation`: discrete Kalman Filter (KF, Joseph-form update)
- [x] `tpt-ctrl-estimation`: Extended Kalman Filter (EKF) with user-provided Jacobian closures
- [x] `tpt-ctrl-estimation`: Unscented Kalman Filter (UKF) for nonlinear observation models
- [x] **Checkpoint:** identify a real nonlinear plant model to validate EKF/UKF against — `tpt-vehicle`/`tpt-vehicle-physics` is the closest existing analog to spec.txt's "tpt-multibody-dynamics"; confirm whether it's usable or whether a synthetic test plant is needed instead
  - **Decision: synthetic plant.** `tpt-vehicle` is not usable as a cross-repo test dependency, so EKF/UKF validate against a range-only (radial) tracking plant and a coordinated-turn polar-measurement plant — genuinely nonlinear geometry with analytic ground truth (documented in the `tpt-ctrl-estimation` crate docs).
- [x] `tpt-ctrl-mpc`: linear MPC formulation with explicit state/input constraint matrix generation
- [x] `tpt-ctrl-mpc`: horizon unrolling (`unroll::unroll`, verified against manual propagation)
- [x] `tpt-ctrl-mpc`: from-scratch active-set (or interior-point) QP solver tailored to MPC's banded/structured matrices
  - **Delivered:** active-set with exact KKT sub-solves, add-before-drop pivoting, and a dependency (parallel-constraint) fallback; validated against analytic solutions and KKT conditions.
- [x] `tpt-ctrl-verify`: Kani harness verifying horizon-unrolling correctness (`horizon_unroll_no_panic`: finiteness + block-lower-triangular structure of `Su`)

## Phase 3 — Explicit MPC & Verification Hardening

- [x] `tpt-ctrl-mpc`: explicit MPC (mpc-MP) lookup table generation for low-latency deployment
  - **Delivered:** grid-sampled nearest-neighbour tables (`ExplicitMpc`), exact at nodes, clamped outside the box; the docs state the piecewise-constant (vs critical-region piecewise-affine) trade-off honestly. Upgrade path to true mpcol-style regions noted.
- [x] `tpt-ctrl-verify`: Lyapunov stability helper traits/checks (V(x) > 0, V̇(x) < 0) as runtime debug-assertions and as Kani proofs
  - **Delivered:** sampled Lyapunov contract checks (`check_lyapunov_contract`) on a damped-oscillator reference; Kani harnesses prove no-panic/finiteness invariants on the underlying kernels.
- [x] `tpt-ctrl-verify`: proptest strategies for generating valid/stable LTI systems and bounded noise profiles (broaden beyond Phase 0's basic matrix-op harness)
  - **Delivered:** SPD via `MᵀM + I`, Hurwitz via spectral shift, Schur via `Q D Qᵀ`, LTI systems, bounded noise, plus an eigenvalue-vs-trace/det invariant proptest.
- [x] `tpt-ctrl-verify`: Kani proofs of no-panic (division by zero, out-of-bounds) across CARE/DARE, QP, and matrix-exponential solver loops
  - **Delivered:** `lu_solve_2x2_no_panic`, `cholesky_2x2_reconstruction`, `zoh_2x2_no_panic`, `horizon_unroll_no_panic`, `dare_2x2_no_panic`, `qp_2d_no_panic`, `dimensioned_matmul_3x2_times_2x2_no_panic`.
- [x] Differentiability hooks: automatic differentiation or finite-difference fallback on cost functions and state transitions (spec.txt "AI-Native Agents" integration point)
  - **Delivered:** `tpt-ctrl-core::fd` central-difference Jacobian/gradient hooks; documented as the portable fallback for exact AD from `tpt-math-autodiff-*`.
- [x] **Checkpoint:** `tpt-ai`/`tpt-infer`/`tpt-pantheon`/`tpt-archon` are currently spec-only stubs in the ecosystem — treat differentiability hooks as forward-looking API design, not an integration test, until one of those crates is actually scaffolded
  - **Status:** confirmed — hooks shipped as forward-looking API only.

## Phase 4 — Umbrella Crate, Docs & Examples

- [x] `tpt-control` umbrella crate: feature-gated re-exports of all sub-crates (`classic`, `optimal`, `estimation`, `mpc`, `verify` features)
- [x] Top-level `README.md`; per-crate README/API docs; doctests for public API surface
- [x] Worked example: inverted pendulum with LQR (reuse Phase 1's integration test) — `crates/tpt-ctrl-optimal/examples/inverted_pendulum.rs`
- [x] Worked example: KF/EKF sensor-fusion — `crates/tpt-ctrl-estimation/examples/kf_sensor_fusion.rs`
- [x] Worked example: linear MPC with constraints — `crates/tpt-ctrl-mpc/examples/constrained_mpc.rs`
- [x] Benchmarks (`criterion`, matching `tpt-math`/`tpt-dsp` convention) for solver hot paths (Riccati iteration, QP solve, matrix exponential)

## Phase 5 — Release v1.0

- [x] Full `cargo-deny` audit pass across all sub-crates (advisories/bans/licenses/sources all ok)
- [x] `CHANGELOG.md`; align versions across all workspace members (all `0.1.0` via `[workspace.package]`)
- [x] Confirm dual LICENSE files and `authors = ["TPT Solutions"]` consistent across every crate (matches `tpt-dsp`/`tpt-cadence`/`tpt-pattern` convention)
- [ ] Decide publish target: crates.io vs. internal/path-dependency-only (confirm which siblings actually publish) — **open decision for the repo owner**
- [ ] Tag and publish v1.0.0 (blocked on the decision above)

## Phase 6 — Adoption, Tooling & Innovation (post-review)

- [x] Python bindings (PyO3 + maturin)
  - **Done:** `bindings/python` (PyO3 0.30, excluded from the Rust workspace). Exposes `lqr_gain`, `dlqr_gain`, `certify`, `c2d`, `tf2ss`, `eigenvalues`. 9/9 pytest pass against a maturin wheel. Not yet published to PyPI.: separate `bindings/python` crate, excluded from the Rust workspace so the core gate stays Python-free; exposes LQR/DLQR, c2d, tf2ss, and certificates as `tpt_control`
- [x] MATLAB migration layer
  - **Done:** `tpt_control::matlab` (`tf2ss`, `c2d`, `lqr`, `dlqr`, `kalman`, MATLAB output order) with unit tests; cheat sheet in `docs/MATLAB.md`.: `tpt_control::matlab` aliases (`tf2ss`, `c2d`, `lqr`, `dlqr`, `kalman`) plus a one-page MATLAB → tpt-control cheat sheet (`docs/MATLAB.md`)
- [x] `tpt` CLI
  - **Done:** `crates/tpt-control-cli`. `tpt lqr plant.toml` exits 0 when certified, 1 when the certificate fails, 2 on bad input or an unstabilizable plant. Sample plants in `plants/`; 5 integration tests.: `crates/tpt-control-cli`, reads a TOML plant description and prints gains, closed-loop poles and the certificate
- [x] WASM playground
  - **Done:** `crates/tpt-control-wasm` (wasm-bindgen `=0.2.128`), `playground/index.html`, `just playground`. Verified via the Node target (`design_lqr` returns certified gains and typed errors). Not yet checked in a real browser.: `crates/tpt-control-wasm` (wasm-bindgen) plus a static `playground/index.html` that runs LQR in the browser
- [x] Embedded export
  - **Done:** `ExplicitMpc::to_c_header` and `to_rust_source` emit tables plus a nearest-neighbour lookup. Rust output is tested. **The C header is not compiled-tested**: no C compiler is installed on this machine.: generate a C header and Rust `const` source from an `ExplicitMpc` table, including a lookup routine
- [x] Self-certifying solves
  - **Done:** `tpt_ctrl_optimal::certify` (`certify_continuous`, `certify_discrete`) with fail-closed handling of non-finite inputs. While building it, found and fixed a real bug: the DARE convergence check passed NaN results, so unstabilizable plants returned NaN gains. Solvers now return `LinalgError::NotStabilizable`.: `tpt_ctrl_optimal::certify` returns Riccati residual, closed-loop stability margin and a pass/fail verdict alongside each gain
- [x] CI hardening
  - **Done (unverified on GitHub):** `wasm` job added to `ci.yml`; `kani.yml` gains a weekly schedule and manual dispatch, still non-blocking. Local equivalents pass; the hosted runs have not been observed.: wasm build job in `ci.yml`; scheduled weekly Kani run in `kani.yml` (still non-blocking)


## Gate status

`just ci` equivalent: fmt clean, `--all-features` build clean, 138 tests +
doctests passing, clippy `-D warnings` clean, cargo-deny clean, rustdoc
warning-free, `thumbv6m-none-eabi` no_std build clean.
