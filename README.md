# tpt-control

A pure-Rust, AI-native control systems theory library. State-space modeling,
classical control, optimal control, state estimation, and model predictive
control — with **zero C/C++ FFI** and **zero external solver dependencies**.

Part of the TPT Solutions `tpt-*` ecosystem. Dual-licensed MIT / Apache-2.0.

## Why

Every numerical algorithm in this workspace is implemented from scratch:
LU, Cholesky, QR, real Schur (Francis double-shift QR), Padé scaling-and-
squaring matrix exponential, Bartels–Stewart Lyapunov, structured-doubling
DARE, Newton–Kleinman CARE, an active-set QP solver for MPC, and KF/EKF/UKF.
No LAPACK, no OSQP, no Apache-2.0-only crates (enforced by `cargo-deny`).

Design principles (see `spec.txt`):

* **Dimensional safety** — `Mat<T, R, C>` carries matrix shapes in const
  generics, so multiplying a 3×1 state by a 2×2 gain is a *compile error*.
* **`no_std + alloc` core** — `tpt-ctrl-core` builds for `thumbv6m-none-eabi`.
* **Mathematically rigorous** — property-based tests (proptest) for control
  invariants, Kani bounded model checking for panic-freedom in solver loops,
  Lyapunov contracts as checked numerical code.
* **AI-native** — cost functions and state transitions expose
  finite-difference Jacobian/gradient hooks (`tpt_ctrl_core::fd`) for
  gradient-based policy optimization.

## Layout

| Crate               | Contents                                                       |
|---------------------|----------------------------------------------------------------|
| `tpt-ctrl-core`     | `StateSpace<N, M, P>`, dimensioned matrices, LU/Cholesky/QR/Schur/expm, Lyapunov, exact ZOH + Tustin, RK4 |
| `tpt-ctrl-classic`  | transfer functions, PID (anti-windup), lead/lag/notch, Bode/Nyquist, gain & phase margins |
| `tpt-ctrl-optimal`  | CARE/DARE from scratch, LQR/DLQR, Kalman gain, LQG/dLQG        |
| `tpt-ctrl-estimation` | discrete KF (Joseph form), EKF (user Jacobians), UKF          |
| `tpt-ctrl-mpc`      | horizon condensing, active-set QP, constrained MPC, explicit-MPC tables |
| `tpt-ctrl-verify`   | proptest strategies, Lyapunov contract checks, Kani harnesses  |
| `tpt-control`       | umbrella crate (feature-gated re-exports)                      |

## Quick start

```
use tpt_control::prelude::*;

let a = DMat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]);
let b = DMat::from_rows(&[&[0.0], &[1.0]]);
let sol = tpt_control::optimal::lqr(&a, &b, &DMat::identity(2), &DMat::from_rows(&[&[1.0]]))
    .unwrap();
// sol.k = [1, sqrt(3)] for Q = I, R = 1 — the classic textbook result.
```

Run the worked examples:

```sh
cargo run -p tpt-ctrl-optimal --example inverted_pendulum
cargo run -p tpt-ctrl-estimation --example kf_sensor_fusion
cargo run -p tpt-ctrl-mpc --example constrained_mpc
```

## Development

```sh
just ci      # fmt + build + test + clippy + deny + doc
just no-std  # build the core for thumbv6m-none-eabi
just kani    # bounded model checking (Linux only)
```

Toolchain pinned in `rust-toolchain.toml`. Benchmarks (`criterion`) cover
the solver hot paths: `cargo bench -p tpt-ctrl-core -p tpt-ctrl-optimal -p tpt-ctrl-mpc`.

## Status

Phases 0–4 of `todo.md` are implemented; Phase 5 (release chores) is in
progress. The publish target (crates.io vs internal path dependencies) is
a pending decision — see `todo.md`.
