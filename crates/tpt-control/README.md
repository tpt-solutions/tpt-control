# tpt-control

Pure-Rust, AI-native control systems theory: state-space models, classical control, optimal control, estimation and MPC, with no LAPACK and no external solver dependencies. This umbrella crate re-exports the whole stack behind feature flags.

## Features

| Feature      | Re-export                   | Contents                                          |
|--------------|-----------------------------|---------------------------------------------------|
| (always)     | `tpt_ctrl_core`             | state-space models, dense linalg, discretization  |
| `classic`    | `tpt_control::classic`      | transfer functions, PID, lead-lag, Bode, margins  |
| `optimal`    | `tpt_control::optimal`      | CARE/DARE, LQR/DLQR, Kalman, LQG                  |
| `estimation` | `tpt_control::estimation`   | Kalman filter, EKF, UKF                           |
| `mpc`        | `tpt_control::mpc`          | condensed MPC, active-set QP, explicit tables     |
| `verify`     | `tpt_control::verify`       | proptest strategies, Lyapunov checks, Kani proofs |

All features are on by default. To pick only what you need:

```toml
[dependencies]
tpt-control = { version = "0.1", default-features = false, features = ["optimal"] }
```

## Install

```sh
cargo add tpt-control
```

## Example

Design an LQR for a double integrator and confirm the closed loop is stable:

```rust
use tpt_control::prelude::*;

let a = DMat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]);
let b = DMat::from_rows(&[&[0.0], &[1.0]]);
let sol = tpt_control::optimal::lqr(&a, &b, &DMat::identity(2), &DMat::from_rows(&[&[1.0]])).unwrap();
let acl = &a - &(&b * &sol.k);
assert!(tpt_control::tpt_ctrl_core::analysis::is_stable_continuous(&acl, 1e-9).unwrap());
```

Runnable tour: [quickstart](https://github.com/tpt-solutions/tpt-control/blob/master/crates/tpt-control/examples/quickstart.rs) covers LQR design, closed-loop verification and a Kalman filter on noisy measurements.

```sh
cargo run -p tpt-control --example quickstart
```

For per-crate walkthroughs, see the READMEs of the individual crates in the [workspace](https://github.com/tpt-solutions/tpt-control).

## Status

Pre-1.0. The API may change between minor versions. See [CHANGELOG.md](CHANGELOG.md).

Dual-licensed under MIT OR Apache-2.0.
