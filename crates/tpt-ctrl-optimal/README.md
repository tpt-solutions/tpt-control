# tpt-ctrl-optimal

Infinite-horizon optimal control for the [tpt-control](https://github.com/tpt-solutions/tpt-control) stack: Riccati solvers, LQR and DLQR gains, steady-state Kalman gains, and LQG.

## Features

- `dare`: discrete algebraic Riccati equation via the structure-preserving doubling algorithm
- `care`: continuous algebraic Riccati equation via Newton-Kleinman iteration, initialized from a stabilizing gain
- `lqr` and `dlqr`: state-feedback gains with the Riccati solution
- `kalman_gain` (discrete) and `kalman_gain_continuous`: steady-state observer gains from the dual Riccati equation
- `lqg` and `dlqg`: LQR plus Kalman observer, combined by the separation principle

## Install

```sh
cargo add tpt-ctrl-optimal tpt-ctrl-core
```

## Example

LQR for a double integrator, `x'' = u`, minimizing the integral of `x^2 + x'^2 + u^2`. The closed form gives `K = [1, sqrt(3)]`:

```rust
use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_optimal::lqr;

let a = DMat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]);
let b = DMat::from_rows(&[&[0.0], &[1.0]]);
let q = DMat::identity(2);
let r = DMat::from_rows(&[&[1.0]]);
let sol = lqr(&a, &b, &q, &r).unwrap();
assert!((sol.k[(0, 1)] - 3.0f64.sqrt()).abs() < 1e-6);
```

Runnable examples:

- [inverted_pendulum](https://github.com/tpt-solutions/tpt-control/blob/master/crates/tpt-ctrl-optimal/examples/inverted_pendulum.rs): LQR stabilization of a linearized cart-pole, simulated with RK4
- [lqg_regulator](https://github.com/tpt-solutions/tpt-control/blob/master/crates/tpt-ctrl-optimal/examples/lqg_regulator.rs): LQG with a noisy position sensor, checking the regulator and observer error dynamics

```sh
cargo run -p tpt-ctrl-optimal --example inverted_pendulum
cargo run -p tpt-ctrl-optimal --example lqg_regulator
```

## Status

Pre-1.0. The API may change between minor versions. See [CHANGELOG.md](CHANGELOG.md).

Dual-licensed under MIT OR Apache-2.0.
