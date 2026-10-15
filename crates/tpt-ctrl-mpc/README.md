# tpt-ctrl-mpc

Model predictive control for the [tpt-control](https://github.com/tpt-solutions/tpt-control) stack: horizon condensing, input bounds and state constraints, an active-set QP solver written from scratch, and explicit-MPC lookup tables.

## Features

- `unroll::unroll`: horizon condensing into an affine map from the input sequence
- `MpcBuilder` / `Mpc`: condensed linear MPC with input bounds and linear state constraints, solved every step (receding horizon)
- `qp`: strictly convex active-set QP solver with exact KKT sub-solves and a parallel-constraint fallback
- `ExplicitMpc`: nearest-neighbour lookup tables for low-latency deployment

## Install

```sh
cargo add tpt-ctrl-mpc tpt-ctrl-core
```

## Example

A double integrator with the input bounded to `|u| <= 0.5`:

```rust
use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_mpc::MpcBuilder;

let a = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
let b = DMat::from_rows(&[&[0.005], &[0.1]]);
let mpc = MpcBuilder::new(a, b, 10)
    .stage_costs(&DMat::identity(2), &DMat::from_rows(&[&[1.0]]))
    .terminal_cost(&(&DMat::identity(2) * 5.0))
    .input_bounds(vec![-0.5], vec![0.5])
    .build();
let sol = mpc.solve(&DMat::from_rows(&[&[0.5], &[0.5]])).unwrap();
assert!(sol.u0[0].abs() <= 0.5 + 1e-9);
```

Runnable example: [constrained_mpc](https://github.com/tpt-solutions/tpt-control/blob/master/crates/tpt-ctrl-mpc/examples/constrained_mpc.rs) drives a large initial state to the origin under an actuator bound, and compares the closed loop against an unconstrained DLQR that ignores the bound.

```sh
cargo run -p tpt-ctrl-mpc --example constrained_mpc
```

## Status

Pre-1.0. The API may change between minor versions. See [CHANGELOG.md](CHANGELOG.md).

Dual-licensed under MIT OR Apache-2.0.
