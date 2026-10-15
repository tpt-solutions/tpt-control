# tpt-ctrl-core

Foundation of the [tpt-control](https://github.com/tpt-solutions/tpt-control) stack: state-space models with compile-time dimensions, from-scratch dense linear algebra, and exact continuous-to-discrete conversion. No LAPACK, no external solvers.

## Features

- `StateSpace<N, M, P>` (compile-time dimensions) and `DynSystem` (runtime dimensions) for the solver crates
- Dense kernels: LU with partial pivoting, Cholesky, Householder QR, Hessenberg reduction, real Schur (Francis double-shift)
- Pade-13 matrix exponential with scaling and squaring
- Continuous and discrete Lyapunov solvers
- Exact zero-order-hold and Tustin discretization
- RK4 integration and discrete propagation
- Poles, spectral radius, stability, controllability and observability tests
- Finite-difference Jacobian and gradient hooks

## Install

```sh
cargo add tpt-ctrl-core
```

## Example

Discretize a double integrator with the exact zero-order hold:

```rust
use tpt_ctrl_core::smat::Mat;
use tpt_ctrl_core::state_space::{Continuous, StateSpace};

// x'' = u
let sys = Continuous::new(StateSpace::new(
    Mat::from_rows_arr(&[[0.0, 1.0], [0.0, 0.0]]),
    Mat::from_rows_arr(&[[0.0], [1.0]]),
    Mat::from_rows_arr(&[[1.0, 0.0]]),
    Mat::from_rows_arr(&[[0.0]]),
));
let disc = sys.discretize_zoh(0.05).unwrap();
assert!((disc.ss.a[(0, 1)] - 0.05).abs() < 1e-12); // exact, not approximated
```

Runnable walkthrough: [discretize_and_simulate](https://github.com/tpt-solutions/tpt-control/blob/master/crates/tpt-ctrl-core/examples/discretize_and_simulate.rs) compares ZOH and Tustin on a damped mass-spring system and simulates a step response.

```sh
cargo run -p tpt-ctrl-core --example discretize_and_simulate
```

## no_std

The default `std` feature enables the heap-backed matrices. For embedded targets use `alloc`, which routes the floating-point kernels through `libm`:

```toml
[dependencies]
tpt-ctrl-core = { version = "0.1", default-features = false, features = ["alloc"] }
```

## Status

Pre-1.0. The API may change between minor versions. See [CHANGELOG.md](CHANGELOG.md).

Dual-licensed under MIT OR Apache-2.0.
