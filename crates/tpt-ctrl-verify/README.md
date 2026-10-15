# tpt-ctrl-verify

Verification tools for the [tpt-control](https://github.com/tpt-solutions/tpt-control) stack. Use it in your own tests to generate well-posed control problems and to check Lyapunov conditions.

## Features

- `strategies`: proptest generators for SPD matrices, Hurwitz and Schur-stable matrices, LTI systems and bounded noise
- `lyapunov`: sampled checks of `V(x) > 0` and `V'(x) < 0` over a box, plus a positive-definiteness test for `P`
- `proofs`: Kani bounded-model-checking harnesses, compiled only under `cfg(kani)`
- `testing`: tolerance-based assertion helpers

## Install

Add it as a dev-dependency so the generators stay out of release builds:

```sh
cargo add --dev tpt-ctrl-verify
```

## Example

Check the Lyapunov contract of a closed loop over a sampled box:

```rust
use tpt_ctrl_verify::lyapunov::{check_lyapunov_contract, QuadraticWithField};

let lyap = QuadraticWithField::new(&p, field);
let report = check_lyapunov_contract(&lyap, &[(-2.0, 2.0), (-2.0, 2.0)], 9, 0.0, 0.0)?;
assert!(report.max_v_dot < 0.0);
```

Runnable example: [lyapunov_contract](https://github.com/tpt-solutions/tpt-control/blob/master/crates/tpt-ctrl-verify/examples/lyapunov_contract.rs) designs an LQR, takes its Riccati solution as the Lyapunov function, and checks the contract on a grid.

```sh
cargo run -p tpt-ctrl-verify --example lyapunov_contract
```

To run the Kani proofs on Linux, use `cargo kani -p tpt-ctrl-verify`. Ordinary `cargo test` does not compile them.

## Status

Pre-1.0. The API may change between minor versions. See [CHANGELOG.md](CHANGELOG.md).

Dual-licensed under MIT OR Apache-2.0.
