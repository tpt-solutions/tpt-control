# tpt-ctrl-classic

Classical single-loop control for the [tpt-control](https://github.com/tpt-solutions/tpt-control) stack: transfer-function algebra, PID with anti-windup, lead, lag and notch compensators, Bode and Nyquist data, and gain and phase margins.

## Features

- `TransferFunction`: series, parallel and closed-loop algebra; poles and zeros via companion-matrix eigenvalues
- `Pid` with back-calculation anti-windup (`kaw`), derivative-on-measurement and derivative filtering
- `leadlag::{lead, lag, notch}` constructors with unity DC gain
- `freq`: frequency response, Bode and Nyquist data, and gain and phase margins with crossover interpolation

## Install

```sh
cargo add tpt-ctrl-classic
```

## Example

Margins of `G(s) = 1 / (s (s + 1))`, which has a textbook phase margin of about 51.8 degrees:

```rust
use tpt_ctrl_classic::freq::margins;
use tpt_ctrl_classic::tf::TransferFunction;

let g = TransferFunction::new(&[1.0], &[1.0, 1.0, 0.0]).unwrap();
let m = margins(&g, 1e-3, 1e3, 4000).unwrap();
assert!((m.phase_margin_deg - 51.8).abs() < 0.5);
assert!(m.gain_margin.is_infinite());
```

Runnable examples:

- [pid_step_response](https://github.com/tpt-solutions/tpt-control/blob/master/crates/tpt-ctrl-classic/examples/pid_step_response.rs): PI and PID tracking, and anti-windup under actuator saturation
- [bode_margins](https://github.com/tpt-solutions/tpt-control/blob/master/crates/tpt-ctrl-classic/examples/bode_margins.rs): margins of a plant before and after a lead compensator

```sh
cargo run -p tpt-ctrl-classic --example pid_step_response
cargo run -p tpt-ctrl-classic --example bode_margins
```

## Status

Pre-1.0. The API may change between minor versions. See [CHANGELOG.md](CHANGELOG.md).

Dual-licensed under MIT OR Apache-2.0.
