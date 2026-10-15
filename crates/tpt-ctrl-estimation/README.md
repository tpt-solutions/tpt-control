# tpt-ctrl-estimation

State estimation for the [tpt-control](https://github.com/tpt-solutions/tpt-control) stack: a discrete Kalman filter, an extended Kalman filter with user-supplied Jacobians, and an unscented Kalman filter. Written from scratch, with no external filtering libraries.

## Features

- `kf`: discrete Kalman filter with the Joseph-form covariance update
- `ekf`: extended Kalman filter driven by model and Jacobian closures
- `ukf`: unscented Kalman filter with Wan-van der Merwe weights and Cholesky sigma points

## Install

```sh
cargo add tpt-ctrl-estimation tpt-ctrl-core
```

## Example

Fuse a noisy position sensor with a constant-velocity model:

```rust
use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_estimation::kf::KalmanFilter;

let dt = 0.2;
let a = DMat::from_rows(&[&[1.0, dt], &[0.0, 1.0]]);
let h = DMat::from_rows(&[&[1.0, 0.0]]);
let q = DMat::from_rows(&[&[1e-6, 0.0], &[0.0, 1e-4]]);
let r = DMat::from_rows(&[&[0.04]]);
let mut kf = KalmanFilter::new(
    DMat::from_rows(&[&[0.0], &[0.0]]),
    DMat::identity(2) * 5.0,
    a, q, h, r,
);

kf.predict();
kf.update(&DMat::from_rows(&[&[2.3]])).unwrap();
let position = kf.x[(0, 0)];
let velocity = kf.x[(1, 0)];
```

Runnable example: [kf_sensor_fusion](https://github.com/tpt-solutions/tpt-control/blob/master/crates/tpt-ctrl-estimation/examples/kf_sensor_fusion.rs) fuses noisy measurements of a target cruising at 1.5 m/s and compares the estimate's error against the raw sensor.

```sh
cargo run -p tpt-ctrl-estimation --example kf_sensor_fusion
```

## Status

Pre-1.0. The API may change between minor versions. See [CHANGELOG.md](CHANGELOG.md).

Dual-licensed under MIT OR Apache-2.0.
