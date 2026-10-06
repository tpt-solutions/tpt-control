//! Worked example: sensor fusion for a constant-velocity target.
//!
//! A noisy position sensor is fused with a constant-velocity motion model
//! using the discrete Kalman filter (Joseph form); the fused estimate is
//! compared against the raw measurements.

use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_estimation::kf::KalmanFilter;

fn main() {
    let dt = 0.2_f64;
    let a = DMat::from_rows(&[&[1.0, dt], &[0.0, 1.0]]);
    let h = DMat::from_rows(&[&[1.0, 0.0]]);
    let q = DMat::from_rows(&[&[1e-6, 0.0], &[0.0, 1e-4]]);
    let r = DMat::from_rows(&[&[0.04]]);

    let mut kf = KalmanFilter::new(
        DMat::from_rows(&[&[0.0], &[0.0]]),
        DMat::identity(2) * 5.0,
        a.clone(),
        q.clone(),
        h.clone(),
        r.clone(),
    );

    // Truth: target cruising at 1.5 m/s from x = 2 m.
    let mut seed = 0x5EED_1D5A_u64;
    let mut noise = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };

    let mut raw_err = 0.0;
    let mut fused_err = 0.0;
    for k in 1..=200 {
        let t = k as f64 * dt;
        let truth = 2.0 + 1.5 * t;
        kf.predict();
        let z = truth + 0.3 * noise();
        kf.update(&DMat::from_rows(&[&[z]])).unwrap();
        raw_err += (z - truth) * (z - truth);
        fused_err += (kf.x[(0, 0)] - truth) * (kf.x[(0, 0)] - truth);
    }
    println!("raw measurement RMSE:   {:.4} m", (raw_err / 200.0).sqrt());
    println!(
        "Kalman estimate RMSE:   {:.4} m",
        (fused_err / 200.0).sqrt()
    );
    println!("fused velocity estimate: {:.4} m/s", kf.x[(1, 0)]);
}
