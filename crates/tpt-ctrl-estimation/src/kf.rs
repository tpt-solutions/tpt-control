//! Discrete Kalman filter.
//!
//! `x⁻ = A x,  P⁻ = A P A^T + Q`
//! `K = P⁻ H^T (H P⁻ H^T + R)^{-1}`
//! `x = x⁻ + K (z - H x⁻)`, covariance in Joseph form for numerical
//! robustness: `P = (I - K H) P⁻ (I - K H)^T + K R K^T`.

use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_core::error::{LinalgError, Result};

/// One KF prediction step; returns the predicted `(x⁻, P⁻)`.
pub fn predict(
    x: &DMat<f64>,
    p: &DMat<f64>,
    a: &DMat<f64>,
    q: &DMat<f64>,
) -> (DMat<f64>, DMat<f64>) {
    let x_pred = a * x;
    let p_pred = &(&(a * p) * &a.transpose()) + q;
    (x_pred, p_pred)
}

/// One KF measurement update (Joseph form).
///
/// Returns `(x_new, p_new, innovation, innovation_cov)`.
#[allow(clippy::type_complexity)]
pub fn update(
    x_pred: &DMat<f64>,
    p_pred: &DMat<f64>,
    z: &DMat<f64>,
    h: &DMat<f64>,
    r: &DMat<f64>,
) -> Result<(DMat<f64>, DMat<f64>, DMat<f64>, DMat<f64>)> {
    let n = x_pred.nrows();
    let m = z.nrows();
    if h.ncols() != n || r.nrows() != m {
        return Err(LinalgError::DimensionMismatch {
            expected: (n, m),
            found: (h.ncols(), r.nrows()),
        });
    }
    let ph = p_pred * &h.transpose();
    let s = &(&(h * p_pred) * &h.transpose()) + r; // innovation covariance
    let k = ph * &s.inverse()?; // Kalman gain
    let innovation = z - &(h * x_pred);
    let x_new = x_pred + &(&k * &innovation);

    let eye = DMat::<f64>::identity(n);
    let ikh = &eye - &(&k * h);
    let p_new = &(&(&ikh * p_pred) * &ikh.transpose()) + &(&(&k * r) * &k.transpose());
    Ok((x_new, p_new, innovation, s))
}

/// Stateful Kalman filter carrying `(x, P)` with fixed model matrices.
#[derive(Clone, Debug)]
pub struct KalmanFilter {
    /// State estimate.
    pub x: DMat<f64>,
    /// Estimate covariance.
    pub p: DMat<f64>,
    a: DMat<f64>,
    q: DMat<f64>,
    h: DMat<f64>,
    r: DMat<f64>,
}

impl KalmanFilter {
    /// Builds a filter for the model `x+ = A x + w`, `z = H x + v`.
    pub fn new(
        x0: DMat<f64>,
        p0: DMat<f64>,
        a: DMat<f64>,
        q: DMat<f64>,
        h: DMat<f64>,
        r: DMat<f64>,
    ) -> KalmanFilter {
        KalmanFilter {
            x: x0,
            p: p0,
            a,
            q,
            h,
            r,
        }
    }

    /// Time update.
    pub fn predict(&mut self) {
        let (x, p) = predict(&self.x, &self.p, &self.a, &self.q);
        self.x = x;
        self.p = p;
    }

    /// Measurement update; returns the innovation `(z - H x⁻)`.
    #[allow(clippy::type_complexity)]
    pub fn update(&mut self, z: &DMat<f64>) -> Result<DMat<f64>> {
        let (x, p, innovation, _) = update(&self.x, &self.p, z, &self.h, &self.r)?;
        self.x = x;
        self.p = p;
        Ok(innovation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ctrl_verify::testing::{assert_close, assert_mat_close};

    #[test]
    fn kf_beats_raw_measurement_on_constant_state() {
        // Static state x = 5, noisy measurements var 4, prior var 100.
        // After one update the estimate variance must be < 4.
        let x0 = DMat::from_rows(&[&[0.0]]);
        let p0 = DMat::from_rows(&[&[100.0]]);
        let a = DMat::from_rows(&[&[1.0]]);
        let q = DMat::from_rows(&[&[0.0]]);
        let h = DMat::from_rows(&[&[1.0]]);
        let r = DMat::from_rows(&[&[4.0]]);
        let (xp, pp) = predict(&x0, &p0, &a, &q);
        assert_close(pp[(0, 0)], 100.0, 1e-12);
        let (x1, p1, _, s) = update(&xp, &pp, &DMat::from_rows(&[&[7.0]]), &h, &r).unwrap();
        // K = 100/104, x1 = 100/104 * 7
        assert_close(x1[(0, 0)], 100.0 / 104.0 * 7.0, 1e-12);
        assert_close(p1[(0, 0)], 100.0 * 4.0 / 104.0, 1e-12);
        assert_close(s[(0, 0)], 104.0, 1e-12);
        assert!(p1[(0, 0)] < 4.0);
    }

    #[test]
    fn kf_matches_optimal_steady_state_gain() {
        // The filter covariance must converge to the DARE steady state from
        // tpt-ctrl-optimal.
        let a = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
        let h = DMat::from_rows(&[&[1.0, 0.0]]);
        let q = DMat::from_rows(&[&[1e-4, 0.0], &[0.0, 1e-2]]);
        let r = DMat::from_rows(&[&[0.05]]);

        let mut kf = KalmanFilter::new(
            DMat::from_rows(&[&[0.0], &[0.0]]),
            DMat::identity(2),
            a.clone(),
            q.clone(),
            h.clone(),
            r.clone(),
        );
        for _ in 0..3000 {
            kf.predict();
            kf.update(&DMat::from_rows(&[&[0.0]])).unwrap();
        }
        // The DARE steady state is the *predicted* covariance: predict once
        // before reading off the gain.
        kf.predict();
        let (_, l) = tpt_ctrl_optimal::kalman_gain(&a, &h, &q, &r).unwrap();
        let kf_gain =
            &(&kf.p * &h.transpose()) * &(&(&h * &kf.p) * &h.transpose() + &r).inverse().unwrap();
        assert_mat_close(&kf_gain, &l, 1e-6);
    }

    #[test]
    fn kf_tracks_constant_velocity() {
        // Two-state CV model, position measurement; estimate error must stay
        // small against a seeded pseudo-random noise sequence.
        let dt = 0.1;
        let a = DMat::from_rows(&[&[1.0, dt], &[0.0, 1.0]]);
        let h = DMat::from_rows(&[&[1.0, 0.0]]);
        let q = DMat::from_rows(&[&[1e-6, 0.0], &[0.0, 1e-4]]);
        let r = DMat::from_rows(&[&[0.01]]);

        let mut kf = KalmanFilter::new(
            DMat::from_rows(&[&[0.0], &[0.0]]),
            DMat::identity(2) * 10.0,
            a.clone(),
            q.clone(),
            h.clone(),
            r.clone(),
        );
        // True trajectory: x = 0.5 * t^2 (accelerating at 1).
        let mut seed = 0x1234_5678_u64;
        let mut noise = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut err_sq = 0.0;
        let steps = 400;
        for k in 1..=steps {
            let t = k as f64 * dt;
            let true_state = [t, 1.0];
            kf.predict();
            let z = true_state[0] + 0.1 * noise();
            kf.update(&DMat::from_rows(&[&[z]])).unwrap();
            let err = kf.x[(0, 0)] - true_state[0];
            err_sq += err * err;
        }
        let rmse = (err_sq / steps as f64).sqrt();
        assert!(rmse < 0.05, "KF RMSE {rmse}");
    }
}
