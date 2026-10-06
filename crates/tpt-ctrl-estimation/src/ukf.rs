//! Unscented Kalman filter.
//!
//! The UKF propagates `2n + 1` sigma points through the full nonlinear
//! model and matches moments to second order — no Jacobians required.
//! Sigma points are generated via the Cholesky factor of the covariance.

use std::vec::Vec;

use tpt_ctrl_core::cholesky::cholesky;
use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_core::error::{LinalgError, Result};

/// UKF scaling parameters (standard formulation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UkfParams {
    /// Primary spread parameter, typically `1e-4..1`.
    pub alpha: f64,
    /// Secondary scaling, usually `0` or `3 - n`.
    pub kappa: f64,
    /// Distribution prior (2 is a good default for Gaussians).
    pub beta: f64,
}

impl Default for UkfParams {
    fn default() -> UkfParams {
        UkfParams {
            alpha: 1e-3,
            kappa: 0.0,
            beta: 2.0,
        }
    }
}

impl UkfParams {
    fn weights(&self, n: usize) -> (Vec<f64>, Vec<f64>) {
        let lambda = self.alpha * self.alpha * (n as f64 + self.kappa) - self.kappa;
        let wm0 = lambda / (n as f64 + lambda);
        let wc0 = wm0 + 1.0 - self.alpha * self.alpha + self.beta;
        let wi = 1.0 / (2.0 * (n as f64 + lambda));
        let mut wm = vec![wi; 2 * n + 1];
        let mut wc = vec![wi; 2 * n + 1];
        wm[0] = wm0;
        wc[0] = wc0;
        (wm, wc)
    }

    fn gamma(&self, n: usize) -> f64 {
        let lambda = self.alpha * self.alpha * (n as f64 + self.kappa) - self.kappa;
        (n as f64 + lambda).sqrt()
    }
}

/// Generates `2n + 1` sigma points as columns of an `n x (2n+1)` matrix.
pub fn sigma_points(x: &DMat<f64>, p: &DMat<f64>, gamma: f64) -> Result<DMat<f64>> {
    let n = x.nrows();
    let l = cholesky(p)?;
    let mut pts = DMat::zeros(n, 2 * n + 1);
    for i in 0..n {
        pts[(i, 0)] = x[(i, 0)];
    }
    for j in 0..n {
        for i in 0..n {
            pts[(i, j + 1)] = x[(i, 0)] + gamma * l[(i, j)];
            pts[(i, n + j + 1)] = x[(i, 0)] - gamma * l[(i, j)];
        }
    }
    Ok(pts)
}

/// Unscented transform: propagates sigma points through `f` and matches
/// mean and covariance. `f` maps column vectors to column vectors.
pub fn unscented_transform(
    pts: &DMat<f64>,
    wm: &[f64],
    wc: &[f64],
    f: impl Fn(&DMat<f64>) -> DMat<f64>,
    add_noise: Option<&DMat<f64>>,
) -> (DMat<f64>, DMat<f64>) {
    let n = pts.nrows();
    let m = pts.ncols();
    let mut propagated = DMat::zeros(n, m);
    for j in 0..m {
        let col = pts.submatrix(0, j, n, 1);
        let out = f(&col);
        for i in 0..n {
            propagated[(i, j)] = out[(i, 0)];
        }
    }
    let mut mean = DMat::zeros(n, 1);
    for (j, w) in wm.iter().enumerate() {
        for i in 0..n {
            mean[(i, 0)] += w * propagated[(i, j)];
        }
    }
    let mut cov = DMat::zeros(n, n);
    for (j, w) in wc.iter().enumerate() {
        let d = propagated.submatrix(0, j, n, 1) - &mean;
        cov = &cov + &(&d * &d.transpose()) * *w;
    }
    if let Some(q) = add_noise {
        cov = &cov + q;
    }
    (mean, cov)
}

/// Stateful UKF.
pub struct Ukf<F, H>
where
    F: Fn(&DMat<f64>) -> DMat<f64>,
    H: Fn(&DMat<f64>) -> DMat<f64>,
{
    /// State estimate.
    pub x: DMat<f64>,
    /// Estimate covariance.
    pub p: DMat<f64>,
    params: UkfParams,
    f: F,
    h: H,
}

impl<F, H> Ukf<F, H>
where
    F: Fn(&DMat<f64>) -> DMat<f64>,
    H: Fn(&DMat<f64>) -> DMat<f64>,
{
    /// Creates a UKF for the given propagate/measure closures.
    pub fn new(x0: DMat<f64>, p0: DMat<f64>, params: UkfParams, f: F, h: H) -> Ukf<F, H> {
        Ukf {
            x: x0,
            p: p0,
            params,
            f,
            h,
        }
    }

    /// UKF time update.
    pub fn predict(&mut self, q: &DMat<f64>) -> Result<()> {
        let n = self.x.nrows();
        let pts = sigma_points(&self.x, &self.p, self.params.gamma(n))?;
        let (wm, wc) = self.params.weights(n);
        let (x_pred, p_pred) = unscented_transform(&pts, &wm, &wc, |c| (self.f)(c), Some(q));
        self.x = x_pred;
        self.p = p_pred;
        Ok(())
    }

    /// UKF measurement update; returns the innovation `z - h(x⁻)`.
    pub fn update(&mut self, z: &DMat<f64>, r: &DMat<f64>) -> Result<DMat<f64>> {
        let n = self.x.nrows();
        let m = z.nrows();
        let pts = sigma_points(&self.x, &self.p, self.params.gamma(n))?;
        let (wm, wc) = self.params.weights(n);
        // Measurement sigma points.
        let mut zpts = DMat::zeros(m, pts.ncols());
        for j in 0..pts.ncols() {
            let col = pts.submatrix(0, j, n, 1);
            let out = (self.h)(&col);
            for i in 0..m {
                zpts[(i, j)] = out[(i, 0)];
            }
        }
        let mut z_mean = DMat::zeros(m, 1);
        for (j, w) in wm.iter().enumerate() {
            for i in 0..m {
                z_mean[(i, 0)] += w * zpts[(i, j)];
            }
        }
        let mut pzz = DMat::zeros(m, m);
        let mut pxz = DMat::zeros(n, m);
        for (j, w) in wc.iter().enumerate() {
            let dz = zpts.submatrix(0, j, m, 1) - &z_mean;
            let dx = pts.submatrix(0, j, n, 1) - &self.x;
            pzz = &pzz + &(&dz * &dz.transpose()) * *w;
            pxz = &pxz + &(&dx * &dz.transpose()) * *w;
        }
        pzz = &pzz + r;
        let k = &pxz * &pzz.inverse()?;
        let innovation = z - &z_mean;
        self.x = &self.x + &(&k * &innovation);
        self.p = &self.p - &(&(&k * &pzz) * &k.transpose());
        Ok(innovation)
    }
}

/// Ensures the covariance stays symmetric positive semidefinite after
/// updates (small numerical drift guard): symmetrizes `P`.
pub fn symmetrize(p: &mut DMat<f64>) {
    let pt = p.transpose();
    let avg = (p.clone() + pt) * 0.5;
    *p = avg;
}

/// Convenience wrapper re-exported for tests.
pub fn ensure_valid_covariance(p: &mut DMat<f64>) -> Result<()> {
    symmetrize(p);
    if !p.is_positive_definite() {
        return Err(LinalgError::NotPositiveDefinite);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ctrl_verify::testing::assert_close;

    #[test]
    fn sigma_points_match_mean_and_covariance() {
        let x = DMat::from_rows(&[&[1.0], &[-2.0]]);
        let p = DMat::from_rows(&[&[4.0, 0.5], &[0.5, 1.0]]);
        let params = UkfParams::default();
        let pts = sigma_points(&x, &p, params.gamma(2)).unwrap();
        let (wm, wc) = params.weights(2);
        // The sigma points must carry exactly the mean and covariance back.
        let (mean, cov) = unscented_transform(&pts, &wm, &wc, |c| c.clone(), None);
        assert_close(mean[(0, 0)], 1.0, 1e-12);
        assert_close(mean[(1, 0)], -2.0, 1e-12);
        for i in 0..2 {
            for j in 0..2 {
                assert_close(cov[(i, j)], p[(i, j)], 1e-10);
            }
        }
    }

    #[test]
    fn ukf_matches_kf_on_linear_model() {
        // Linear plant + linear measurement: UKF == KF to tolerance.
        let dt = 0.1;
        let a = DMat::from_rows(&[&[1.0, dt], &[0.0, 1.0]]);
        let h = DMat::from_rows(&[&[1.0, 0.0]]);
        let q = DMat::from_rows(&[&[1e-6, 0.0], &[0.0, 1e-4]]);
        let r = DMat::from_rows(&[&[0.05]]);

        let mut ukf = Ukf::new(
            DMat::from_rows(&[&[0.5], &[-0.2]]),
            DMat::identity(2),
            UkfParams::default(),
            |x| &a * x,
            |x| &h * x,
        );
        ukf.predict(&q).unwrap();
        ukf.update(&DMat::from_rows(&[&[0.7]]), &r).unwrap();

        let mut kf_x = DMat::from_rows(&[&[0.5], &[-0.2]]);
        let mut kf_p = DMat::identity(2);
        let (xp, pp) = crate::kf::predict(&kf_x, &kf_p, &a, &q);
        let (xn, pn, _, _) =
            crate::kf::update(&xp, &pp, &DMat::from_rows(&[&[0.7]]), &h, &r).unwrap();
        kf_x = xn;
        kf_p = pn;
        assert!(
            (&ukf.x - &kf_x).norm_fro() < 1e-8,
            "x {}",
            (&ukf.x - &kf_x).norm_fro()
        );
        assert!((&ukf.p - &kf_p).norm_fro() < 1e-6);
    }

    #[test]
    fn ukf_tracks_coordinated_turn() {
        // Nonlinear plant: constant turn-rate motion. Ground truth rotates
        // about the origin; UKF tracks position from polar-ish measurements
        // (range + angle with noise).
        let dt = 0.1_f64;
        let omega = 0.5_f64;
        let f = move |x: &DMat<f64>| {
            // exact rotation model
            DMat::from_rows(&[
                &[x[(0, 0)] * (omega * dt).cos() - x[(1, 0)] * (omega * dt).sin()],
                &[x[(0, 0)] * (omega * dt).sin() + x[(1, 0)] * (omega * dt).cos()],
            ])
        };
        let h = |x: &DMat<f64>| {
            let range = (x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)]).sqrt();
            let ang = x[(1, 0)].atan2(x[(0, 0)]);
            DMat::from_rows(&[&[range], &[ang]])
        };
        let q = DMat::identity(2) * 1e-8;
        let r = DMat::from_rows(&[&[0.01, 0.0], &[0.0, 1e-4]]);
        let mut ukf = Ukf::new(
            DMat::from_rows(&[&[3.0], &[0.2]]),
            DMat::identity(2) * 0.5,
            UkfParams::default(),
            f,
            h,
        );
        let mut seed = 0xDEAD_BEEF_u64;
        let mut noise = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let steps = 150;
        let mut err_sq = 0.0;
        for k in 1..=steps {
            let t = k as f64 * dt;
            let truth = [3.0 * (omega * t).cos(), 3.0 * (omega * t).sin()];
            ukf.predict(&q).unwrap();
            let range = (truth[0] * truth[0] + truth[1] * truth[1]).sqrt();
            let ang = truth[1].atan2(truth[0]);
            let z = DMat::from_rows(&[&[range + 0.1 * noise()], &[ang + 0.01 * noise()]]);
            ukf.update(&z, &r).unwrap();
            let dx = ukf.x[(0, 0)] - truth[0];
            let dy = ukf.x[(1, 0)] - truth[1];
            if k > steps / 2 {
                err_sq += dx * dx + dy * dy;
            }
        }
        let rmse = (err_sq / (steps as f64 / 2.0)).sqrt();
        assert!(rmse < 0.2, "UKF polar RMSE {rmse}");
    }
}
