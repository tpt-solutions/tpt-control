//! Extended Kalman filter with user-provided Jacobian closures.
//!
//! The nonlinear model is supplied per step as closures:
//! `f(x)` propagates the state, `jac_f(x)` returns `∂f/∂x`;
//! `h(x)` predicts the measurement, `jac_h(x)` returns `∂h/∂x`.
//!
//! The first-order EKF linearizes about the current estimate — no
//! external filtering or autodiff libraries are involved.

use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_core::error::Result;

use crate::kf;

/// A nonlinear plant + measurement model for the EKF.
pub trait EkfModel {
    /// Propagates the state: `x+ = f(x)`.
    fn propagate(&self, x: &DMat<f64>) -> DMat<f64>;
    /// State Jacobian at `x`: `F = ∂f/∂x`.
    fn state_jacobian(&self, x: &DMat<f64>) -> DMat<f64>;
    /// Predicts the measurement: `z = h(x)`.
    fn measure(&self, x: &DMat<f64>) -> DMat<f64>;
    /// Measurement Jacobian at `x`: `H = ∂h/∂x`.
    fn measurement_jacobian(&self, x: &DMat<f64>) -> DMat<f64>;
}

/// Stateful EKF estimate `(x, P)`.
#[derive(Clone, Debug)]
pub struct Ekf {
    /// State estimate.
    pub x: DMat<f64>,
    /// Estimate covariance.
    pub p: DMat<f64>,
}

impl Ekf {
    /// Creates a filter from an initial estimate and covariance.
    pub fn new(x0: DMat<f64>, p0: DMat<f64>) -> Ekf {
        Ekf { x: x0, p: p0 }
    }

    /// EKF time update with process covariance `q`.
    pub fn predict<M: EkfModel>(&mut self, model: &M, q: &DMat<f64>) {
        let f = model.state_jacobian(&self.x);
        let x_new = model.propagate(&self.x);
        self.x = x_new;
        let fp = &f * &self.p;
        self.p = &(fp * &f.transpose()) + q;
    }

    /// EKF measurement update; returns the innovation `z - h(x⁻)`.
    pub fn update<M: EkfModel>(
        &mut self,
        model: &M,
        z: &DMat<f64>,
        r: &DMat<f64>,
    ) -> Result<DMat<f64>> {
        let h = model.measurement_jacobian(&self.x);
        let (x_new, p_new, innovation, _) = kf::update(&self.x, &self.p, z, &h, r)?;
        self.x = x_new;
        self.p = p_new;
        Ok(innovation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ctrl_verify::testing::assert_close;

    /// Constant-velocity plant with a range-only measurement
    /// `z = sqrt(px^2 + py^2)` — the synthetic nonlinear validation plant
    /// (see crate docs).
    struct RangeOnly {
        dt: f64,
    }
    impl EkfModel for RangeOnly {
        fn propagate(&self, x: &DMat<f64>) -> DMat<f64> {
            DMat::from_rows(&[
                &[x[(0, 0)] + self.dt * x[(2, 0)]],
                &[x[(1, 0)] + self.dt * x[(3, 0)]],
                &[x[(2, 0)]],
                &[x[(3, 0)]],
            ])
        }
        fn state_jacobian(&self, _x: &DMat<f64>) -> DMat<f64> {
            let mut f = DMat::identity(4);
            f[(0, 2)] = self.dt;
            f[(1, 3)] = self.dt;
            f
        }
        fn measure(&self, x: &DMat<f64>) -> DMat<f64> {
            DMat::from_rows(&[&[(x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)]).sqrt()]])
        }
        fn measurement_jacobian(&self, x: &DMat<f64>) -> DMat<f64> {
            let range = (x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)])
                .sqrt()
                .max(1e-9);
            DMat::from_rows(&[&[x[(0, 0)] / range, x[(1, 0)] / range, 0.0, 0.0]])
        }
    }

    #[test]
    fn ekf_equals_kf_on_linear_model() {
        // With linear f and h, the EKF must reproduce the KF exactly.
        let a = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
        let h = DMat::from_rows(&[&[1.0, 0.0]]);
        let q = DMat::from_rows(&[&[1e-6, 0.0], &[0.0, 1e-4]]);
        let r = DMat::from_rows(&[&[0.05]]);
        let linear = LinearAdapter {
            a: a.clone(),
            h: h.clone(),
        };
        let mut ekf = Ekf::new(DMat::from_rows(&[&[0.5], &[-0.2]]), DMat::identity(2));
        ekf.predict(&linear, &q);
        let (kf_xp, kf_pp) = kf::predict(
            &DMat::from_rows(&[&[0.5], &[-0.2]]),
            &DMat::identity(2),
            &a,
            &q,
        );
        assert!((&ekf.x - &kf_xp).norm_fro() < 1e-12);
        assert!((&ekf.p - &kf_pp).norm_fro() < 1e-12);
        ekf.update(&linear, &DMat::from_rows(&[&[0.7]]), &r)
            .unwrap();
        let (kf_x, kf_p, _, _) =
            kf::update(&kf_xp, &kf_pp, &DMat::from_rows(&[&[0.7]]), &h, &r).unwrap();
        assert!((&ekf.x - &kf_x).norm_fro() < 1e-12);
        assert!((&ekf.p - &kf_p).norm_fro() < 1e-12);
    }

    #[test]
    fn ekf_tracks_range_only_target() {
        // Target moving in +x at unit speed, starting near (2, 2); range
        // measurements with moderate noise. Position RMSE must converge.
        let model = RangeOnly { dt: 0.2 };
        let q = DMat::identity(4) * 1e-6;
        let r = DMat::from_rows(&[&[0.04]]);
        let mut ekf = Ekf::new(
            DMat::from_rows(&[&[3.3], &[4.3], &[0.85], &[0.15]]),
            DMat::identity(4) * 0.25,
        );
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut noise = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let steps = 60;
        let mut err_sq_tail = 0.0;
        for k in 1..=steps {
            let t = k as f64 * model.dt;
            let truth = [3.0 + t, 4.0];
            let range = (truth[0] * truth[0] + truth[1] * truth[1]).sqrt();
            ekf.predict(&model, &q);
            let z = range + 0.05 * noise();
            ekf.update(&model, &DMat::from_rows(&[&[z]]), &r).unwrap();
            if k > steps / 2 {
                let ex = ekf.x[(0, 0)] - truth[0];
                let ey = ekf.x[(1, 0)] - truth[1];
                err_sq_tail += ex * ex + ey * ey;
            }
        }
        let rmse = (err_sq_tail / (steps as f64 / 2.0)).sqrt();
        assert!(rmse < 0.3, "EKF position RMSE {rmse}");
        assert_close(ekf.x[(2, 0)], 1.0, 0.3);
    }

    /// Adapter running a linear model through the EKF interface.
    struct LinearAdapter {
        a: DMat<f64>,
        h: DMat<f64>,
    }
    impl EkfModel for LinearAdapter {
        fn propagate(&self, x: &DMat<f64>) -> DMat<f64> {
            &self.a * x
        }
        fn state_jacobian(&self, _x: &DMat<f64>) -> DMat<f64> {
            self.a.clone()
        }
        fn measure(&self, x: &DMat<f64>) -> DMat<f64> {
            &self.h * x
        }
        fn measurement_jacobian(&self, _x: &DMat<f64>) -> DMat<f64> {
            self.h.clone()
        }
    }
}
