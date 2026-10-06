//! LQR gain computation for continuous and discrete plants.

use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_core::error::Result;

use crate::care::care;
use crate::dare::dare;

/// LQR solution: Riccati matrix `X` and gain `K` (`u = -K x`).
#[derive(Clone, Debug, PartialEq)]
pub struct LqrSolution {
    /// Optimal cost-to-go matrix (`V(x) = x^T X x`).
    pub x: DMat<f64>,
    /// Optimal gain.
    pub k: DMat<f64>,
}

/// Continuous-time LQR: minimizes `int_0^inf x^T Q x + u^T R u dt`
/// for `x' = A x + B u` (via the CARE).
pub fn lqr(a: &DMat<f64>, b: &DMat<f64>, q: &DMat<f64>, r: &DMat<f64>) -> Result<LqrSolution> {
    let x = care(a, b, q, r)?;
    let r_inv = r.inverse()?;
    let k = &r_inv * &(b.transpose() * &x);
    Ok(LqrSolution { x, k })
}

/// Discrete-time LQR: minimizes `sum x^T Q x + u^T R u`
/// for `x+ = A x + B u` (via the DARE).
pub fn dlqr(a: &DMat<f64>, b: &DMat<f64>, q: &DMat<f64>, r: &DMat<f64>) -> Result<LqrSolution> {
    let sol = dare(a, b, q, r)?;
    Ok(LqrSolution { x: sol.x, k: sol.k })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ctrl_verify::testing::assert_close;

    #[test]
    fn dlqr_gain_satisfies_closed_form_check() {
        let a = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
        let b = DMat::from_rows(&[&[0.005], &[0.1]]);
        let q = DMat::identity(2);
        let r = DMat::from_rows(&[&[1.0]]);
        let sol = dlqr(&a, &b, &q, &r).unwrap();
        // Optimality: closed loop must be stable and the one-step
        // improvement identity holds:
        // K = (R + B^T X B)^-1 B^T X A.
        let btxb = &(b.transpose() * &sol.x) * &b;
        let k_direct = (&btxb + r)
            .solve(&(&(b.transpose() * &sol.x) * &a))
            .unwrap();
        assert!((&k_direct - &sol.k).norm_fro() < 1e-10);
        let cl = &a - &(&b * &sol.k);
        assert!(tpt_ctrl_core::analysis::is_stable_discrete(&cl, 1e-9).unwrap());
    }

    #[test]
    fn continuous_lqr_double_integrator_gain() {
        let a = DMat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]);
        let b = DMat::from_rows(&[&[0.0], &[1.0]]);
        let q = DMat::identity(2);
        let r = DMat::from_rows(&[&[1.0]]);
        let sol = lqr(&a, &b, &q, &r).unwrap();
        assert_close(sol.k[(0, 0)], 1.0, 1e-6);
        assert_close(sol.k[(0, 1)], 3.0f64.sqrt(), 1e-6);
    }

    #[test]
    fn dlqr_cost_decreases_with_q_scaling() {
        // Heavier state weight must give a larger Riccati value at x0.
        let a = DMat::from_rows(&[&[0.95, 0.2], &[-0.1, 0.85]]);
        let b = DMat::from_rows(&[&[0.05], &[0.3]]);
        let q1 = DMat::identity(2);
        let q2 = &DMat::identity(2) * 10.0;
        let r = DMat::from_rows(&[&[1.0]]);
        let x0 = DMat::from_rows(&[&[1.0], &[-0.5]]);
        let s1 = dlqr(&a, &b, &q1, &r).unwrap();
        let s2 = dlqr(&a, &b, &q2, &r).unwrap();
        let v1 = &(&x0.transpose() * &s1.x) * &x0;
        let v2 = &(&x0.transpose() * &s2.x) * &x0;
        assert!(v2[(0, 0)] > v1[(0, 0)]);
    }
}
