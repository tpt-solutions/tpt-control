//! Discrete algebraic Riccati equation via the structure-preserving
//! doubling algorithm (SDA).
//!
//! Solves `X = A^T X A - A^T X B (R + B^T X B)^{-1} B^T X A + Q` for the
//! stabilizing solution `X >= 0`. The doubling iteration uses only matrix
//! products and solves with `(I + H G)` type matrices, and converges
//! quadratically for the stabilizable/detectable problems arising in
//! control (no unit-circle eigenvalue pairs of the symplectic pencil).

use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_core::error::{LinalgError, Result};

/// Result of the DARE solve: covariance `X` and the LQR gain
/// `K = (R + B^T X B)^{-1} B^T X A`.
#[derive(Clone, Debug, PartialEq)]
pub struct DareSolution {
    /// Stabilizing Riccati solution.
    pub x: DMat<f64>,
    /// Optimal gain `K` such that `u = -K x` is optimal.
    pub k: DMat<f64>,
}

/// Solves the DARE for `(A, B, Q, R)` and returns `(X, K)`.
///
/// Requires `R` symmetric positive definite, `Q` symmetric positive
/// semidefinite, and `(A, B)` stabilizable with `(A, Q^{1/2})` detectable.
pub fn dare(a: &DMat<f64>, b: &DMat<f64>, q: &DMat<f64>, r: &DMat<f64>) -> Result<DareSolution> {
    let n = a.nrows();
    let m = b.ncols();
    if !a.is_square() || b.nrows() != n || q.nrows() != n || r.nrows() != m {
        return Err(LinalgError::DimensionMismatch {
            expected: (n, n),
            found: (b.nrows(), q.nrows()),
        });
    }
    if n == 0 {
        return Ok(DareSolution {
            x: a.clone(),
            k: DMat::zeros(0, m),
        });
    }
    let r_inv = r.inverse()?;

    // Doubling state: (A_k, G_k, H_k) with H_k -> X, A_k -> 0.
    let mut ak = a.clone();
    let mut gk = &(b * &r_inv) * b.transpose();
    let mut hk = q.clone();

    let scale = a.norm_1().max(1.0);
    let tol = 1e-14 * scale;
    let max_iter = 60;
    let eye_n = DMat::<f64>::identity(n);
    for _ in 0..max_iter {
        if ak.norm_1() <= tol {
            break;
        }
        // z = (I + G H)^{-1}, w = (I + H G)^{-1} (both n x n: G, H are n x n)
        let z = (&eye_n + &(&gk * &hk)).inverse()?;
        let w = (&eye_n + &(&hk * &gk)).inverse()?;

        // A+ = A (I + G H)^-1 A
        // G+ = G + A G (I + H G)^-1 A^T
        // H+ = H + A^T (I + H G)^-1 H A
        let ak_next = &(&ak * &z) * &ak;
        let gk_next = &gk + &(&(&(&ak * &gk) * &w) * &ak.transpose());
        let hk_next = &hk + &(&ak.transpose() * &(&(&w * &hk) * &ak));

        ak = ak_next;
        gk = gk_next;
        hk = hk_next;
    }
    if ak.norm_1() > 1e-6 * scale {
        return Err(LinalgError::NoConvergence);
    }

    // Gain: K = (R + B^T X B)^{-1} B^T X A.
    let x = hk;
    let btxb = &(b.transpose() * &x) * b;
    let k = (&btxb + r).solve(&(&(b.transpose() * &x) * a))?;
    Ok(DareSolution { x, k })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ctrl_verify::testing::{assert_close, assert_mat_close};

    #[test]
    fn dare_double_integrator_known_solution() {
        // Ad = [[1, 1], [0, 1]], Bd = [1/2; 1], Q = I, R = 1 (dt = 1).
        let a = DMat::from_rows(&[&[1.0, 1.0], &[0.0, 1.0]]);
        let b = DMat::from_rows(&[&[0.5], &[1.0]]);
        let q = DMat::identity(2);
        let r = DMat::from_rows(&[&[1.0]]);
        let sol = dare(&a, &b, &q, &r).unwrap();
        // Residual: X - A^T X A + A^T X B (R + B^T X B)^-1 B^T X A - Q = 0.
        let btxb = &(b.transpose() * &sol.x) * &b;
        let s = &(&btxb + &r).inverse().unwrap();
        let residual = &(&sol.x - &(&a.transpose() * &sol.x) * &a)
            + &(&(&(&a.transpose() * &sol.x) * &b) * s) * &(&(b.transpose() * &sol.x) * &a)
            - &q;
        assert!(
            residual.norm_fro() < 1e-9,
            "residual {}",
            residual.norm_fro()
        );
        // Closed loop A - B K Schur.
        let cl = &a - &(&b * &sol.k);
        assert!(tpt_ctrl_core::analysis::is_stable_discrete(&cl, 1e-9).unwrap());
    }

    #[test]
    fn dare_matches_fixed_point_iteration() {
        // Cross-check SDA against the slow Riccati recursion
        // X_{j+1} = A^T X_j (I + G X_j)^-1 A + Q from X_0 = Q.
        let a = DMat::from_rows(&[&[0.8, 0.2], &[-0.1, 0.7]]);
        let b = DMat::from_rows(&[&[0.3], &[0.9]]);
        let q = DMat::from_rows(&[&[2.0, 0.3], &[0.3, 1.5]]);
        let r = DMat::from_rows(&[&[1.2]]);
        let sol = dare(&a, &b, &q, &r).unwrap();

        let r_inv = r.inverse().unwrap();
        let g = &(&b * &r_inv) * &b.transpose();
        let mut x = q.clone();
        for _ in 0..5000 {
            let inner = &DMat::identity(2) + &(&g * &x);
            let x_next = &(&(&a.transpose() * &x) * &inner.inverse().unwrap()) * &a + &q;
            let change = (&x_next - &x).norm_fro() / x_next.norm_fro();
            x = x_next;
            if change < 1e-14 {
                break;
            }
        }
        assert_mat_close(&sol.x, &x, 1e-8);
    }

    #[test]
    fn dare_scalar_closed_form() {
        // Scalar: x = a^2 x - a^2 b^2 x^2 / (r + b^2 x) + q.
        let a = DMat::from_rows(&[&[1.1]]);
        let b = DMat::from_rows(&[&[0.7]]);
        let q = DMat::from_rows(&[&[2.0]]);
        let r = DMat::from_rows(&[&[1.5]]);
        let sol = dare(&a, &b, &q, &r).unwrap();
        // Closed form (stabilizing root): x = (r(1 - a^2) + q b^2 + sqrt(D)) / (2 b^2)? ...
        // verify via direct substitution instead:
        let x = sol.x[(0, 0)];
        let a2 = 1.21;
        let b2 = 0.49;
        let lhs = x;
        let rhs = a2 * x - a2 * b2 * x * x / (1.5 + b2 * x) + 2.0;
        assert_close(lhs, rhs, 1e-9);
        assert!(x > 0.0);
    }

    #[test]
    fn kalman_style_dual_dare_is_consistent() {
        // The dual DARE on (A^T, C^T) must also converge and give a stable
        // estimation error dynamics (I - L C) A.
        let a = DMat::from_rows(&[&[0.9, 0.5], &[-0.2, 0.8]]);
        let c = DMat::from_rows(&[&[1.0, 0.0]]);
        let qn = DMat::identity(2);
        let rn = DMat::from_rows(&[&[0.5]]);
        let sol = dare(&a.transpose(), &c.transpose(), &qn, &rn).unwrap();
        // Innovation gain L = P C^T (C P C^T + R)^-1.
        let s_cov = &c * &sol.x * &c.transpose() + &rn;
        let l = &(&sol.x * &c.transpose()) * &s_cov.inverse().unwrap();
        let err_dyn = &(&DMat::identity(2) - &(&l * c)) * &a;
        assert!(tpt_ctrl_core::analysis::is_stable_discrete(&err_dyn, 1e-9).unwrap());
    }
}
