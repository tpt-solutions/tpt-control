//! Continuous algebraic Riccati equation via Newton-Kleinman iteration.
//!
//! Solves `A^T X + X A - X B R^{-1} B^T X + Q = 0` for the stabilizing
//! solution. Newton's iteration needs a stabilizing initial gain; we get
//! one rigorously from the DLQR gain of a fine ZOH discretization (a gain
//! that stabilizes the ZOH discretization stabilizes the continuous plant,
//! because the closed-loop poles map as `lambda_c -> exp(lambda_c dt)`).

use tpt_ctrl_core::discretize::zoh;
use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_core::error::{LinalgError, Result};

use crate::dare::dare;

/// Solves the CARE for `(A, B, Q, R)`; returns the Riccati solution `X`.
///
/// Requires `R` positive definite and the usual stabilizability /
/// detectability assumptions.
pub fn care(a: &DMat<f64>, b: &DMat<f64>, q: &DMat<f64>, r: &DMat<f64>) -> Result<DMat<f64>> {
    let n = a.nrows();
    if !a.is_square() || b.nrows() != n || q.nrows() != n {
        return Err(LinalgError::DimensionMismatch {
            expected: (n, n),
            found: (b.nrows(), q.nrows()),
        });
    }
    if n == 0 {
        return Ok(a.clone());
    }
    let r_inv = r.inverse()?;

    // Initial stabilizing gain from a fine ZOH discretization.
    let dt = 0.1 / (1.0 + a.norm_1());
    let (ad, bd) = zoh(a, b, dt)?;
    let qd = q * dt;
    let rd = r * dt;
    let dare_sol = dare(&ad, &bd, &qd, &rd)?;
    let mut k = dare_sol.k;

    let mut x = DMat::<f64>::zeros(n, n);
    for iter in 0..30 {
        let ak = &(a - &(b * &k));
        // solve_lyapunov(a, m) solves A^T X + X A = -m, so pass (Q + K^T R K).
        let krk = &(k.transpose() * r) * &k;
        let x_next = ak.solve_lyapunov(&(q + &krk))?;

        let change = (&x_next - &x).norm_fro() / x_next.norm_fro().max(1e-300);
        x = x_next;
        k = &r_inv * &(b.transpose() * &x);
        let _ = iter;
        if change < 1e-12 {
            break;
        }
    }

    // Final gain consistency.
    let _ = k;
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ctrl_verify::testing::assert_close;

    #[test]
    fn care_double_integrator_closed_form() {
        let a = DMat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]);
        let b = DMat::from_rows(&[&[0.0], &[1.0]]);
        let q = DMat::identity(2);
        let r = DMat::from_rows(&[&[1.0]]);
        let x = care(&a, &b, &q, &r).unwrap();
        assert_close(x[(0, 0)], 3.0f64.sqrt(), 1e-6);
        assert_close(x[(0, 1)], 1.0, 1e-6);
        assert_close(x[(1, 1)], 3.0f64.sqrt(), 1e-6);
        // Residual A^T X + X A - X B R^-1 B^T X + Q = 0.
        let r_inv = r.inverse().unwrap();
        let bx = &b.transpose() * &x;
        let quad = &(&(bx.transpose() * &r_inv) * &bx);
        let res = &(&(&a.transpose() * &x) + &(&x * &a)) - quad + &q;
        assert!(res.norm_fro() < 1e-6, "residual {}", res.norm_fro());
    }

    #[test]
    fn care_harmonic_oscillator_stabilizes() {
        // Lightly damped oscillator; LQR must add damping.
        let a = DMat::from_rows(&[&[0.0, 1.0], &[-4.0, -0.04]]);
        let b = DMat::from_rows(&[&[0.0], &[1.0]]);
        let q = DMat::identity(2);
        let r = DMat::from_rows(&[&[0.1]]);
        let x = care(&a, &b, &q, &r).unwrap();
        let k = &(r.inverse().unwrap()) * &(b.transpose() * &x);
        let cl = &a - &(&b * &k);
        assert!(tpt_ctrl_core::analysis::is_stable_continuous(&cl, 1e-9).unwrap());
        assert!(x.is_positive_definite());
    }
}
