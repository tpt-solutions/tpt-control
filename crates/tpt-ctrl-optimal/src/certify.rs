//! Self-certifying solves: after a Riccati solve, check the result without
//! trusting the solver that produced it.
//!
//! A [`Certificate`] reports
//!
//! * the **relative Riccati residual** — how far `X` is from satisfying the
//!   algebraic equation (`‖R(X)‖_F / (1 + ‖X‖_F)`), and
//! * the **closed-loop margin** of `A − BK` — `−max Re(λ)` for continuous
//!   plants, `1 − max |λ|` for discrete ones. Positive means stable.
//!
//! Both numbers are cheap (one residual evaluation and one eigen-solve), so a
//! controller can refuse to deploy a gain whose certificate fails.

use tpt_ctrl_core::analysis::poles;
use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_core::error::Result;

/// Verification report for an LQR/Riccati solution.
#[derive(Clone, Debug, PartialEq)]
pub struct Certificate {
    /// Relative Frobenius residual of the Riccati equation.
    pub riccati_residual: f64,
    /// Closed-loop stability margin (positive = stable).
    pub closed_loop_margin: f64,
    /// `true` when `closed_loop_margin > 0`.
    pub stable: bool,
}

impl Certificate {
    /// The verdict for inputs that are not finite: never certified.
    fn invalid() -> Certificate {
        Certificate {
            riccati_residual: f64::INFINITY,
            closed_loop_margin: f64::NEG_INFINITY,
            stable: false,
        }
    }

    /// `true` when the residual is at most `tol` and the loop is stable.
    pub fn is_certified(&self, tol: f64) -> bool {
        self.stable && self.riccati_residual <= tol
    }
}

/// Certifies a continuous-time solution `(X, K)` of the CARE for `(A, B, Q, R)`.
pub fn certify_continuous(
    a: &DMat<f64>,
    b: &DMat<f64>,
    q: &DMat<f64>,
    r: &DMat<f64>,
    x: &DMat<f64>,
    k: &DMat<f64>,
) -> Result<Certificate> {
    let cl = a - &(b * k);
    if !all_finite(x) || !all_finite(&cl) {
        return Ok(Certificate::invalid());
    }
    // R_c(X) = AᵀX + XA − XBR⁻¹BᵀX + Q
    let bt_x = &b.transpose() * x;
    let residual =
        &(&(&(&a.transpose() * x) + &(x * a)) - &(&(&(x * b) * &r.inverse()?) * &bt_x)) + q;
    let margin = {
        let mut worst = f64::NEG_INFINITY;
        for p in poles(&cl)? {
            worst = worst.max(p.re);
        }
        -worst
    };
    Ok(Certificate {
        riccati_residual: relative_norm(&residual, x),
        closed_loop_margin: margin,
        stable: margin > 0.0,
    })
}

/// Certifies a discrete-time solution `(X, K)` of the DARE for `(A, B, Q, R)`.
pub fn certify_discrete(
    a: &DMat<f64>,
    b: &DMat<f64>,
    q: &DMat<f64>,
    r: &DMat<f64>,
    x: &DMat<f64>,
    k: &DMat<f64>,
) -> Result<Certificate> {
    let cl = a - &(b * k);
    if !all_finite(x) || !all_finite(&cl) {
        return Ok(Certificate::invalid());
    }
    // R_d(X) = AᵀXA − X − AᵀXB(R + BᵀXB)⁻¹BᵀXA + Q
    let at_x = &a.transpose() * x;
    let bt_x = &b.transpose() * x;
    let s = &(r + &(&bt_x * b)).inverse()?;
    let at_x_b = &at_x * b;
    let correction = &(&(&at_x_b * s) * &b.transpose()) * &(x * a);
    let residual = &(&(&(&at_x * a) - x) - &correction) + q;
    let margin = {
        let mut worst = 0.0f64;
        for p in poles(&cl)? {
            worst = worst.max((p.re * p.re + p.im * p.im).sqrt());
        }
        1.0 - worst
    };
    Ok(Certificate {
        riccati_residual: relative_norm(&residual, x),
        closed_loop_margin: margin,
        stable: margin > 0.0,
    })
}

fn all_finite(m: &DMat<f64>) -> bool {
    m.as_slice().iter().all(|v| v.is_finite())
}

fn relative_norm(residual: &DMat<f64>, x: &DMat<f64>) -> f64 {
    residual.norm_fro() / (1.0 + x.norm_fro())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dlqr, lqr};

    fn double_integrator() -> (DMat<f64>, DMat<f64>, DMat<f64>, DMat<f64>) {
        (
            DMat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]),
            DMat::from_rows(&[&[0.0], &[1.0]]),
            DMat::identity(2),
            DMat::from_rows(&[&[1.0]]),
        )
    }

    #[test]
    fn continuous_lqr_is_certified() {
        let (a, b, q, r) = double_integrator();
        let sol = lqr(&a, &b, &q, &r).unwrap();
        let cert = certify_continuous(&a, &b, &q, &r, &sol.x, &sol.k).unwrap();
        assert!(cert.is_certified(1e-8), "{cert:?}");
        // Closed-loop poles of the double integrator with this gain sit at
        // -sqrt(3)/2 ± i/2, so the margin is sqrt(3)/2.
        assert!((cert.closed_loop_margin - 3.0f64.sqrt() / 2.0).abs() < 1e-6);
    }

    #[test]
    fn discrete_lqr_is_certified() {
        let a = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
        let b = DMat::from_rows(&[&[0.005], &[0.1]]);
        let (_, _, q, r) = double_integrator();
        let sol = dlqr(&a, &b, &q, &r).unwrap();
        let cert = certify_discrete(&a, &b, &q, &r, &sol.x, &sol.k).unwrap();
        assert!(cert.is_certified(1e-8), "{cert:?}");
        assert!(cert.closed_loop_margin > 0.0 && cert.closed_loop_margin < 1.0);
    }

    #[test]
    fn corrupted_solution_fails_certificate() {
        let (a, b, q, r) = double_integrator();
        let sol = lqr(&a, &b, &q, &r).unwrap();
        let bad_x = sol.x.clone() * 1.5;
        let cert = certify_continuous(&a, &b, &q, &r, &bad_x, &sol.k).unwrap();
        assert!(cert.riccati_residual > 1e-3);
        assert!(!cert.is_certified(1e-8));
    }

    #[test]
    fn non_finite_solution_fails_closed() {
        let (a, b, q, r) = double_integrator();
        let nan = DMat::from_rows(&[&[f64::NAN, f64::NAN]]);
        let cert = certify_continuous(
            &a,
            &b,
            &q,
            &r,
            &DMat::from_rows(&[&[f64::NAN, 0.0], &[0.0, 1.0]]),
            &nan,
        )
        .unwrap();
        assert!(!cert.stable);
        assert!(!cert.is_certified(f64::MAX));
    }

    #[test]
    fn unstable_gain_is_flagged() {
        let (a, b, q, r) = double_integrator();
        let sol = lqr(&a, &b, &q, &r).unwrap();
        let wrong_k = sol.k.clone() * -1.0;
        let cert = certify_continuous(&a, &b, &q, &r, &sol.x, &wrong_k).unwrap();
        assert!(!cert.stable);
        assert!(!cert.is_certified(1.0));
    }
}
