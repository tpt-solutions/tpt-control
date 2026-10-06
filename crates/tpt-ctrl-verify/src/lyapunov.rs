//! Numerical Lyapunov stability contracts.
//!
//! spec.txt requires documented, tested Lyapunov conditions: `V(x) > 0`
//! (positive definiteness) and `V'(x) < 0` (strict decrease along
//! trajectories). This module provides the runtime checks used in debug
//! builds and the vocabulary the Kani proofs reuse.

use tpt_ctrl_core::dmat::DMat;

/// A candidate Lyapunov function `V` with its time derivative along the
/// vector field.
pub trait LyapunovFunction {
    /// Evaluates `V(x)`.
    fn v(&self, x: &[f64]) -> f64;
    /// Evaluates `V'(x)` along the system's vector field at state `x`.
    fn v_dot(&self, x: &[f64]) -> f64;
}

/// Quadratic Lyapunov function `V(x) = x^T P x` bound to a vector field
/// `x' = f(x)`.
pub struct QuadraticWithField<'p, F>
where
    F: Fn(&[f64], &mut [f64]),
{
    p: &'p DMat<f64>,
    field: F,
}

impl<'p, F> QuadraticWithField<'p, F>
where
    F: Fn(&[f64], &mut [f64]),
{
    /// Binds `P` and the closed-loop vector field.
    pub fn new(p: &'p DMat<f64>, field: F) -> QuadraticWithField<'p, F> {
        QuadraticWithField { p, field }
    }
}

impl<F> LyapunovFunction for QuadraticWithField<'_, F>
where
    F: Fn(&[f64], &mut [f64]),
{
    fn v(&self, x: &[f64]) -> f64 {
        let n = self.p.nrows();
        let col = DMat::from_fn(n, 1, |i, _| x[i]);
        let v = &(&col.transpose() * self.p) * &col;
        v[(0, 0)]
    }

    fn v_dot(&self, x: &[f64]) -> f64 {
        // V' = 2 x^T P x' computed numerically from the field.
        let n = self.p.nrows();
        let mut xdot = vec![0.0; n];
        (self.field)(x, &mut xdot);
        let col = DMat::from_fn(n, 1, |i, _| x[i]);
        let d = DMat::from_fn(n, 1, |i, _| xdot[i]);
        let v = &(&col.transpose() * self.p) * &d;
        2.0 * v[(0, 0)]
    }
}

/// Result of sampling a Lyapunov contract over a state set.
#[derive(Debug, Clone, PartialEq)]
pub struct LyapunovReport {
    /// Number of sample points checked.
    pub samples: usize,
    /// Smallest observed `V(x)`.
    pub min_v: f64,
    /// Largest observed `V'(x)` (must stay negative for asymptotic
    /// stability of the sampled region).
    pub max_v_dot: f64,
}

/// Samples `V` and `V'` over the box `bounds[i].0 <= x[i] <= bounds[i].1`
/// using `k` uniform points per axis, checking `V > v_tol` and
/// `V' < -v_dot_tol`.
///
/// This is a *numerical* contract over a sampled region — a witness for
/// bounded regions, not a global proof (Kani harnesses in [`crate::proofs`]
/// and analytic arguments cover the exact statements).
pub fn check_lyapunov_contract<L: LyapunovFunction>(
    l: &L,
    bounds: &[(f64, f64)],
    points_per_axis: usize,
    v_tol: f64,
    v_dot_tol: f64,
) -> Result<LyapunovReport, String> {
    let n = bounds.len();
    let mut report = LyapunovReport {
        samples: 0,
        min_v: f64::INFINITY,
        max_v_dot: f64::NEG_INFINITY,
    };
    let mut x = vec![0.0; n];
    sample_rec(
        l,
        bounds,
        points_per_axis,
        &mut x,
        0,
        &mut report,
        v_tol,
        v_dot_tol,
    )?;
    if report.samples == 0 {
        return Err("no sample points generated".to_string());
    }
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn sample_rec<L: LyapunovFunction>(
    l: &L,
    bounds: &[(f64, f64)],
    points: usize,
    x: &mut [f64],
    axis: usize,
    report: &mut LyapunovReport,
    v_tol: f64,
    v_dot_tol: f64,
) -> Result<(), String> {
    if axis == x.len() {
        // Lyapunov conditions hold on the *punctured* neighborhood — the
        // origin itself is excluded (V(0) = 0 there by definition).
        if x.iter().all(|v| *v == 0.0) {
            return Ok(());
        }
        report.samples += 1;
        let v = l.v(x);
        let vd = l.v_dot(x);
        if !v.is_finite() || !vd.is_finite() {
            return Err(format!("non-finite Lyapunov value at x = {x:?}"));
        }
        report.min_v = report.min_v.min(v);
        report.max_v_dot = report.max_v_dot.max(vd);
        if v <= v_tol {
            return Err(format!("V(x) = {v} not positive at x = {x:?}"));
        }
        if vd >= -v_dot_tol {
            return Err(format!("V'(x) = {vd} not decreasing at x = {x:?}"));
        }
        return Ok(());
    }
    let (lo, hi) = bounds[axis];
    for k in 0..points {
        let frac = if points == 1 {
            0.5
        } else {
            k as f64 / (points as f64 - 1.0)
        };
        x[axis] = lo + (hi - lo) * frac;
        sample_rec(l, bounds, points, x, axis + 1, report, v_tol, v_dot_tol)?;
    }
    Ok(())
}

/// Convenience: checks `P` symmetric positive definite (the `V > 0`
/// condition for quadratic Lyapunov functions) via Cholesky.
pub fn p_is_positive_definite(p: &DMat<f64>, tol: f64) -> bool {
    let n = p.nrows();
    for i in 0..n {
        for j in 0..n {
            if (p[(i, j)] - p[(j, i)]).abs() > tol * (p[(i, i)].abs() + 1.0) {
                return false;
            }
        }
    }
    p.is_positive_definite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::assert_close;

    #[test]
    fn stable_harmonic_damped_system_passes() {
        // Damped oscillator x1' = x2, x2' = -4 x1 - 0.4 x2.
        let field = |x: &[f64], out: &mut [f64]| {
            out[0] = x[1];
            out[1] = -4.0 * x[0] - 0.4 * x[1];
        };
        // P solving A^T P + P A = -I for A = [[0, 1], [-4, -0.4]]:
        // p12 = 1/8, p22 = 25/16, p11 = 63/10.
        let p = DMat::from_rows(&[&[6.3, 0.125], &[0.125, 1.5625]]);
        let l = QuadraticWithField::new(&p, field);
        let report = check_lyapunov_contract(&l, &[(-1.0, 1.0), (-1.0, 1.0)], 5, 1e-9, 1e-9);
        let report = report.unwrap();
        assert!(report.min_v > 0.0);
        assert!(report.max_v_dot < 0.0);
    }

    #[test]
    fn unstable_system_fails_contract() {
        let field = |x: &[f64], out: &mut [f64]| {
            out[0] = x[0];
            out[1] = x[1];
        };
        let p = DMat::identity(2);
        let l = QuadraticWithField::new(&p, field);
        assert!(check_lyapunov_contract(&l, &[(-1.0, 1.0), (-1.0, 1.0)], 3, 1e-9, 1e-9).is_err());
    }

    #[test]
    fn quadratic_value_matches_hand_computation() {
        let p = DMat::from_rows(&[&[2.0, 0.0], &[0.0, 3.0]]);
        let field = |_: &[f64], _: &mut [f64]| {};
        let l = QuadraticWithField::new(&p, field);
        assert_close(l.v(&[1.0, 2.0]), 14.0, 1e-12);
    }
}
