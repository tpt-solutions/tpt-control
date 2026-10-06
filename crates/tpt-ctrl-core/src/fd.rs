//! Finite-difference differentiation hooks (the "AI-Native Agents"
//! integration point from spec.txt).
//!
//! Cost functions and state transitions expose gradient information to
//! external optimizers (e.g. reinforcement-learning agents) without
//! requiring a full automatic-differentiation pass: central finite
//! differences give a Jacobian anywhere a vector-valued function is
//! evaluable. This is the portable *fallback*; when one of the
//! `tpt-math-autodiff` crates is wired in, its exact Jacobians can replace
//! these numerically (same call shape).

use alloc::vec;
use alloc::vec::Vec;

use crate::dmat::DMat;

/// Central-difference Jacobian of a vector-valued function.
///
/// `f` maps an `n`-vector to an `m`-vector; the result is the `m x n`
/// matrix with `J[(i, j)] = df_i/dx_j` approximated with step
/// `eps * max(1, |x_j|)` per coordinate.
pub fn jacobian<F>(f: F, x: &[f64], eps: f64) -> DMat<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    assert!(eps > 0.0, "eps must be positive");
    let n = x.len();
    let f0 = f(x);
    let m = f0.len();
    let mut out = DMat::zeros(m, n);
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    for j in 0..n {
        let h = eps * (1.0 + x[j].abs());
        xp[j] = x[j] + h;
        xm[j] = x[j] - h;
        let fp = f(&xp);
        let fm = f(&xm);
        for i in 0..m {
            out[(i, j)] = (fp[i] - fm[i]) / (2.0 * h);
        }
        xp[j] = x[j];
        xm[j] = x[j];
    }
    out
}

/// Finite-difference gradient of a scalar cost — a convenience wrapper
/// over [`jacobian`] for the common `f: R^n -> R` case (agents optimizing
/// a cost function).
pub fn gradient<F>(f: F, x: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let col = jacobian(|xs| vec![f(xs)], x, eps);
    (0..x.len()).map(|j| col[(0, j)]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ctrl_verify::testing::assert_close;

    #[test]
    fn jacobian_of_linear_map_is_exact() {
        // f(x) = A x with A = [[1, 2], [3, 4]].
        let a = [(1.0, 2.0), (3.0, 4.0)];
        let f = |x: &[f64]| -> Vec<f64> {
            vec![a[0].0 * x[0] + a[0].1 * x[1], a[1].0 * x[0] + a[1].1 * x[1]]
        };
        let j = jacobian(f, &[0.7, -1.2], 1e-6);
        assert_close(j[(0, 0)], 1.0, 1e-6);
        assert_close(j[(0, 1)], 2.0, 1e-6);
        assert_close(j[(1, 0)], 3.0, 1e-6);
        assert_close(j[(1, 1)], 4.0, 1e-6);
    }

    #[test]
    fn gradient_of_quadratic() {
        // f(x) = x1^2 + 3 x2^2: grad = (2 x1, 6 x2).
        let f = |x: &[f64]| x[0] * x[0] + 3.0 * x[1] * x[1];
        let g = gradient(f, &[1.5, -0.7], 1e-6);
        assert_close(g[0], 3.0, 1e-4);
        assert_close(g[1], -4.2, 1e-4);
    }

    #[test]
    fn jacobian_of_nonlinear_matches_analytic() {
        // f(x) = [sin(x1) * x2, x1 + x2^2]
        let f = |x: &[f64]| -> Vec<f64> { vec![x[0].sin() * x[1], x[0] + x[1] * x[1]] };
        let x = [0.6, -1.1];
        let j = jacobian(f, &x, 1e-6);
        assert_close(j[(0, 0)], x[0].cos() * x[1], 1e-6);
        assert_close(j[(0, 1)], x[0].sin(), 1e-6);
        assert_close(j[(1, 0)], 1.0, 1e-6);
        assert_close(j[(1, 1)], 2.0 * x[1], 1e-6);
    }
}
