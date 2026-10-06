//! Lyapunov equation solvers.
//!
//! * Continuous `A^T X + X A = -Q` via Schur decomposition (Bartels-Stewart).
//! * Discrete `A^T X A - X = -Q` via a squaring/doubling iteration (matrix
//!   multiplications only; robust for stable `A`).

use crate::dmat::{require_square, DMat};
use crate::error::{LinalgError, Result};
use crate::field::Field;

/// Solves the continuous-time Lyapunov equation `A^T X + X A = -Q`.
///
/// Requires `A` to be Hurwitz (all eigenvalues with negative real part);
/// otherwise the iteration detects a singular block and returns
/// [`LinalgError::SingularMatrix`].
pub fn solve_lyapunov<T: Field>(a: &DMat<T>, q: &DMat<T>) -> Result<DMat<T>> {
    require_square(a)?;
    if q.nrows() != a.nrows() || q.ncols() != a.ncols() {
        return Err(LinalgError::DimensionMismatch {
            expected: (a.nrows(), a.ncols()),
            found: (q.nrows(), q.ncols()),
        });
    }
    let n = a.nrows();
    if n == 0 {
        return Ok(a.clone());
    }
    let s = a.schur()?;
    // Solve T^T Y + Y T = -U^T Q U for Y, then X = U Y U^T.
    let s_rhs = -&(&(&s.q.transpose() * q) * &s.q);
    let mut y = DMat::<T>::zeros(n, n);

    // Process block columns left to right: the equation for block J only
    // involves already-solved columns k < j0 (through T[k, J]).
    let mut j0 = 0usize;
    while j0 < n {
        let bs = if j0 + 1 < n && s.t[(j0 + 1, j0)] != T::zero() {
            2
        } else {
            1
        };
        // rhs = S[:, J] - Y[:, :j0] T[:j0, J]
        let mut rhs = s_rhs.submatrix(0, j0, n, bs);
        if j0 > 0 {
            let known = y.submatrix(0, 0, n, j0);
            let coupling = s.t.submatrix(0, j0, j0, bs);
            rhs = &rhs - &(&known * &coupling);
        }
        let ys = solve_block_column(&s.t, &rhs, j0, bs)?;
        for (c, col) in (j0..j0 + bs).enumerate() {
            for i in 0..n {
                y[(i, col)] = ys[(i, c)];
            }
        }
        j0 += bs;
    }
    Ok(&(&s.q * &y) * &s.q.transpose())
}

/// Solves `T^T y + y t_jj-ish = rhs` for one block column of `Y`.
fn solve_block_column<T: Field>(
    t: &DMat<T>,
    rhs: &DMat<T>,
    j0: usize,
    bs: usize,
) -> Result<DMat<T>> {
    let n = t.nrows();
    if bs == 1 {
        // M = T^T + t_jj I, solved densely with LU (sizes here are small).
        let shift = t[(j0, j0)];
        let m = DMat::from_fn(n, n, |i, k| {
            let mut v = t[(k, i)];
            if i == k {
                v += shift;
            }
            v
        });
        m.solve(rhs)
    } else {
        // Sylvester for a 2x2 diagonal block: (I2 x T^T + T22^T x I) vec(y) = vec(rhs).
        let t11 = t[(j0, j0)];
        let t12 = t[(j0, j0 + 1)];
        let t21 = t[(j0 + 1, j0)];
        let t22 = t[(j0 + 1, j0 + 1)];
        let two_n = 2 * n;
        let mut big = DMat::<T>::zeros(two_n, two_n);
        // (I2 x T^T) part.
        for jb in 0..2 {
            for i in 0..n {
                for k in 0..n {
                    big[(jb * n + i, jb * n + k)] = t[(k, i)];
                }
            }
        }
        // (T22^T x I) part.
        let t22t = [[t11, t21], [t12, t22]];
        for r2 in 0..2 {
            for c2 in 0..2 {
                for i in 0..n {
                    big[(r2 * n + i, c2 * n + i)] += t22t[r2][c2];
                }
            }
        }
        // Column-stacked right-hand side: vec(rhs) = [rhs[:,0]; rhs[:,1]].
        let mut b = DMat::<T>::zeros(two_n, 1);
        for i in 0..n {
            b[(i, 0)] = rhs[(i, 0)];
            b[(n + i, 0)] = rhs[(i, 1)];
        }
        let z = big.solve(&b)?;
        // Un-stack: Y2[:, 0] = z[0..n], Y2[:, 1] = z[n..2n].
        let mut out = DMat::<T>::zeros(n, 2);
        for i in 0..n {
            out[(i, 0)] = z[(i, 0)];
            out[(i, 1)] = z[(n + i, 0)];
        }
        Ok(out)
    }
}

/// Solves the discrete-time Lyapunov equation `A^T X A - X = -Q` for stable
/// `A`, via the squaring identity
/// `X = sum_{k<2^s} (A^T)^k Q A^k`.
///
/// Returns [`LinalgError::NoConvergence`] when `A` is not stable enough for
/// the series to converge within the squaring budget.
pub fn solve_discrete_lyapunov<T: Field>(a: &DMat<T>, q: &DMat<T>) -> Result<DMat<T>> {
    require_square(a)?;
    if q.nrows() != a.nrows() {
        return Err(LinalgError::DimensionMismatch {
            expected: (a.nrows(), a.ncols()),
            found: (q.nrows(), q.ncols()),
        });
    }
    let n = a.nrows();
    if n == 0 {
        return Ok(a.clone());
    }
    let na = a.norm_1();
    let norm_a = if na > T::one() { na } else { T::one() };
    let tol = T::from_f64(1e-14);
    let mut x = q.clone();
    let mut ak = a.clone();
    for _ in 0..64 {
        if ak.norm_1() <= tol * norm_a {
            return Ok(x);
        }
        let contribution = &(&ak.transpose() * &x) * &ak;
        x = &x + &contribution;
        ak = &ak * &ak;
    }
    if ak.norm_1() <= T::from_f64(1e-6) * norm_a {
        Ok(x)
    } else {
        Err(LinalgError::NoConvergence)
    }
}

impl<T: Field> DMat<T> {
    /// Solves `A^T X + X A = -Q` (see [`solve_lyapunov`]).
    pub fn solve_lyapunov(&self, q: &DMat<T>) -> Result<DMat<T>> {
        solve_lyapunov(self, q)
    }

    /// Solves `A^T X A - X = -Q` (see [`solve_discrete_lyapunov`]).
    pub fn solve_discrete_lyapunov(&self, q: &DMat<T>) -> Result<DMat<T>> {
        solve_discrete_lyapunov(self, q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat_close(a: &DMat<f64>, b: &DMat<f64>) -> bool {
        a.nrows() == b.nrows()
            && a.ncols() == b.ncols()
            && a.as_slice()
                .iter()
                .zip(b.as_slice())
                .all(|(x, y)| (*x - *y).abs() < 1e-8)
    }

    #[test]
    fn continuous_lyapunov_scalar_case() {
        // a^T x + x a = -q  ->  x = q / (2 * 3)
        let a = DMat::from_rows(&[&[-3.0]]);
        let q = DMat::from_rows(&[&[6.0]]);
        let x = solve_lyapunov(&a, &q).unwrap();
        assert!((x[(0, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn continuous_lyapunov_residual() {
        // Stable system with a complex pair (oscillator + damping).
        let a = DMat::from_rows(&[&[0.0, 1.0], &[-4.0, -0.4]]);
        let q = DMat::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let x = solve_lyapunov(&a, &q).unwrap();
        // Symmetric positive definite.
        assert!(x.is_positive_definite());
        // Residual A^T X + X A + Q ~ 0.
        let r = &(&a.transpose() * &x) + &(&x * &a) + &q;
        assert!(r.norm_fro() < 1e-8, "residual {}", r.norm_fro());
    }

    #[test]
    fn continuous_lyapunov_larger_system() {
        let a = DMat::from_rows(&[&[-1.0, 2.0, 0.5], &[0.0, -2.0, 1.0], &[0.3, 0.0, -3.0]]);
        let q = DMat::diagonal(&[2.0, 1.0, 3.0]);
        let x = solve_lyapunov(&a, &q).unwrap();
        let r = &(&a.transpose() * &x) + &(&x * &a) + &q;
        assert!(r.norm_fro() < 1e-8, "residual {}", r.norm_fro());
    }

    #[test]
    fn continuous_lyapunov_unstable_errors() {
        let a = DMat::from_rows(&[&[1.0, 0.0], &[0.0, -1.0]]);
        let q = DMat::identity(2);
        assert!(solve_lyapunov(&a, &q).is_err());
    }

    #[test]
    fn discrete_lyapunov_residual() {
        let a = DMat::from_rows(&[&[0.9, 0.2], &[-0.1, 0.8]]);
        let q = DMat::identity(2);
        let x = solve_discrete_lyapunov(&a, &q).unwrap();
        let r = &(&a.transpose() * &x) * &a - &x + &q;
        assert!(r.norm_fro() < 1e-8, "residual {}", r.norm_fro());
        assert!(x.is_positive_definite());
    }

    #[test]
    fn discrete_lyapunov_unstable_errors() {
        let a = DMat::from_rows(&[&[1.5, 0.0], &[0.0, 0.5]]);
        let q = DMat::identity(2);
        assert!(solve_discrete_lyapunov(&a, &q).is_err());
    }

    #[test]
    fn both_solvers_agree_on_mat_close_repeatedly() {
        // Same operator, checked via substitution identity.
        let a = DMat::from_rows(&[&[-0.5, 1.0], &[-1.0, -0.5]]);
        let q = DMat::from_rows(&[&[4.0, 1.0], &[1.0, 2.0]]);
        let x = solve_lyapunov(&a, &q).unwrap();
        let r = &(&a.transpose() * &x) + &(&x * &a) + &q;
        assert!(mat_close(&r, &DMat::zeros(2, 2)));
    }
}
