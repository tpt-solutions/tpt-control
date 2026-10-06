//! Cholesky factorization for symmetric positive definite matrices.

use crate::dmat::{require_square, DMat};
use crate::error::{LinalgError, Result};
use crate::field::Field;

/// Computes the lower-triangular `L` with `A = L L^T`.
///
/// Only the lower triangle of `a` is read. Returns
/// [`LinalgError::NotPositiveDefinite`] when a non-positive pivot appears.
pub fn cholesky<T: Field>(a: &DMat<T>) -> Result<DMat<T>> {
    require_square(a)?;
    let n = a.nrows();
    let mut l = DMat::zeros(n, n);
    for j in 0..n {
        let mut s = a[(j, j)];
        for k in 0..j {
            let v = l[(j, k)];
            s -= v * v;
        }
        // `!(s > 0)` (not `s <= 0`) also rejects NaN entries.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(s > T::zero()) {
            return Err(LinalgError::NotPositiveDefinite);
        }
        let ljj = s.sqrt();
        l[(j, j)] = ljj;
        for i in j + 1..n {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            l[(i, j)] = s / ljj;
        }
    }
    Ok(l)
}

/// Solves `A x = b` given the Cholesky factor `L` of `A`
/// (forward substitution with `L`, back substitution with `L^T`).
pub fn cholesky_solve<T: Field>(l: &DMat<T>, b: &DMat<T>) -> Result<DMat<T>> {
    if l.nrows() != b.nrows() {
        return Err(LinalgError::DimensionMismatch {
            expected: (l.nrows(), l.ncols()),
            found: (b.nrows(), b.ncols()),
        });
    }
    let n = l.nrows();
    let mut x = b.clone();
    // L y = b (forward)
    for i in 0..n {
        for j in 0..b.ncols() {
            let mut acc = x[(i, j)];
            for k in 0..i {
                acc -= l[(i, k)] * x[(k, j)];
            }
            x[(i, j)] = acc / l[(i, i)];
        }
    }
    // L^T x = y (backward)
    for i in (0..n).rev() {
        for j in 0..b.ncols() {
            let mut acc = x[(i, j)];
            for k in i + 1..n {
                acc -= l[(k, i)] * x[(k, j)];
            }
            x[(i, j)] = acc / l[(i, i)];
        }
    }
    Ok(x)
}

impl<T: Field> DMat<T> {
    /// Cholesky factorization of a symmetric positive definite matrix.
    pub fn cholesky(&self) -> Result<DMat<T>> {
        cholesky(self)
    }

    /// Solves a symmetric positive definite system via Cholesky.
    pub fn solve_spd(&self, b: &DMat<T>) -> Result<DMat<T>> {
        let l = cholesky(self)?;
        cholesky_solve(&l, b)
    }

    /// `true` when the matrix is (numerically) symmetric positive definite.
    pub fn is_positive_definite(&self) -> bool {
        cholesky(self).is_ok()
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
                .all(|(x, y)| (*x - *y).abs() < 1e-10)
    }

    #[test]
    fn factorization_reconstructs() {
        let a = DMat::from_rows(&[&[4.0, 2.0, -2.0], &[2.0, 10.0, 2.0], &[-2.0, 2.0, 5.0]]);
        let l = cholesky(&a).unwrap();
        assert!(mat_close(&(&l * &l.transpose()), &a));
    }

    #[test]
    fn rejects_indefinite() {
        let a = DMat::from_rows(&[&[1.0, 2.0], &[2.0, 1.0]]);
        assert_eq!(cholesky(&a), Err(LinalgError::NotPositiveDefinite));
        assert!(!a.is_positive_definite());
    }

    #[test]
    fn solve_spd_matches_lu() {
        let a = DMat::from_rows(&[&[4.0, 2.0], &[2.0, 3.0]]);
        let b = DMat::from_rows(&[&[1.0], &[-1.0]]);
        let x_spd = a.solve_spd(&b).unwrap();
        let x_lu = a.solve(&b).unwrap();
        assert!(mat_close(&x_spd, &x_lu));
    }

    #[test]
    fn diagonal_matrix_is_spd_when_positive() {
        let a = DMat::diagonal(&[1.0, 2.0, 3.0]);
        assert!(a.is_positive_definite());
        let neg = DMat::diagonal(&[1.0, -2.0, 3.0]);
        assert!(!neg.is_positive_definite());
    }
}
