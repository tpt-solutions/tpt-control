//! Reduction to upper Hessenberg form via Householder similarity transforms.

use alloc::vec::Vec;

use crate::dmat::{require_square, DMat};
use crate::error::Result;
use crate::field::Field;

/// Reduces `a` to upper Hessenberg form, returning `(H, Q)` with
/// `A = Q H Q^T` and `Q` orthogonal.
pub fn hessenberg<T: Field>(a: &DMat<T>) -> Result<(DMat<T>, DMat<T>)> {
    require_square(a)?;
    let n = a.nrows();
    let mut h = a.clone();
    let mut q = DMat::identity(n);

    for k in 0..n.saturating_sub(2) {
        // Householder vector for rows k+1..n of column k.
        let mut norm = T::zero();
        for i in k + 1..n {
            norm = norm.hypot(h[(i, k)]);
        }
        if norm == T::zero() {
            continue;
        }
        let alpha = if h[(k + 1, k)] >= T::zero() {
            -norm
        } else {
            norm
        };
        let mut v: Vec<T> = (k + 1..n).map(|i| h[(i, k)]).collect();
        v[0] -= alpha;
        let mut vv = T::zero();
        for x in &v {
            vv += *x * *x;
        }
        if vv == T::zero() {
            continue;
        }
        let beta = T::from_f64(2.0) / vv;
        let lo = k + 1;

        // H <- P H P with P = I - beta v v^T acting on rows/cols lo..n.
        // Left: rows lo..n, all columns.
        for j in 0..n {
            let mut s = T::zero();
            for (idx, i) in (lo..n).enumerate() {
                s += v[idx] * h[(i, j)];
            }
            s *= beta;
            for (idx, i) in (lo..n).enumerate() {
                h[(i, j)] -= s * v[idx];
            }
        }
        // Right: all rows, columns lo..n.
        for i in 0..n {
            let mut s = T::zero();
            for (idx, j) in (lo..n).enumerate() {
                s += h[(i, j)] * v[idx];
            }
            s *= beta;
            for (idx, j) in (lo..n).enumerate() {
                h[(i, j)] -= s * v[idx];
            }
        }
        // Accumulate Q <- Q P.
        for i in 0..n {
            let mut s = T::zero();
            for (idx, j) in (lo..n).enumerate() {
                s += q[(i, j)] * v[idx];
            }
            s *= beta;
            for (idx, j) in (lo..n).enumerate() {
                q[(i, j)] -= s * v[idx];
            }
        }
        // Exact zeros below the subdiagonal.
        h[(lo, k)] = alpha;
        for i in lo + 1..n {
            h[(i, k)] = T::zero();
        }
    }
    Ok((h, q))
}

impl<T: Field> DMat<T> {
    /// Hessenberg reduction `A = Q H Q^T` (see [`hessenberg`]).
    pub fn hessenberg(&self) -> Result<(DMat<T>, DMat<T>)> {
        hessenberg(self)
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
    fn similarity_reconstruction() {
        let a = DMat::from_rows(&[
            &[4.0, 1.0, -2.0, 2.0],
            &[1.0, 2.0, 0.0, -1.0],
            &[-2.0, 0.0, 3.0, 1.0],
            &[2.0, -1.0, 1.0, -1.0],
        ]);
        let (h, q) = hessenberg(&a).unwrap();
        let qhq = &(&q * &h) * &q.transpose();
        assert!(mat_close(&qhq, &a));
        // Q orthogonal.
        assert!(mat_close(&(&q.transpose() * &q), &DMat::identity(4)));
        // H is upper Hessenberg.
        for i in 2..h.nrows() {
            for j in 0..i - 1 {
                assert!(h[(i, j)].abs() < 1e-12);
            }
        }
    }

    #[test]
    fn small_matrices_unchanged() {
        let a = DMat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let (h, q) = hessenberg(&a).unwrap();
        assert!(mat_close(&h, &a));
        assert!(mat_close(&q, &DMat::identity(2)));
    }
}
