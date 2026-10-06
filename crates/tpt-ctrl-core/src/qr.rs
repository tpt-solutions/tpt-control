//! Householder QR decomposition.

use alloc::vec::Vec;

use crate::dmat::DMat;
use crate::field::Field;

/// Computes the full QR decomposition `A = Q R` via Householder reflectors.
///
/// `Q` is `m x m` orthogonal, `R` is `m x n` upper triangular.
pub fn qr<T: Field>(a: &DMat<T>) -> (DMat<T>, DMat<T>) {
    let (m, n) = (a.nrows(), a.ncols());
    let mut r = a.clone();
    let mut q = DMat::identity(m);

    let kmax = if m - 1 < n { m - 1 } else { n };
    for k in 0..kmax {
        // Householder vector for column k, rows k..m.
        let mut norm = T::zero();
        for i in k..m {
            norm = norm.hypot(r[(i, k)]);
        }
        if norm == T::zero() {
            continue;
        }
        let alpha = if r[(k, k)] >= T::zero() { -norm } else { norm };
        let mut v: Vec<T> = (k..m).map(|i| r[(i, k)]).collect();
        v[0] -= alpha;
        let mut vv = T::zero();
        for x in &v {
            vv += *x * *x;
        }
        if vv == T::zero() {
            continue;
        }
        let beta = T::from_f64(2.0) / vv;

        // R[k.., k..n] <- (I - beta v v^T) R[k.., k..n]
        for j in k..n {
            let mut s = T::zero();
            for (idx, i) in (k..m).enumerate() {
                s += v[idx] * r[(i, j)];
            }
            s *= beta;
            for (idx, i) in (k..m).enumerate() {
                r[(i, j)] -= s * v[idx];
            }
        }
        // Q[:, k..m] <- Q[:, k..m] (I - beta v v^T)
        for i in 0..m {
            let mut s = T::zero();
            for (idx, j) in (k..m).enumerate() {
                s += q[(i, j)] * v[idx];
            }
            s *= beta;
            for (idx, j) in (k..m).enumerate() {
                q[(i, j)] -= s * v[idx];
            }
        }
        // Enforce exact zeros below the diagonal of R.
        r[(k, k)] = alpha;
        for i in k + 1..m {
            r[(i, k)] = T::zero();
        }
    }
    (q, r)
}

impl<T: Field> DMat<T> {
    /// Full QR decomposition `A = Q R` (see [`qr`]).
    pub fn qr(&self) -> (DMat<T>, DMat<T>) {
        qr(self)
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
    fn reconstructs_product() {
        let a = DMat::from_rows(&[
            &[12.0, -51.0, 4.0],
            &[6.0, 167.0, -68.0],
            &[-4.0, 24.0, -41.0],
        ]);
        let (q, r) = qr(&a);
        assert!(mat_close(&(&q * &r), &a));
    }

    #[test]
    fn q_is_orthogonal_r_is_upper() {
        let a = DMat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        let (q, r) = qr(&a);
        let qtq = &q.transpose() * &q;
        assert!(mat_close(&qtq, &DMat::identity(3)));
        for i in 1..r.nrows() {
            for j in 0..i.min(r.ncols()) {
                assert!(
                    r[(i, j)].abs() < 1e-12,
                    "R not upper triangular at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn wide_matrix() {
        let a = DMat::from_rows(&[&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]]);
        let (q, r) = qr(&a);
        assert!(mat_close(&(&q * &r), &a));
    }
}
