//! LU decomposition with partial pivoting.
//!
//! The single dense solver behind `solve`, `inverse`, and `det`; also the
//! linear solve used internally by the Padé matrix exponential, the
//! Bartels-Stewart Lyapunov solve, the Riccati iterations, and the MPC QP.

use alloc::vec::Vec;

use crate::dmat::{require_square, DMat};
use crate::error::{LinalgError, Result};
use crate::field::Field;

/// An `L U P` factorization of a square matrix.
///
/// Produced by [`DMat`]'s `lu` method (see [`DMat::lu`]); consumes one `O(n^3)`
/// pass and amortizes over many right-hand sides.
pub struct Lu<T> {
    lu: DMat<T>,
    /// `perm[i]` is the original row index that now sits at row `i`.
    perm: Vec<usize>,
    singular: bool,
    negated: bool,
}

impl<T: Field + core::fmt::Debug> core::fmt::Debug for Lu<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Lu")
            .field("lu", &self.lu)
            .field("perm", &self.perm)
            .field("singular", &self.singular)
            .field("negated", &self.negated)
            .finish()
    }
}

impl<T: Field> Lu<T> {
    /// Factors `a` with row-swap partial pivoting.
    ///
    /// Near-zero pivots are recorded (the factorization reports
    /// [`LinalgError::SingularMatrix`] on use) rather than aborting, so the
    /// caller sees a single consistent error type.
    pub fn new(a: &DMat<T>) -> Lu<T> {
        let n = a.nrows();
        debug_assert!(a.is_square(), "lu requires a square matrix");
        let mut lu = a.clone();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut singular = false;
        let mut negated = false;

        for k in 0..n {
            // Pivot: largest magnitude in column k at or below the diagonal.
            let mut p = k;
            let mut best = lu[(k, k)].abs();
            for i in k + 1..n {
                let v = lu[(i, k)].abs();
                if v > best {
                    best = v;
                    p = i;
                }
            }
            if best == T::zero() {
                singular = true;
                continue;
            }
            if p != k {
                for j in 0..n {
                    let tmp = lu[(k, j)];
                    lu[(k, j)] = lu[(p, j)];
                    lu[(p, j)] = tmp;
                }
                perm.swap(k, p);
                negated = !negated;
            }
            let pivot = lu[(k, k)];
            for i in k + 1..n {
                let f = lu[(i, k)] / pivot;
                lu[(i, k)] = f;
                if f != T::zero() {
                    for j in k + 1..n {
                        let ukj = lu[(k, j)];
                        lu[(i, j)] -= f * ukj;
                    }
                }
            }
        }

        Lu {
            lu,
            perm,
            singular,
            negated,
        }
    }

    /// `true` when a zero pivot was encountered.
    pub fn is_singular(&self) -> bool {
        self.singular
    }

    /// Solves `A X = B` for a matrix right-hand side (`B` is `n x m`).
    pub fn solve(&self, b: &DMat<T>) -> Result<DMat<T>> {
        if self.singular {
            return Err(LinalgError::SingularMatrix);
        }
        if b.nrows() != self.lu.nrows() {
            return Err(LinalgError::DimensionMismatch {
                expected: (self.lu.nrows(), self.lu.ncols()),
                found: (b.nrows(), b.ncols()),
            });
        }
        let n = self.lu.nrows();
        let mut x = b.clone();
        // Forward substitution with the permutation applied on the fly, then
        // back substitution. x currently holds P B row-wise: row i of the
        // working copy must be original row perm[i].
        let mut pb = DMat::<T>::zeros(n, b.ncols());
        for i in 0..n {
            for j in 0..b.ncols() {
                pb[(i, j)] = b[(self.perm[i], j)];
            }
        }
        for i in 0..n {
            for j in 0..b.ncols() {
                let mut acc = pb[(i, j)];
                for k in 0..i {
                    acc -= self.lu[(i, k)] * x[(k, j)];
                }
                x[(i, j)] = acc;
            }
        }
        for i in (0..n).rev() {
            for j in 0..b.ncols() {
                let mut acc = x[(i, j)];
                for k in i + 1..n {
                    acc -= self.lu[(i, k)] * x[(k, j)];
                }
                x[(i, j)] = acc / self.lu[(i, i)];
            }
        }
        Ok(x)
    }

    /// Solves `A x = b` for a single right-hand side.
    pub fn solve_vec(&self, b: &[T]) -> Result<Vec<T>> {
        let n = self.lu.nrows();
        let col = DMat::from_fn(n, 1, |i, _| b[i]);
        let x = self.solve(&col)?;
        Ok(x.column(0))
    }

    /// Computes `A^{-1}`.
    pub fn inverse(&self) -> Result<DMat<T>> {
        self.solve(&DMat::identity(self.lu.nrows()))
    }

    /// Determinant via the product of `U`'s diagonal, signed by the swap count.
    pub fn det(&self) -> Result<T> {
        if self.singular {
            return Err(LinalgError::SingularMatrix);
        }
        let n = self.lu.nrows();
        let mut acc = if self.negated { -T::one() } else { T::one() };
        for i in 0..n {
            acc *= self.lu[(i, i)];
        }
        Ok(acc)
    }
}

impl<T: Field> DMat<T> {
    /// Factors this matrix into `P A = L U` (see [`Lu`]).
    pub fn lu(&self) -> Lu<T> {
        Lu::new(self)
    }

    /// Solves `A X = B` via LU with partial pivoting.
    pub fn solve(&self, b: &DMat<T>) -> Result<DMat<T>> {
        require_square(self)?;
        self.lu().solve(b)
    }

    /// Computes the matrix inverse via LU.
    pub fn inverse(&self) -> Result<DMat<T>> {
        require_square(self)?;
        self.lu().inverse()
    }

    /// Computes the determinant via LU.
    pub fn det(&self) -> Result<T> {
        require_square(self)?;
        self.lu().det()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    fn mat_close(a: &DMat<f64>, b: &DMat<f64>) -> bool {
        a.nrows() == b.nrows()
            && a.ncols() == b.ncols()
            && a.as_slice()
                .iter()
                .zip(b.as_slice())
                .all(|(x, y)| close(*x, *y))
    }

    #[test]
    fn solve_known_system() {
        let a = DMat::from_rows(&[&[2.0, 1.0], &[1.0, 3.0]]);
        let b = DMat::from_rows(&[&[3.0], &[5.0]]);
        let x = a.solve(&b).unwrap();
        // x = [4/5, 7/5]
        assert!(close(x[(0, 0)], 0.8));
        assert!(close(x[(1, 0)], 1.4));
    }

    #[test]
    fn solve_needs_pivoting() {
        let a = DMat::from_rows(&[&[0.0, 1.0], &[1.0, 0.0]]);
        let b = DMat::from_rows(&[&[2.0], &[3.0]]);
        let x = a.solve(&b).unwrap();
        assert!(close(x[(0, 0)], 3.0));
        assert!(close(x[(1, 0)], 2.0));
    }

    #[test]
    fn inverse_and_det() {
        let a = DMat::from_rows(&[&[4.0, 7.0], &[2.0, 6.0]]);
        let inv = a.inverse().unwrap();
        assert!(mat_close(&(&a * &inv), &DMat::identity(2)));
        assert!(close(a.det().unwrap(), 10.0));
    }

    #[test]
    fn singular_matrix_reports_error() {
        let a = DMat::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);
        assert_eq!(
            a.solve(&DMat::identity(2)),
            Err(LinalgError::SingularMatrix)
        );
        assert!(a.lu().is_singular());
    }

    #[test]
    fn det_tracks_permutation_sign() {
        // Permutation matrix with odd number of swaps: det = -1.
        let p = DMat::from_rows(&[&[0.0, 1.0], &[1.0, 0.0]]);
        assert!(close(p.det().unwrap(), -1.0));
    }

    #[test]
    fn solve_multi_rhs() {
        let a = DMat::from_rows(&[&[3.0, 1.0], &[1.0, 2.0]]);
        let b = DMat::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let x = a.solve(&b).unwrap();
        assert!(mat_close(&(&a * &x), &b));
    }
}
