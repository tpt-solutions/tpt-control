//! Compile-time dimensioned matrix wrapper.
//!
//! [`Mat`] carries its shape in const-generic parameters, so dimensional
//! mismatches (e.g. multiplying a 3x1 state vector by a 2x2 gain) are
//! compile errors. Storage is a heap `Vec` of length `R * C`, which keeps
//! the type `no_std + alloc` friendly and the kernels simple: `Mat`
//! delegates the numeric work to [`crate::dmat::DMat`].

use alloc::vec::Vec;
use core::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use crate::dmat::DMat;
use crate::error::{LinalgError, Result};
use crate::field::Field;

/// A fixed-shape dense matrix with `R` rows and `C` columns.
///
/// # Example
/// ```
/// use tpt_ctrl_core::smat::Mat;
///
/// let a = Mat::<f64, 2, 2>::from_rows_arr(&[[1.0, 2.0], [3.0, 4.0]]);
/// let x = Mat::<f64, 2, 1>::from_row_slice(&[5.0, 6.0]);
/// let y = a * x; // 2x1: compile-checked inner dimension
/// assert_eq!(y[(0, 0)], 17.0);
/// assert_eq!(y[(1, 0)], 39.0);
/// ```
#[derive(Clone, PartialEq)]
pub struct Mat<T, const R: usize, const C: usize> {
    data: Vec<T>,
}

/// A column vector with `N` entries.
pub type Vector<T, const N: usize> = Mat<T, N, 1>;

impl<T: Field, const R: usize, const C: usize> Mat<T, R, C> {
    /// The zero matrix.
    pub fn zeros() -> Mat<T, R, C> {
        Mat {
            data: alloc::vec![T::zero(); R * C],
        }
    }

    /// Builds from row-major slice; length must be exactly `R * C`.
    ///
    /// # Panics
    /// Panics when `data.len() != R * C`.
    pub fn from_row_slice(data: &[T]) -> Mat<T, R, C> {
        assert_eq!(data.len(), R * C, "Mat::from_row_slice: length mismatch");
        Mat {
            data: data.to_vec(),
        }
    }

    /// Builds from a 2-D row array literal:
    /// `Mat::from_rows_arr(&[[1.0, 0.0], [0.0, 1.0]])`.
    pub fn from_rows_arr(rows: &[[T; C]; R]) -> Mat<T, R, C> {
        let mut data = Vec::with_capacity(R * C);
        for row in rows {
            data.extend_from_slice(row);
        }
        Mat { data }
    }

    /// Builds by evaluating `f(i, j)`.
    pub fn from_fn(mut f: impl FnMut(usize, usize) -> T) -> Mat<T, R, C> {
        let mut data = Vec::with_capacity(R * C);
        for i in 0..R {
            for j in 0..C {
                data.push(f(i, j));
            }
        }
        Mat { data }
    }

    /// Number of rows (compile-time constant).
    pub const fn nrows() -> usize {
        R
    }

    /// Number of columns (compile-time constant).
    pub const fn ncols() -> usize {
        C
    }

    /// Element access with bounds check returning `Option`.
    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        if i < R && j < C {
            Some(&self.data[i * C + j])
        } else {
            None
        }
    }

    /// Copies the dynamic-matrix view of this matrix.
    pub fn to_dynamic(&self) -> DMat<T> {
        DMat::from_row_slice(R, C, &self.data)
    }

    /// Frobenius norm.
    pub fn norm_fro(&self) -> T {
        let mut acc = T::zero();
        for v in &self.data {
            acc += *v * *v;
        }
        acc.sqrt()
    }

    /// Transpose.
    pub fn transpose(&self) -> Mat<T, C, R> {
        Mat::from_fn(|i, j| self.data[j * C + i])
    }
}

impl<T: Field, const N: usize> Mat<T, N, N> {
    /// The `N x N` identity matrix.
    pub fn identity() -> Mat<T, N, N> {
        Mat::from_fn(|i, j| if i == j { T::one() } else { T::zero() })
    }

    /// Solves `self * x = b` when this matrix is square.
    pub fn solve(&self, b: &Mat<T, N, N>) -> Result<Mat<T, N, N>> {
        let x = self.to_dynamic().solve(&b.to_dynamic())?;
        Mat::from_dynamic(&x)
    }
}

impl<T: Clone, const R: usize, const C: usize> Mat<T, R, C> {
    /// Converts from a dynamic matrix, checking dimensions.
    pub fn from_dynamic(d: &DMat<T>) -> Result<Mat<T, R, C>> {
        if d.nrows() != R || d.ncols() != C {
            return Err(LinalgError::DimensionMismatch {
                expected: (R, C),
                found: (d.nrows(), d.ncols()),
            });
        }
        Ok(Mat {
            data: d.as_slice().to_vec(),
        })
    }
}

impl<T, const R: usize, const C: usize> Index<(usize, usize)> for Mat<T, R, C> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.data[i * C + j]
    }
}

impl<T, const R: usize, const C: usize> IndexMut<(usize, usize)> for Mat<T, R, C> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        &mut self.data[i * C + j]
    }
}

impl<T: Field, const A: usize, const B: usize, const D: usize> Mul<&Mat<T, B, D>>
    for &Mat<T, A, B>
{
    type Output = Mat<T, A, D>;
    fn mul(self, rhs: &Mat<T, B, D>) -> Mat<T, A, D> {
        Mat::from_fn(|i, j| {
            let mut acc = T::zero();
            for k in 0..B {
                acc += self[(i, k)] * rhs[(k, j)];
            }
            acc
        })
    }
}

impl<T: Field, const A: usize, const B: usize, const D: usize> Mul<Mat<T, B, D>> for Mat<T, A, B> {
    type Output = Mat<T, A, D>;
    fn mul(self, rhs: Mat<T, B, D>) -> Mat<T, A, D> {
        &self * &rhs
    }
}

impl<T: Field, const R: usize, const C: usize> Add for Mat<T, R, C> {
    type Output = Mat<T, R, C>;
    fn add(self, rhs: Mat<T, R, C>) -> Mat<T, R, C> {
        let lhs = &self;
        let rhs = &rhs;
        Mat::from_fn(|i, j| lhs[(i, j)] + rhs[(i, j)])
    }
}

impl<T: Field, const R: usize, const C: usize> Sub for Mat<T, R, C> {
    type Output = Mat<T, R, C>;
    fn sub(self, rhs: Mat<T, R, C>) -> Mat<T, R, C> {
        let lhs = &self;
        let rhs = &rhs;
        Mat::from_fn(|i, j| lhs[(i, j)] - rhs[(i, j)])
    }
}

impl<T: Field, const R: usize, const C: usize> Neg for Mat<T, R, C> {
    type Output = Mat<T, R, C>;
    fn neg(self) -> Mat<T, R, C> {
        let lhs = &self;
        Mat::from_fn(|i, j| -lhs[(i, j)])
    }
}

impl<T: Field, const R: usize, const C: usize> Mul<T> for Mat<T, R, C> {
    type Output = Mat<T, R, C>;
    fn mul(self, k: T) -> Mat<T, R, C> {
        let lhs = &self;
        Mat::from_fn(|i, j| lhs[(i, j)] * k)
    }
}

impl<T: core::fmt::Debug, const R: usize, const C: usize> core::fmt::Debug for Mat<T, R, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Mat[{R}x{C}](")?;
        for i in 0..R {
            if i > 0 {
                write!(f, "; ")?;
            }
            for j in 0..C {
                if j > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{:?}", self.data[i * C + j])?;
            }
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_flow_through_mul() {
        let a = Mat::<f64, 2, 3>::from_rows_arr(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let b = Mat::<f64, 3, 2>::from_rows_arr(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
        let c = a * b;
        assert_eq!(c[(0, 0)], 4.0);
        assert_eq!(c[(1, 1)], 11.0);
    }

    #[test]
    fn transpose_swaps_shape() {
        let a = Mat::<f64, 2, 3>::zeros();
        let t: Mat<f64, 3, 2> = a.transpose();
        let _ = t;
    }

    #[test]
    fn dynamic_roundtrip() {
        let a = Mat::<f64, 2, 2>::identity();
        let d = a.to_dynamic();
        let back = Mat::<f64, 2, 2>::from_dynamic(&d).unwrap();
        assert_eq!(a, back);
        assert!(Mat::<f64, 3, 2>::from_dynamic(&d).is_err());
    }

    #[test]
    fn identity_and_solve() {
        let a =
            Mat::<f64, 3, 3>::from_rows_arr(&[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 8.0]]);
        let eye = Mat::<f64, 3, 3>::identity();
        let x = a.solve(&eye).unwrap();
        assert!((x[(2, 2)] - 0.125).abs() < 1e-12);
    }
}
