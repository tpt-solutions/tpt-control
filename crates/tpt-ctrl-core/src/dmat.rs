//! Dynamically-sized, heap-backed dense matrix with row-major storage.
//!
//! `DMat` is the numeric workhorse of the workspace: the decompositions (LU,
//! Cholesky, QR, Schur), the matrix exponential, and all solver crates
//! operate on it. Dimensions are runtime-checked (panics on mismatches);
//! compile-time dimensional safety is provided by [`crate::smat::Mat`],
//! which delegates its kernels to this type.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

use crate::error::{LinalgError, Result};
use crate::field::Field;

/// A dynamically-sized dense matrix of `T`, stored row-major in a `Vec`
/// (`(i, j)` lives at `i * ncols + j`).
#[derive(Clone, PartialEq)]
pub struct DMat<T = f64> {
    nrows: usize,
    ncols: usize,
    data: Vec<T>,
}

impl<T> DMat<T> {
    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Total element count (`nrows * ncols`).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// `true` when the matrix has no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// `true` when `nrows == ncols`.
    pub fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

    /// Borrows the backing storage in row-major order.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Element access with bounds checking, returning `None` when out of range.
    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        if i < self.nrows && j < self.ncols {
            Some(&self.data[i * self.ncols + j])
        } else {
            None
        }
    }

    /// Mutable element access with bounds checking.
    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        if i < self.nrows && j < self.ncols {
            Some(&mut self.data[i * self.ncols + j])
        } else {
            None
        }
    }

    /// Overwrites row `i` from a slice.
    ///
    /// # Panics
    /// Panics when `values.len() != ncols` or `i >= nrows`.
    #[allow(clippy::needless_range_loop)]
    pub fn set_row(&mut self, i: usize, values: &[T])
    where
        T: Clone,
    {
        assert_eq!(values.len(), self.ncols, "set_row: length mismatch");
        for j in 0..self.ncols {
            self.data[i * self.ncols + j] = values[j].clone();
        }
    }

    /// Copies a column out as a `Vec`.
    pub fn column(&self, j: usize) -> Vec<T>
    where
        T: Clone,
    {
        assert!(j < self.ncols, "column index out of bounds");
        (0..self.nrows)
            .map(|i| self.data[i * self.ncols + j].clone())
            .collect()
    }

    /// Copies a row out as a `Vec`.
    pub fn row(&self, i: usize) -> Vec<T>
    where
        T: Clone,
    {
        assert!(i < self.nrows, "row index out of bounds");
        self.data[i * self.ncols..(i + 1) * self.ncols].to_vec()
    }
}

impl<T: Field> DMat<T> {
    /// A zero matrix of the given shape.
    pub fn zeros(nrows: usize, ncols: usize) -> DMat<T> {
        DMat {
            nrows,
            ncols,
            data: vec![T::zero(); nrows * ncols],
        }
    }

    /// The `n x n` identity matrix.
    pub fn identity(n: usize) -> DMat<T> {
        let mut m = DMat::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = T::one();
        }
        m
    }

    /// A square diagonal matrix with the given diagonal entries.
    pub fn diagonal(diag: &[T]) -> DMat<T> {
        let n = diag.len();
        let mut m = DMat::zeros(n, n);
        for (i, v) in diag.iter().enumerate() {
            m.data[i * n + i] = *v;
        }
        m
    }

    /// Builds a matrix from row slices, e.g. `DMat::from_rows(&[[1., 0.], [0., 1.]])`.
    ///
    /// # Panics
    /// Panics when the rows have differing lengths.
    pub fn from_rows(rows: &[&[T]]) -> DMat<T> {
        let nrows = rows.len();
        let ncols = if nrows == 0 { 0 } else { rows[0].len() };
        let mut m = DMat::zeros(nrows, ncols);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), ncols, "from_rows: ragged input");
            m.set_row(i, row);
        }
        m
    }

    /// Builds a matrix by evaluating `f(i, j)` for every entry.
    pub fn from_fn(nrows: usize, ncols: usize, mut f: impl FnMut(usize, usize) -> T) -> DMat<T> {
        let mut m = DMat::zeros(nrows, ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                m.data[i * ncols + j] = f(i, j);
            }
        }
        m
    }

    /// Transpose.
    #[allow(clippy::needless_range_loop)]
    pub fn transpose(&self) -> DMat<T> {
        let mut out = DMat::zeros(self.ncols, self.nrows);
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                out.data[j * self.nrows + i] = self.data[i * self.ncols + j];
            }
        }
        out
    }

    /// Sum of the diagonal entries.
    ///
    /// # Panics
    /// Panics when the matrix is not square.
    pub fn trace(&self) -> T {
        assert!(self.is_square(), "trace requires a square matrix");
        let mut acc = T::zero();
        for i in 0..self.nrows {
            acc += self.data[i * self.ncols + i];
        }
        acc
    }

    /// Maximum absolute column sum (the matrix 1-norm). Propagates NaN.
    pub fn norm_1(&self) -> T {
        let mut best = T::zero();
        for j in 0..self.ncols {
            let mut col = T::zero();
            for i in 0..self.nrows {
                col += self.data[i * self.ncols + j].abs();
            }
            // `!(col <= best)` (not `col > best`) so NaN replaces the max.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(col <= best) {
                best = col;
            }
        }
        best
    }

    /// Maximum absolute row sum (the matrix infinity-norm). Propagates NaN.
    pub fn norm_inf(&self) -> T {
        let mut best = T::zero();
        for i in 0..self.nrows {
            let mut row = T::zero();
            for j in 0..self.ncols {
                row += self.data[i * self.ncols + j].abs();
            }
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(row <= best) {
                best = row;
            }
        }
        best
    }

    /// Frobenius norm.
    pub fn norm_fro(&self) -> T {
        let mut acc = T::zero();
        for v in &self.data {
            acc += *v * *v;
        }
        acc.sqrt()
    }

    /// Extracts the sub-matrix at rows `r0..r0+nrows`, columns `c0..c0+ncols`.
    ///
    /// # Panics
    /// Panics when the requested window exceeds the matrix bounds.
    pub fn submatrix(&self, r0: usize, c0: usize, nrows: usize, ncols: usize) -> DMat<T> {
        assert!(r0 + nrows <= self.nrows && c0 + ncols <= self.ncols);
        DMat::from_fn(nrows, ncols, |i, j| {
            self.data[(r0 + i) * self.ncols + c0 + j]
        })
    }

    /// Horizontal concatenation `[self | right]`.
    ///
    /// # Panics
    /// Panics when the row counts differ.
    pub fn augment(&self, right: &DMat<T>) -> DMat<T> {
        assert_eq!(self.nrows, right.nrows, "augment: row count mismatch");
        DMat::from_fn(self.nrows, self.ncols + right.ncols, |i, j| {
            if j < self.ncols {
                self.data[i * self.ncols + j]
            } else {
                right.data[i * right.ncols + (j - self.ncols)]
            }
        })
    }

    /// Vertical concatenation `[self; bottom]`.
    ///
    /// # Panics
    /// Panics when the column counts differ.
    pub fn stack(&self, bottom: &DMat<T>) -> DMat<T> {
        assert_eq!(self.ncols, bottom.ncols, "stack: column count mismatch");
        DMat::from_fn(self.nrows + bottom.nrows, self.ncols, |i, j| {
            if i < self.nrows {
                self.data[i * self.ncols + j]
            } else {
                bottom.data[(i - self.nrows) * self.ncols + j]
            }
        })
    }

    /// Builds the block matrix `[[a, b], [c, d]]`.
    ///
    /// # Panics
    /// Panics on incompatible block shapes.
    pub fn block2(a: &DMat<T>, b: &DMat<T>, c: &DMat<T>, d: &DMat<T>) -> DMat<T> {
        assert_eq!(a.nrows, b.nrows, "block2: top row mismatch");
        assert_eq!(c.nrows, d.nrows, "block2: bottom row mismatch");
        assert_eq!(a.ncols, c.ncols, "block2: left column mismatch");
        assert_eq!(b.ncols, d.ncols, "block2: right column mismatch");
        a.augment(b).stack(&c.augment(d))
    }
}

impl<T: Clone> DMat<T> {
    /// Builds a matrix from a flat row-major slice.
    ///
    /// # Panics
    /// Panics when `data.len() != nrows * ncols`.
    pub fn from_row_slice(nrows: usize, ncols: usize, data: &[T]) -> DMat<T> {
        assert_eq!(data.len(), nrows * ncols, "from_row_slice: length mismatch");
        DMat {
            nrows,
            ncols,
            data: data.to_vec(),
        }
    }
}

impl<T> Index<(usize, usize)> for DMat<T> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.data[i * self.ncols + j]
    }
}

impl<T> IndexMut<(usize, usize)> for DMat<T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        &mut self.data[i * self.ncols + j]
    }
}

impl<T: Field> Add for &DMat<T> {
    type Output = DMat<T>;
    /// # Panics
    /// Panics when the shapes differ.
    fn add(self, rhs: &DMat<T>) -> DMat<T> {
        assert_eq!(
            (self.nrows, self.ncols),
            (rhs.nrows, rhs.ncols),
            "matrix add: shape mismatch"
        );
        let mut out = DMat::zeros(self.nrows, self.ncols);
        for k in 0..out.data.len() {
            out.data[k] = self.data[k] + rhs.data[k];
        }
        out
    }
}

impl<T: Field> Sub for &DMat<T> {
    type Output = DMat<T>;
    /// # Panics
    /// Panics when the shapes differ.
    fn sub(self, rhs: &DMat<T>) -> DMat<T> {
        assert_eq!(
            (self.nrows, self.ncols),
            (rhs.nrows, rhs.ncols),
            "matrix sub: shape mismatch"
        );
        let mut out = DMat::zeros(self.nrows, self.ncols);
        for k in 0..out.data.len() {
            out.data[k] = self.data[k] - rhs.data[k];
        }
        out
    }
}

impl<T: Field> Neg for &DMat<T> {
    type Output = DMat<T>;
    fn neg(self) -> DMat<T> {
        let mut out = DMat::zeros(self.nrows, self.ncols);
        for k in 0..out.data.len() {
            out.data[k] = -self.data[k];
        }
        out
    }
}

// By-value / mixed-borrow forwarding impls so expression chains can combine
// owned and borrowed operands freely.
impl<T: Field> Add for DMat<T> {
    type Output = DMat<T>;
    fn add(self, rhs: DMat<T>) -> DMat<T> {
        &self + &rhs
    }
}

impl<T: Field> Add<&DMat<T>> for DMat<T> {
    type Output = DMat<T>;
    fn add(self, rhs: &DMat<T>) -> DMat<T> {
        &self + rhs
    }
}

impl<T: Field> Add<DMat<T>> for &DMat<T> {
    type Output = DMat<T>;
    fn add(self, rhs: DMat<T>) -> DMat<T> {
        self + &rhs
    }
}

impl<T: Field> Sub for DMat<T> {
    type Output = DMat<T>;
    fn sub(self, rhs: DMat<T>) -> DMat<T> {
        &self - &rhs
    }
}

impl<T: Field> Sub<&DMat<T>> for DMat<T> {
    type Output = DMat<T>;
    fn sub(self, rhs: &DMat<T>) -> DMat<T> {
        &self - rhs
    }
}

impl<T: Field> Sub<DMat<T>> for &DMat<T> {
    type Output = DMat<T>;
    fn sub(self, rhs: DMat<T>) -> DMat<T> {
        self - &rhs
    }
}

impl<T: Field> Mul for DMat<T> {
    type Output = DMat<T>;
    fn mul(self, rhs: DMat<T>) -> DMat<T> {
        &self * &rhs
    }
}

impl<T: Field> Mul<&DMat<T>> for DMat<T> {
    type Output = DMat<T>;
    fn mul(self, rhs: &DMat<T>) -> DMat<T> {
        &self * rhs
    }
}

impl<T: Field> Mul<DMat<T>> for &DMat<T> {
    type Output = DMat<T>;
    fn mul(self, rhs: DMat<T>) -> DMat<T> {
        self * &rhs
    }
}

/// Scalar product `self * k` (owned).
impl<T: Field> Mul<T> for DMat<T> {
    type Output = DMat<T>;
    fn mul(self, k: T) -> DMat<T> {
        let mut out = self;
        for v in &mut out.data {
            *v *= k;
        }
        out
    }
}

/// Quotient `self / k` (borrowed).
impl<T: Field> Div<T> for &DMat<T> {
    type Output = DMat<T>;
    fn div(self, k: T) -> DMat<T> {
        let mut out = DMat::zeros(self.nrows, self.ncols);
        for l in 0..out.data.len() {
            out.data[l] = self.data[l] / k;
        }
        out
    }
}

/// Quotient `self / k` (owned).
impl<T: Field> Div<T> for DMat<T> {
    type Output = DMat<T>;
    fn div(self, k: T) -> DMat<T> {
        let mut out = self;
        for v in &mut out.data {
            *v /= k;
        }
        out
    }
}

/// Matrix product `self * rhs`.
///
/// # Panics
/// Panics when `self.ncols != rhs.nrows`.
impl<T: Field> Mul for &DMat<T> {
    type Output = DMat<T>;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: &DMat<T>) -> DMat<T> {
        assert_eq!(
            self.ncols, rhs.nrows,
            "matrix mul: inner dimension mismatch"
        );
        let mut out = DMat::zeros(self.nrows, rhs.ncols);
        for i in 0..self.nrows {
            for k in 0..self.ncols {
                let a_ik = self.data[i * self.ncols + k];
                if a_ik == T::zero() {
                    continue;
                }
                for j in 0..rhs.ncols {
                    out.data[i * rhs.ncols + j] += a_ik * rhs.data[k * rhs.ncols + j];
                }
            }
        }
        out
    }
}

/// Scalar product `self * k`.
impl<T: Field> Mul<T> for &DMat<T> {
    type Output = DMat<T>;
    fn mul(self, k: T) -> DMat<T> {
        let mut out = DMat::zeros(self.nrows, self.ncols);
        for l in 0..out.data.len() {
            out.data[l] = self.data[l] * k;
        }
        out
    }
}

impl<T: fmt::Debug> fmt::Debug for DMat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DMat[{}x{}](", self.nrows, self.ncols)?;
        for i in 0..self.nrows {
            if i > 0 {
                write!(f, "; ")?;
            }
            for j in 0..self.ncols {
                if j > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{:?}", self.data[i * self.ncols + j])?;
            }
        }
        write!(f, ")")
    }
}

impl<T: Field + fmt::Display> fmt::Display for DMat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                if j > 0 {
                    s.push(' ');
                }
                s.push_str(&format!("{}", self.data[i * self.ncols + j]));
            }
            if i + 1 < self.nrows {
                s.push('\n');
            }
        }
        f.write_str(&s)
    }
}

/// Helpers shared by the factorization modules.
pub(crate) fn require_square<T>(a: &DMat<T>) -> Result<()> {
    if a.is_square() {
        Ok(())
    } else {
        Err(LinalgError::NotSquare {
            shape: (a.nrows(), a.ncols()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn transpose_twice_is_identity() {
        let a = DMat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        assert_eq!(a.transpose().transpose(), a);
    }

    #[test]
    fn matmul_small() {
        let a = DMat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b = DMat::from_rows(&[&[0.0, 1.0], &[1.0, 0.0]]);
        let c = &a * &b;
        assert!(close(c[(0, 0)], 2.0));
        assert!(close(c[(0, 1)], 1.0));
        assert!(close(c[(1, 0)], 4.0));
        assert!(close(c[(1, 1)], 3.0));
    }

    #[test]
    fn block2_roundtrip() {
        let a = DMat::from_rows(&[&[1.0]]);
        let b = DMat::from_rows(&[&[2.0, 3.0]]);
        let c = DMat::from_rows(&[&[4.0], &[5.0]]);
        let d = DMat::from_rows(&[&[6.0, 7.0], &[8.0, 9.0]]);
        let m = DMat::block2(&a, &b, &c, &d);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
        assert!(close(m[(0, 0)], 1.0));
        assert!(close(m[(0, 2)], 3.0));
        assert!(close(m[(2, 2)], 9.0));
    }

    #[test]
    fn norms() {
        let a = DMat::from_rows(&[&[1.0, -2.0], &[-4.0, 3.0]]);
        // Column sums: |1| + |-4| = 5, |-2| + |3| = 5.
        assert!(close(a.norm_1(), 5.0));
        // Row sums: |1| + |-2| = 3, |-4| + |3| = 7.
        assert!(close(a.norm_inf(), 7.0));
        assert!(close(a.trace(), 4.0));
    }

    #[test]
    fn submatrix_extraction() {
        let a = DMat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);
        let s = a.submatrix(1, 1, 2, 2);
        assert!(close(s[(0, 0)], 5.0) && close(s[(1, 1)], 9.0));
    }
}
