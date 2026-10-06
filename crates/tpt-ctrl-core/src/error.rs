//! Error types shared by the numeric kernels.

use core::fmt;

/// Errors returned by linear-algebra and solver routines.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LinalgError {
    /// A factorization met a (numerically) zero pivot: the matrix is singular.
    SingularMatrix,
    /// A matrix expected to be symmetric positive definite is not.
    NotPositiveDefinite,
    /// An iterative solver (e.g. the Francis QR iteration) failed to converge
    /// within its iteration budget.
    NoConvergence,
    /// Operand dimensions do not match the operation.
    DimensionMismatch {
        /// Expected shape as `(rows, cols)`.
        expected: (usize, usize),
        /// Found shape as `(rows, cols)`.
        found: (usize, usize),
    },
    /// A square matrix was required.
    NotSquare {
        /// Actual `(rows, cols)`.
        shape: (usize, usize),
    },
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinalgError::SingularMatrix => write!(f, "matrix is singular to working precision"),
            LinalgError::NotPositiveDefinite => {
                write!(f, "matrix is not symmetric positive definite")
            }
            LinalgError::NoConvergence => write!(f, "iterative solver failed to converge"),
            LinalgError::DimensionMismatch { expected, found } => write!(
                f,
                "dimension mismatch: expected {}x{}, found {}x{}",
                expected.0, expected.1, found.0, found.1
            ),
            LinalgError::NotSquare { shape } => {
                write!(f, "square matrix required, got {}x{}", shape.0, shape.1)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LinalgError {}

/// Convenient alias for results produced by the numeric kernels.
pub type Result<T, E = LinalgError> = core::result::Result<T, E>;
