//! Python bindings for tpt-control.
//!
//! Matrices cross the boundary as lists of rows (`[[...], [...]]`), so the
//! package needs no NumPy. Results are plain tuples and one `Certificate`
//! class. Dimension and solver errors raise `ValueError`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use tpt_ctrl_classic::tf::TransferFunction;
use tpt_ctrl_core::analysis::poles;
use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_core::discretize::zoh;
use tpt_ctrl_optimal::certify::{certify_continuous, certify_discrete};
use tpt_ctrl_optimal::{dlqr, lqr};

type Rows = Vec<Vec<f64>>;

fn to_dmat(rows: &Rows, name: &str) -> PyResult<DMat<f64>> {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    if nrows == 0 || ncols == 0 {
        return Err(PyValueError::new_err(format!("`{name}` is empty")));
    }
    if rows.iter().any(|r| r.len() != ncols) {
        return Err(PyValueError::new_err(format!("`{name}` is not rectangular")));
    }
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    Ok(DMat::from_row_slice(nrows, ncols, &flat))
}

fn to_rows(m: &DMat<f64>) -> Rows {
    (0..m.nrows()).map(|i| m.row(i)).collect()
}

fn solver_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Verification report for a Riccati solution.
#[pyclass(module = "tpt_control")]
#[derive(Clone, Debug)]
pub struct Certificate {
    /// Relative Frobenius residual of the Riccati equation.
    #[pyo3(get)]
    pub riccati_residual: f64,
    /// Closed-loop margin: `-max Re(λ)` (continuous) or `1 - max |λ|` (discrete).
    #[pyo3(get)]
    pub closed_loop_margin: f64,
    /// `True` when the closed loop is strictly stable.
    #[pyo3(get)]
    pub stable: bool,
}

#[pymethods]
impl Certificate {
    /// `True` when the residual is at most `tol` and the loop is stable.
    #[pyo3(signature = (tol = 1e-8))]
    fn is_certified(&self, tol: f64) -> bool {
        self.stable && self.riccati_residual <= tol
    }

    fn __repr__(&self) -> String {
        format!(
            "Certificate(riccati_residual={:.3e}, closed_loop_margin={:.6}, stable={})",
            self.riccati_residual, self.closed_loop_margin, self.stable
        )
    }
}

impl From<tpt_ctrl_optimal::Certificate> for Certificate {
    fn from(c: tpt_ctrl_optimal::Certificate) -> Self {
        Certificate {
            riccati_residual: c.riccati_residual,
            closed_loop_margin: c.closed_loop_margin,
            stable: c.stable,
        }
    }
}

/// Continuous-time LQR. Returns `(K, X)` with `u = -K x`.
#[pyfunction]
fn lqr_gain(a: Rows, b: Rows, q: Rows, r: Rows) -> PyResult<(Rows, Rows)> {
    let sol = lqr(&to_dmat(&a, "a")?, &to_dmat(&b, "b")?, &to_dmat(&q, "q")?, &to_dmat(&r, "r")?)
        .map_err(solver_err)?;
    Ok((to_rows(&sol.k), to_rows(&sol.x)))
}

/// Discrete-time LQR. Returns `(K, X)` with `u = -K x`.
#[pyfunction]
fn dlqr_gain(a: Rows, b: Rows, q: Rows, r: Rows) -> PyResult<(Rows, Rows)> {
    let sol = dlqr(&to_dmat(&a, "a")?, &to_dmat(&b, "b")?, &to_dmat(&q, "q")?, &to_dmat(&r, "r")?)
        .map_err(solver_err)?;
    Ok((to_rows(&sol.k), to_rows(&sol.x)))
}

/// Certifies a `(X, K)` solution. `continuous=True` checks the CARE, otherwise the DARE.
#[pyfunction]
#[pyo3(signature = (a, b, q, r, x, k, continuous = true))]
fn certify(
    a: Rows,
    b: Rows,
    q: Rows,
    r: Rows,
    x: Rows,
    k: Rows,
    continuous: bool,
) -> PyResult<Certificate> {
    let (a, b, q, r) = (to_dmat(&a, "a")?, to_dmat(&b, "b")?, to_dmat(&q, "q")?, to_dmat(&r, "r")?);
    let (x, k) = (to_dmat(&x, "x")?, to_dmat(&k, "k")?);
    let cert = if continuous {
        certify_continuous(&a, &b, &q, &r, &x, &k)
    } else {
        certify_discrete(&a, &b, &q, &r, &x, &k)
    }
    .map_err(solver_err)?;
    Ok(cert.into())
}

/// Exact zero-order-hold discretization. Returns `(Ad, Bd)`.
#[pyfunction]
fn c2d(a: Rows, b: Rows, ts: f64) -> PyResult<(Rows, Rows)> {
    let (ad, bd) = zoh(&to_dmat(&a, "a")?, &to_dmat(&b, "b")?, ts).map_err(solver_err)?;
    Ok((to_rows(&ad), to_rows(&bd)))
}

/// SISO transfer function to controllable-canonical state space. Returns `(A, B, C, D)`.
#[pyfunction]
fn tf2ss(num: Vec<f64>, den: Vec<f64>) -> PyResult<(Rows, Rows, Rows, Rows)> {
    let tf = TransferFunction::new(&num, &den).map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
    let (a, b, c, d) = tf.to_state_space();
    Ok((to_rows(&a), to_rows(&b), to_rows(&c), to_rows(&d)))
}

/// Eigenvalues of a square matrix as `(real, imag)` pairs.
#[pyfunction]
fn eigenvalues(a: Rows) -> PyResult<Vec<(f64, f64)>> {
    let eig = poles(&to_dmat(&a, "a")?).map_err(solver_err)?;
    Ok(eig.into_iter().map(|p| (p.re, p.im)).collect())
}

#[pymodule]
fn tpt_control(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Certificate>()?;
    m.add_function(wrap_pyfunction!(lqr_gain, m)?)?;
    m.add_function(wrap_pyfunction!(dlqr_gain, m)?)?;
    m.add_function(wrap_pyfunction!(certify, m)?)?;
    m.add_function(wrap_pyfunction!(c2d, m)?)?;
    m.add_function(wrap_pyfunction!(tf2ss, m)?)?;
    m.add_function(wrap_pyfunction!(eigenvalues, m)?)?;
    Ok(())
}
