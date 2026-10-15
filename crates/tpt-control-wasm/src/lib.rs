//! WebAssembly entry points for the tpt-control playground.
//!
//! The browser passes flat, row-major matrices and gets back a JSON object:
//! `{"k":[...],"riccati_residual":..,"closed_loop_margin":..,"certified":..}`.
//! Exposing one function keeps the JS boundary simple: no matrix classes cross it.

use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_optimal::certify::{certify_continuous, certify_discrete};
use tpt_ctrl_optimal::{dlqr, lqr};
use wasm_bindgen::prelude::*;

/// Residual tolerance used for the `certified` verdict.
pub const RESIDUAL_TOL: f64 = 1e-8;

/// Designs an LQR (`continuous = true`) or DLQR gain and certifies it.
///
/// `a` is `n×n`, `b` is `n×m`, `q` is `n×n`, `r` is `m×m`, all row-major.
#[wasm_bindgen]
pub fn design_lqr(
    continuous: bool,
    n: usize,
    m: usize,
    a: &[f64],
    b: &[f64],
    q: &[f64],
    r: &[f64],
) -> Result<String, JsError> {
    design(continuous, n, m, a, b, q, r).map_err(|e| JsError::new(&e))
}

/// Host-testable core of [`design_lqr`]; errors are plain strings.
pub fn design(
    continuous: bool,
    n: usize,
    m: usize,
    a: &[f64],
    b: &[f64],
    q: &[f64],
    r: &[f64],
) -> Result<String, String> {
    let a = matrix(n, n, a, "A")?;
    let b = matrix(n, m, b, "B")?;
    let q = matrix(n, n, q, "Q")?;
    let r = matrix(m, m, r, "R")?;

    let (k, cert) = if continuous {
        let sol = lqr(&a, &b, &q, &r).map_err(|e| e.to_string())?;
        let cert = certify_continuous(&a, &b, &q, &r, &sol.x, &sol.k).map_err(|e| e.to_string())?;
        (sol.k, cert)
    } else {
        let sol = dlqr(&a, &b, &q, &r).map_err(|e| e.to_string())?;
        let cert = certify_discrete(&a, &b, &q, &r, &sol.x, &sol.k).map_err(|e| e.to_string())?;
        (sol.k, cert)
    };

    let k_json: Vec<String> = k.as_slice().iter().map(|v| format!("{v:?}")).collect();
    Ok(format!(
        "{{\"k\":[{}],\"riccati_residual\":{:?},\"closed_loop_margin\":{:?},\"certified\":{}}}",
        k_json.join(","),
        cert.riccati_residual,
        cert.closed_loop_margin,
        cert.is_certified(RESIDUAL_TOL),
    ))
}

fn matrix(rows: usize, cols: usize, data: &[f64], name: &str) -> Result<DMat<f64>, String> {
    if data.len() != rows * cols {
        return Err(format!(
            "{name} must have {} entries ({rows}x{cols}), got {}",
            rows * cols,
            data.len()
        ));
    }
    Ok(DMat::from_row_slice(rows, cols, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_integrator_design_is_certified() {
        let out = design(
            true,
            2,
            1,
            &[0.0, 1.0, 0.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[1.0],
        )
        .unwrap();
        assert!(out.contains("\"certified\":true"), "{out}");
        assert!(out.contains("\"k\":["));
    }

    #[test]
    fn wrong_sized_input_is_rejected() {
        let err = design(true, 2, 1, &[0.0; 3], &[0.0; 2], &[0.0; 4], &[1.0]).unwrap_err();
        assert!(err.contains("A must have 4 entries"), "{err}");
    }
}
