//! Tolerance-based assertion helpers shared by workspace test suites.

/// Asserts `|a - b| <= tol` (panics with both values on failure).
#[track_caller]
pub fn assert_close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "assert_close failed: {a} vs {b} (tol {tol})"
    );
}

/// Asserts two matrices have matching shapes and elementwise tolerance.
#[track_caller]
pub fn assert_mat_close(a: &tpt_ctrl_core::DMat<f64>, b: &tpt_ctrl_core::DMat<f64>, tol: f64) {
    assert_eq!(
        (a.nrows(), a.ncols()),
        (b.nrows(), b.ncols()),
        "shape mismatch in assert_mat_close"
    );
    for (i, (va, vb)) in a.as_slice().iter().zip(b.as_slice()).enumerate() {
        assert!(
            (va - vb).abs() <= tol,
            "assert_mat_close failed at index {i}: {va} vs {vb} (tol {tol})"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ctrl_core::dmat::DMat;

    #[test]
    #[should_panic(expected = "assert_close failed")]
    fn assert_close_panics_outside_tolerance() {
        assert_close(1.0, 2.0, 0.1);
    }

    #[test]
    fn assert_close_accepts_within_tolerance() {
        assert_close(1.0000001, 1.0, 1e-3);
    }

    #[test]
    fn assert_mat_close_passes_on_equal() {
        let a = DMat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        assert_mat_close(&a, &a, 1e-12);
    }
}
