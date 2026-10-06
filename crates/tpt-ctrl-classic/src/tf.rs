//! SISO transfer functions with rational algebra and companion-matrix
//! poles/zeros.

use std::fmt;
use std::vec::Vec;
use tpt_ctrl_core::c64::C64;
use tpt_ctrl_core::dmat::DMat;

/// A SISO rational transfer function `num(s) / den(s)`, coefficients in
/// **descending** powers of `s`.
///
/// The leading denominator coefficient is normalized to 1; a zero leading
/// coefficient is rejected at construction.
#[derive(Clone, PartialEq)]
pub struct TransferFunction {
    num: Vec<f64>,
    den: Vec<f64>,
}

/// Errors from transfer-function construction and algebra.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TfError {
    /// Denominator was empty or its leading coefficient was zero.
    InvalidDenominator,
    /// Numerator degree exceeded denominator degree (non-proper TF).
    Improper,
}

impl fmt::Display for TfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TfError::InvalidDenominator => {
                write!(f, "denominator is empty or has zero leading coefficient")
            }
            TfError::Improper => write!(f, "numerator degree exceeds denominator degree"),
        }
    }
}

impl std::error::Error for TfError {}

fn trim_leading_zeros(v: &[f64]) -> &[f64] {
    let mut i = 0;
    while i < v.len() - 1 && v[i].abs() < 1e-14 {
        i += 1;
    }
    &v[i..]
}

impl TransferFunction {
    /// Builds `num(s) / den(s)` from descending-power coefficients.
    ///
    /// # Errors
    /// [`TfError::InvalidDenominator`] when `den` is empty or its leading
    /// coefficient is (numerically) zero.
    pub fn new(num: &[f64], den: &[f64]) -> Result<TransferFunction, TfError> {
        if den.is_empty() || den[0].abs() < 1e-14 {
            return Err(TfError::InvalidDenominator);
        }
        // Normalize the leading denominator coefficient to 1, scaling the
        // numerator by the same factor so the function is unchanged.
        let lead = den[0];
        let den: Vec<f64> = den.iter().map(|c| c / lead).collect();
        let num: Vec<f64> = trim_leading_zeros(num).iter().map(|c| c / lead).collect();
        if num.len() > den.len() {
            return Err(TfError::Improper);
        }
        Ok(TransferFunction { num, den })
    }

    /// Pure gain `k`.
    pub fn gain(k: f64) -> TransferFunction {
        TransferFunction {
            num: vec![k],
            den: vec![1.0],
        }
    }

    /// Numerator coefficients, descending powers.
    pub fn num(&self) -> &[f64] {
        &self.num
    }

    /// Denominator coefficients (leading 1), descending powers.
    pub fn den(&self) -> &[f64] {
        &self.den
    }

    /// Evaluates `H(s)` at a complex point by Horner's rule.
    pub fn eval(&self, s: C64) -> C64 {
        let poly = |c: &[f64]| -> C64 {
            let mut acc = C64::ZERO;
            for &coef in c {
                acc = acc * s + C64::real(coef);
            }
            acc
        };
        poly(&self.num) / poly(&self.den)
    }

    /// Evaluates on the imaginary axis at frequency `omega` (s = j omega).
    pub fn eval_jw(&self, omega: f64) -> C64 {
        self.eval(C64::new(0.0, omega))
    }

    /// DC gain `H(0)` (denominator must not have a zero at the origin).
    pub fn dc_gain(&self) -> f64 {
        let inv = 1.0 / *self.den.last().unwrap();
        inv * *self.num.last().unwrap()
    }

    /// Series (cascade) connection: `self * other`.
    pub fn series(&self, other: &TransferFunction) -> TransferFunction {
        let num = poly_mul(&self.num, &other.num);
        let den = poly_mul(&self.den, &other.den);
        TransferFunction { num, den }
    }

    /// Parallel connection: `self + other` (common denominator).
    pub fn parallel(&self, other: &TransferFunction) -> TransferFunction {
        let den = poly_mul(&self.den, &other.den);
        let n1 = poly_mul(&self.num, &other.den);
        let n2 = poly_mul(&other.num, &self.den);
        let num = poly_add(&n1, &n2);
        TransferFunction { num, den }
    }

    /// Unity-feedback closed loop `self / (1 + self)` (negative feedback).
    pub fn closed_loop(&self) -> TransferFunction {
        let one = TransferFunction::gain(1.0);
        // 1 + G has numerator (den + num) — parallel()'s numerator.
        let den = self.parallel(&one).num;
        let num = self.num.clone();
        TransferFunction { num, den }
    }

    /// Roots of a polynomial via the eigenvalues of its companion
    /// matrix — reuses the core real-Schur solver.
    fn poly_roots(coeffs: &[f64]) -> Vec<C64> {
        // Normalize to monic: p(s) = s^n + c1 s^{n-1} + ... + cn.
        let lead = coeffs[0];
        let c: Vec<f64> = coeffs.iter().map(|x| x / lead).collect();
        let n = c.len() - 1;
        if n == 0 {
            return Vec::new();
        }
        // Companion form with subdiagonal ones; the last column holds
        // -[c_n, c_{n-1}, ..., c_1] top to bottom, giving
        // det(lI - C) = l^n + c1 l^{n-1} + ... + cn.
        let mut comp = DMat::<f64>::zeros(n, n);
        for i in 0..n - 1 {
            comp[(i + 1, i)] = 1.0;
        }
        for (j, &cf) in c[1..].iter().rev().enumerate() {
            comp[(j, n - 1)] = -cf;
        }
        comp.eigenvalues().unwrap_or_default()
    }

    /// Poles of the transfer function.
    pub fn poles(&self) -> Vec<C64> {
        Self::poly_roots(&self.den)
    }

    /// Zeros of the transfer function (numerator roots; a strictly proper
    /// TF has trailing zeros at infinity, which are not returned).
    pub fn zeros(&self) -> Vec<C64> {
        Self::poly_roots(&self.num)
    }

    /// Order of the denominator.
    pub fn order(&self) -> usize {
        self.den.len() - 1
    }

    /// Controllable-canonical state-space realization
    /// `x' = A x + B u, y = C x + D u` (phase-variable form).
    pub fn to_state_space(&self) -> (DMat<f64>, DMat<f64>, DMat<f64>, DMat<f64>) {
        let n = self.order();
        let a = DMat::from_fn(n, n, |i, j| {
            if i + 1 == j {
                1.0
            } else if i == n - 1 {
                -self.den[n - j]
            } else {
                0.0
            }
        });
        let b = DMat::from_fn(n, 1, |i, _| if i == n - 1 { 1.0 } else { 0.0 });
        // num is padded to length n+1 (descending): [b0, b1, ..., bn]
        let mut bp = vec![0.0; n + 1];
        let off = n + 1 - self.num.len();
        bp[off..].copy_from_slice(&self.num);
        // Biproper feedthrough and the strictly-proper remainder
        // num - d * den = r1 s^{n-1} + ... + rn, with C = [rn, ..., r1]
        // because (sI - A)^-1 B = [1, s, ..., s^{n-1}]^T / den for this form.
        let d = if self.num.len() == n + 1 { bp[0] } else { 0.0 };
        let c = DMat::from_fn(1, n, |_, j| bp[n - j] - d * self.den[n - j]);
        (a, b, c, DMat::from_rows(&[&[d]]))
    }
}

impl fmt::Debug for TransferFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TF(num: {:?}, den: {:?})", self.num, self.den)
    }
}

/// Polynomial multiplication (descending coefficients).
fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

/// Polynomial addition, aligning by descending powers.
fn poly_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len().max(b.len());
    let mut out = vec![0.0; n];
    for (i, &x) in a.iter().enumerate() {
        out[n - a.len() + i] += x;
    }
    for (i, &x) in b.iter().enumerate() {
        out[n - b.len() + i] += x;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ctrl_verify::testing::assert_close;

    #[test]
    fn eval_matches_hand_computation() {
        // H(s) = (s + 2) / (s^2 + 3s + 4)
        let h = TransferFunction::new(&[1.0, 2.0], &[1.0, 3.0, 4.0]).unwrap();
        let s = C64::new(0.0, 2.0);
        let num = s + C64::real(2.0);
        let den = s * s + C64::real(3.0) * s + C64::real(4.0);
        let expected = num / den;
        let got = h.eval(s);
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn poles_of_quadratic() {
        // s^2 + 2 s + 5: poles -1 +/- 2j
        let h = TransferFunction::new(&[1.0], &[1.0, 2.0, 5.0]).unwrap();
        let poles = h.poles();
        assert_eq!(poles.len(), 2);
        let mut sorted: Vec<(f64, f64)> = poles.iter().map(|z| (z.re, z.im)).collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        assert_close(sorted[0].0, -1.0, 1e-9);
        assert_close(sorted[0].1, -2.0, 1e-9);
        assert_close(sorted[1].0, -1.0, 1e-9);
        assert_close(sorted[1].1, 2.0, 1e-9);
    }

    #[test]
    fn series_and_parallel() {
        let g1 = TransferFunction::new(&[1.0], &[1.0, 1.0]).unwrap(); // 1/(s+1)
        let g2 = TransferFunction::new(&[2.0], &[1.0, 2.0]).unwrap(); // 2/(s+2)
        let ser = g1.series(&g2);
        let s = C64::new(0.5, 1.0);
        assert!((ser.eval(s) - g1.eval(s) * g2.eval(s)).abs() < 1e-12);
        let par = g1.parallel(&g2);
        assert!((par.eval(s) - (g1.eval(s) + g2.eval(s))).abs() < 1e-12);
    }

    #[test]
    fn closed_loop_first_order() {
        // G = 2/(s+3), closed loop G/(1+G) = 2/(s+5).
        let g = TransferFunction::new(&[2.0], &[1.0, 3.0]).unwrap();
        let cl = g.closed_loop();
        assert_eq!(cl.den().len(), 2);
        assert_close(cl.den()[1], 5.0, 1e-12);
        assert_close(cl.dc_gain(), 0.4, 1e-12);
    }

    #[test]
    fn rejects_improper() {
        assert_eq!(
            TransferFunction::new(&[1.0, 1.0], &[1.0]),
            Err(TfError::Improper)
        );
        assert!(TransferFunction::new(&[1.0], &[0.0, 1.0]).is_err());
    }

    #[test]
    fn state_space_realization_matches_tf() {
        let h = TransferFunction::new(&[2.0, 1.0], &[1.0, 3.0, 4.0]).unwrap();
        let (a, b, c, d) = h.to_state_space();
        // Evaluate C (sI - A)^-1 B + D on a complex grid via the real 2n
        // system [[sI - A, sIm? ...]]: solve (sI - A) z = B with complex s
        // using the equivalent real system.
        let n = 2;
        for omega in [0.0, 0.7, 3.1] {
            let sigma = 0.5;
            let s = C64::new(sigma, omega);
            let mut m = DMat::zeros(2 * n, 2 * n);
            for i in 0..n {
                for j in 0..n {
                    // block layout: [R  -wI; wI  R] acting on (re, im)
                    m[(i, j)] = -a[(i, j)];
                    m[(i + n, j + n)] = -a[(i, j)];
                    if i == j {
                        m[(i, j)] += sigma;
                        m[(i + n, j + n)] += sigma;
                    }
                    m[(i, j + n)] += -omega * if i == j { 1.0 } else { 0.0 };
                    m[(i + n, j)] += omega * if i == j { 1.0 } else { 0.0 };
                }
            }
            let mut rhs = DMat::zeros(2 * n, 1);
            for i in 0..n {
                rhs[(i, 0)] = b[(i, 0)];
            }
            let sol = m.solve(&rhs).unwrap();
            let mut y = C64::ZERO;
            for j in 0..n {
                y = y + C64::real(c[(0, j)]) * C64::new(sol[(j, 0)], sol[(j + n, 0)]);
            }
            let total = y + C64::real(d[(0, 0)]);
            let tf_val = h.eval(s);
            assert!(
                (total - tf_val).abs() < 1e-9,
                "omega={omega}: ss {total} vs tf {tf_val}"
            );
        }
    }
}
