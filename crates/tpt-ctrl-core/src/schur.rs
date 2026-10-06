//! Real Schur decomposition via the Francis double-shift QR iteration.
//!
//! For a real square matrix `A`, produces orthogonal `Q` and upper
//! quasi-triangular `T` (1x1 and 2x2 diagonal blocks) with `A = Q T Q^T`.
//! This is the backbone for eigenvalue computation, the Bartels-Stewart
//! Lyapunov solve, and (in `tpt-ctrl-optimal`) Riccati solvers.

use alloc::vec::Vec;

use crate::c64::C64;
use crate::dmat::{require_square, DMat};
use crate::error::{LinalgError, Result};
use crate::field::Field;

/// Result of a real Schur decomposition: `A = Q T Q^T`.
pub struct Schur<T> {
    /// Upper quasi-triangular Schur factor.
    pub t: DMat<T>,
    /// Orthogonal factor.
    pub q: DMat<T>,
}

impl<T: Field + core::fmt::Debug> core::fmt::Debug for Schur<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Schur")
            .field("t", &self.t)
            .field("q", &self.q)
            .finish()
    }
}

/// Householder reflector `(v, beta)` for `I - beta v v^T` zeroing all but the
/// first entry of `x`. `beta == 0` means "already reduced, skip".
fn house<T: Field>(x: &[T]) -> (Vec<T>, T) {
    let mut norm = T::zero();
    for &xi in x {
        norm = norm.hypot(xi);
    }
    if norm == T::zero() {
        return (x.to_vec(), T::zero());
    }
    let alpha = if x[0] >= T::zero() { -norm } else { norm };
    let mut v = x.to_vec();
    v[0] -= alpha;
    let mut vv = T::zero();
    for &vi in &v {
        vv += vi * vi;
    }
    if vv == T::zero() {
        return (v, T::zero());
    }
    (v, T::from_f64(2.0) / vv)
}

/// Applies the Francis double-shift QR iteration to the Hessenberg matrix
/// `h` in place, accumulating orthogonal transforms into `q`.
fn francis_qr<T: Field>(h: &mut DMat<T>, q: &mut DMat<T>, want_q: bool) -> Result<()> {
    let n = h.nrows();
    if n <= 2 {
        // 1x1 done; 2x2 handled by the caller via block deflation.
        return Ok(());
    }
    let eps = T::from_f64(f64::EPSILON);
    let anorm = h.norm_inf();
    let budget = 100 * n;
    let mut total_iters = 0usize;
    let mut block_iters = 0usize;
    let mut iend = n - 1;

    while iend > 0 {
        // Zero negligible subdiagonal entries.
        for i in 1..=iend {
            let s = h[(i - 1, i - 1)].abs() + h[(i, i)].abs();
            let tol = if s == T::zero() { eps * anorm } else { eps * s };
            if h[(i, i - 1)].abs() <= tol {
                h[(i, i - 1)] = T::zero();
            }
        }
        // Start of the trailing unreduced block.
        let mut l = iend;
        while l > 0 && h[(l, l - 1)] != T::zero() {
            l -= 1;
        }
        let m = iend - l + 1;
        if m == 1 {
            iend -= 1;
            block_iters = 0;
            continue;
        }
        if m == 2 {
            deflate_2x2(h, q, l, want_q);
            if l == 0 {
                iend = 0;
            } else {
                iend = l - 1;
            }
            block_iters = 0;
            continue;
        }

        total_iters += 1;
        block_iters += 1;
        if total_iters > budget {
            return Err(LinalgError::NoConvergence);
        }

        // Double shifts: eigenvalues of the trailing 2x2, or an exceptional
        // shift every 10 stalled iterations (EISPACK-style ad hoc values).
        let (s_shift, t_shift) = if block_iters % 10 == 0 {
            let ss = h[(iend, iend - 1)].abs() + h[(iend - 1, iend - 2)].abs();
            (T::from_f64(1.5) * ss, -T::from_f64(0.4375) * ss * ss)
        } else {
            let s = h[(iend - 1, iend - 1)] + h[(iend, iend)];
            let t = h[(iend - 1, iend - 1)] * h[(iend, iend)]
                - h[(iend - 1, iend)] * h[(iend, iend - 1)];
            (s, t)
        };

        // First column of (H - s1 I)(H - s2 I) on the active block.
        let mut x =
            h[(l, l)] * h[(l, l)] + h[(l, l + 1)] * h[(l + 1, l)] - s_shift * h[(l, l)] + t_shift;
        let mut y = h[(l + 1, l)] * (h[(l, l)] + h[(l + 1, l + 1)] - s_shift);
        let mut z = h[(l + 1, l)] * h[(l + 2, l + 1)];

        // Bulge chase: every step is a 3-row reflector (Golub & Van Loan,
        // Alg. 7.5.2). Step r acts on rows r..r+2 with bulge column r-1.
        for r in l..=(iend - 2) {
            let (v, beta) = house(&[x, y, z]);
            if beta != T::zero() {
                // Left: rows r..r+2, columns from the bulge column.
                let c_lo = if r > 0 { r - 1 } else { 0 };
                #[allow(clippy::needless_range_loop)]
                for j in c_lo..n {
                    let mut acc = T::zero();
                    for (idx, i) in (r..r + 3).enumerate() {
                        acc += v[idx] * h[(i, j)];
                    }
                    acc *= beta;
                    for (idx, i) in (r..r + 3).enumerate() {
                        h[(i, j)] -= acc * v[idx];
                    }
                }
                // Right: columns r..r+2, rows down to the bulge row.
                let r_hi = (r + 4).min(iend + 1);
                for i in 0..r_hi {
                    let mut acc = T::zero();
                    for (idx, j) in (r..r + 3).enumerate() {
                        acc += h[(i, j)] * v[idx];
                    }
                    acc *= beta;
                    for (idx, j) in (r..r + 3).enumerate() {
                        h[(i, j)] -= acc * v[idx];
                    }
                }
                if want_q {
                    for i in 0..n {
                        let mut acc = T::zero();
                        for (idx, j) in (r..r + 3).enumerate() {
                            acc += q[(i, j)] * v[idx];
                        }
                        acc *= beta;
                        for (idx, j) in (r..r + 3).enumerate() {
                            q[(i, j)] -= acc * v[idx];
                        }
                    }
                }
                // The reflector zeroes the bulge entries of column r-1.
                if r > l {
                    h[(r + 1, r - 1)] = T::zero();
                    h[(r + 2, r - 1)] = T::zero();
                }
            }
            // Advance the bulge column.
            x = h[(r + 1, r)];
            y = h[(r + 2, r)];
            if r + 3 <= iend {
                z = h[(r + 3, r)];
            }
        }

        // Final 2-row reflector on the last bulge column (rows iend-1, iend).
        let (v, beta) = house(&[x, y]);
        if beta != T::zero() {
            let r = iend - 1;
            let c_lo = iend - 2;
            #[allow(clippy::needless_range_loop)]
            for j in c_lo..n {
                let mut acc = T::zero();
                for (idx, i) in (r..r + 2).enumerate() {
                    acc += v[idx] * h[(i, j)];
                }
                acc *= beta;
                for (idx, i) in (r..r + 2).enumerate() {
                    h[(i, j)] -= acc * v[idx];
                }
            }
            for i in 0..iend + 1 {
                let mut acc = T::zero();
                for (idx, j) in (r..r + 2).enumerate() {
                    acc += h[(i, j)] * v[idx];
                }
                acc *= beta;
                for (idx, j) in (r..r + 2).enumerate() {
                    h[(i, j)] -= acc * v[idx];
                }
            }
            if want_q {
                for i in 0..n {
                    let mut acc = T::zero();
                    for (idx, j) in (r..r + 2).enumerate() {
                        acc += q[(i, j)] * v[idx];
                    }
                    acc *= beta;
                    for (idx, j) in (r..r + 2).enumerate() {
                        q[(i, j)] -= acc * v[idx];
                    }
                }
            }
            h[(iend, iend - 2)] = T::zero();
        }
    }
    Ok(())
}

/// Reduces the trailing 2x2 block at `(l, l)` to Schur form: real eigenvalue
/// pairs are split into two 1x1 blocks; complex pairs are left as a 2x2 block.
fn deflate_2x2<T: Field>(h: &mut DMat<T>, q: &mut DMat<T>, l: usize, want_q: bool) {
    let n = h.nrows();
    let a = h[(l, l)];
    let b = h[(l, l + 1)];
    let c = h[(l + 1, l)];
    let d = h[(l + 1, l + 1)];
    let p = (a - d) * T::from_f64(0.5);
    let disc = p * p + b * c;
    if disc < T::zero() {
        return; // complex conjugate pair: already in real Schur form
    }
    let sq = disc.sqrt();
    let mean = (a + d) * T::from_f64(0.5);

    // Pick the root with the larger magnitude to avoid cancellation.
    let lam = if mean >= T::zero() {
        mean + sq
    } else {
        mean - sq
    };
    // Schur vector for `lam`: row 2 gives (lam - d, c), row 1 gives (b, lam - a).
    // Use whichever candidate is better scaled — when the block is nearly
    // converged one of them degenerates to rounding noise.
    let e1 = (lam - d, c);
    let e2 = (b, lam - a);
    let (v0, v1) = if e1.0.hypot(e1.1) >= e2.0.hypot(e2.1) {
        e1
    } else {
        e2
    };
    let nv = v0.hypot(v1);
    if nv == T::zero() {
        return;
    }
    let cs = v0 / nv;
    let sn = v1 / nv;
    // G = [[cs, -sn], [sn, cs]]; T <- G^T T G and Q <- Q G.
    // Left multiplication by G^T on rows l, l+1:
    let col_lo = if l > 0 { l - 1 } else { 0 };
    for j in col_lo..n {
        let t0 = h[(l, j)];
        let t1 = h[(l + 1, j)];
        h[(l, j)] = cs * t0 + sn * t1;
        h[(l + 1, j)] = -sn * t0 + cs * t1;
    }
    // Right multiplication by G on columns l, l+1:
    let r_hi = (l + 3).min(n);
    for i in 0..r_hi {
        let t0 = h[(i, l)];
        let t1 = h[(i, l + 1)];
        h[(i, l)] = cs * t0 + sn * t1;
        h[(i, l + 1)] = -sn * t0 + cs * t1;
    }
    if want_q {
        for i in 0..n {
            let t0 = q[(i, l)];
            let t1 = q[(i, l + 1)];
            q[(i, l)] = cs * t0 + sn * t1;
            q[(i, l + 1)] = -sn * t0 + cs * t1;
        }
    }
    h[(l + 1, l)] = T::zero();
}

/// Computes the real Schur decomposition `A = Q T Q^T`.
pub fn schur<T: Field>(a: &DMat<T>, want_q: bool) -> Result<Schur<T>> {
    require_square(a)?;
    let n = a.nrows();
    let (mut t, mut q) = crate::hessenberg::hessenberg(a)?;
    if n >= 3 {
        francis_qr(&mut t, &mut q, want_q)?;
    } else if n == 2 {
        deflate_2x2(&mut t, &mut q, 0, want_q);
    }
    Ok(Schur { t, q })
}

/// Extracts the eigenvalues of a real Schur factor `T` (quasi-triangular).
pub fn eigenvalues_from_schur<T: Field>(t: &DMat<T>) -> Vec<C64> {
    let n = t.nrows();
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if i + 1 < n && t[(i + 1, i)] != T::zero() {
            let a = t[(i, i)];
            let b = t[(i, i + 1)];
            let c = t[(i + 1, i)];
            let d = t[(i + 1, i + 1)];
            let half = T::from_f64(0.5);
            let p = (a - d) * half;
            let disc = p * p + b * c;
            let mean = (a + d) * half;
            if disc < T::zero() {
                let im = (-disc).sqrt();
                out.push(C64::new(mean.to_f64(), im.to_f64()));
                out.push(C64::new(mean.to_f64(), -im.to_f64()));
            } else {
                let sq = disc.sqrt();
                out.push(C64::real((mean + sq).to_f64()));
                out.push(C64::real((mean - sq).to_f64()));
            }
            i += 2;
        } else {
            out.push(C64::real(t[(i, i)].to_f64()));
            i += 1;
        }
    }
    out
}

impl<T: Field> DMat<T> {
    /// Real Schur decomposition `A = Q T Q^T` (see [`schur`]).
    pub fn schur(&self) -> Result<Schur<T>> {
        schur(self, true)
    }

    /// Eigenvalues of the matrix (real Schur iteration, no transform
    /// accumulation).
    pub fn eigenvalues(&self) -> Result<Vec<C64>> {
        let s = schur(self, false)?;
        Ok(eigenvalues_from_schur(&s.t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn mat_close(a: &DMat<f64>, b: &DMat<f64>) -> bool {
        a.nrows() == b.nrows()
            && a.ncols() == b.ncols()
            && a.as_slice()
                .iter()
                .zip(b.as_slice())
                .all(|(x, y)| (*x - *y).abs() < 1e-9)
    }

    fn sort_eigs(mut v: Vec<C64>) -> Vec<C64> {
        v.sort_by(|a, b| {
            a.re.partial_cmp(&b.re)
                .unwrap()
                .then(a.im.partial_cmp(&b.im).unwrap())
        });
        v
    }

    #[test]
    fn diagonal_matrix_eigenvalues() {
        let a = DMat::from_rows(&[&[2.0, 0.0], &[0.0, -3.0]]);
        let mut ev = a.eigenvalues().unwrap();
        ev.sort_by(|x, y| x.re.partial_cmp(&y.re).unwrap());
        assert!(close(ev[0].re, -3.0) && close(ev[1].re, 2.0));
    }

    #[test]
    fn rotation_matrix_has_complex_pair() {
        // 90-degree rotation: eigenvalues ±i.
        let a = DMat::from_rows(&[&[0.0, -1.0], &[1.0, 0.0]]);
        let ev = a.eigenvalues().unwrap();
        assert_eq!(ev.len(), 2);
        assert!(close(ev[0].re, 0.0));
        assert!((ev[0].im.abs() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn schur_reconstructs_and_eigs_match() {
        // Companion matrix with a nasty mix of real and complex eigenvalues.
        let a = DMat::from_rows(&[
            &[0.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0, 1.0],
            &[-1.0, 0.5, 3.0, -2.0],
        ]);
        let s = a.schur().unwrap();
        let qhq = &(&s.q * &s.t) * &s.q.transpose();
        assert!(mat_close(&qhq, &a), "Q T Q^T != A");
        assert!(mat_close(&(&s.q.transpose() * &s.q), &DMat::identity(4)));
        let e1 = sort_eigs(a.eigenvalues().unwrap());
        let e2 = sort_eigs(eigenvalues_from_schur(&s.t));
        for (x, y) in e1.iter().zip(&e2) {
            assert!((x.re - y.re).abs() < 1e-9 && (x.im - y.im).abs() < 1e-9);
        }
    }

    #[test]
    fn upper_triangular_already_schur() {
        let a = DMat::from_rows(&[&[1.0, 5.0, 7.0], &[0.0, -2.0, 3.0], &[0.0, 0.0, 4.0]]);
        let ev = a.eigenvalues().unwrap();
        let mut reals: Vec<f64> = ev.iter().map(|z| z.re).collect();
        reals.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!(close(reals[0], -2.0) && close(reals[1], 1.0) && close(reals[2], 4.0));
        for z in ev {
            assert!(z.im.abs() < 1e-12);
        }
    }

    #[test]
    fn defective_matrix_converges() {
        // Jordan block: eigenvalue 2 with algebraic multiplicity 3.
        let a = DMat::from_rows(&[&[2.0, 1.0, 0.0], &[0.0, 2.0, 1.0], &[0.0, 0.0, 2.0]]);
        let ev = a.eigenvalues().unwrap();
        for z in ev {
            assert!((z.re - 2.0).abs() < 1e-9 && z.im.abs() < 1e-9);
        }
    }

    #[test]
    fn symmetric_matrix_real_spectrum() {
        let a = DMat::from_rows(&[&[2.0, 1.0, 0.0], &[1.0, 2.0, 1.0], &[0.0, 1.0, 2.0]]);
        let ev = a.eigenvalues().unwrap();
        let mut reals: Vec<f64> = ev.iter().map(|z| z.re).collect();
        reals.sort_by(|x, y| x.partial_cmp(y).unwrap());
        // 2 - sqrt(2), 2, 2 + sqrt(2)
        assert!(close(reals[0], 2.0 - core::f64::consts::SQRT_2));
        assert!(close(reals[1], 2.0));
        assert!(close(reals[2], 2.0 + core::f64::consts::SQRT_2));
        for z in ev {
            assert!(z.im.abs() < 1e-9);
        }
    }
}
