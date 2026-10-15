//! `tpt` — design and certify a state-feedback gain from a plant file.
//!
//! ```text
//! tpt lqr plant.toml      # design LQR/DLQR and print a certificate
//! tpt --help
//! ```
//!
//! Plant file (`kind` is `"continuous"` or `"discrete"`):
//!
//! ```toml
//! kind = "continuous"
//! a = [[0.0, 1.0], [0.0, 0.0]]
//! b = [[0.0], [1.0]]
//! q = [[1.0, 0.0], [0.0, 1.0]]
//! r = [[1.0]]
//! ```
//!
//! Exit status: `0` when the design is certified, `1` when the certificate
//! fails (so the command can gate a CI job), `2` for bad input.

use std::process::ExitCode;

use serde::Deserialize;
use tpt_control::optimal::{certify_continuous, certify_discrete, dlqr, lqr, Certificate};
use tpt_control::prelude::*;

const USAGE: &str = "\
tpt — design and certify LQR gains

USAGE:
    tpt lqr <plant.toml>     design LQR (or DLQR) and certify the result
    tpt --help               show this message

Certification tolerance: relative Riccati residual <= 1e-8 and a strictly
stable closed loop.";

const RESIDUAL_TOL: f64 = 1e-8;

#[derive(Deserialize)]
struct Plant {
    kind: Kind,
    a: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    q: Vec<Vec<f64>>,
    r: Vec<Vec<f64>>,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Kind {
    Continuous,
    Discrete,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [flag] if flag == "--help" || flag == "-h" => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        [cmd, path] if cmd == "lqr" => match run_lqr(path) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::from(1),
            Err(msg) => {
                eprintln!("tpt: {msg}");
                ExitCode::from(2)
            }
        },
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// Returns `Ok(true)` when the certificate passes.
fn run_lqr(path: &str) -> Result<bool, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let plant: Plant = toml::from_str(&text).map_err(|e| format!("{path}: {e}"))?;

    let a = to_dmat(&plant.a, "a")?;
    let b = to_dmat(&plant.b, "b")?;
    let q = to_dmat(&plant.q, "q")?;
    let r = to_dmat(&plant.r, "r")?;

    let (k, x, cert) = match plant.kind {
        Kind::Continuous => {
            let sol = lqr(&a, &b, &q, &r).map_err(|e| e.to_string())?;
            let cert =
                certify_continuous(&a, &b, &q, &r, &sol.x, &sol.k).map_err(|e| e.to_string())?;
            (sol.k, sol.x, cert)
        }
        Kind::Discrete => {
            let sol = dlqr(&a, &b, &q, &r).map_err(|e| e.to_string())?;
            let cert =
                certify_discrete(&a, &b, &q, &r, &sol.x, &sol.k).map_err(|e| e.to_string())?;
            (sol.k, sol.x, cert)
        }
    };

    print_matrix("K (u = -K x)", &k);
    print_matrix("X (Riccati solution)", &x);
    print_certificate(&cert, plant.kind);
    Ok(cert.is_certified(RESIDUAL_TOL))
}

fn to_dmat(rows: &[Vec<f64>], name: &str) -> Result<DMat<f64>, String> {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    if nrows == 0 || ncols == 0 {
        return Err(format!("matrix `{name}` is empty"));
    }
    if rows.iter().any(|row| row.len() != ncols) {
        return Err(format!("matrix `{name}` is not rectangular"));
    }
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    Ok(DMat::from_row_slice(nrows, ncols, &flat))
}

fn print_matrix(title: &str, m: &DMat<f64>) {
    println!("{title}:");
    for i in 0..m.nrows() {
        let cells: Vec<String> = (0..m.ncols())
            .map(|j| format!("{:>12.6}", m[(i, j)]))
            .collect();
        println!("  [{}]", cells.join(" "));
    }
}

fn print_certificate(cert: &Certificate, kind: Kind) {
    let margin_label = match kind {
        Kind::Continuous => "closed-loop margin (-max Re λ)",
        Kind::Discrete => "closed-loop margin (1 - max |λ|)",
    };
    println!("certificate:");
    println!(
        "  riccati residual      {:.3e}  (tol {RESIDUAL_TOL:.0e})",
        cert.riccati_residual
    );
    println!("  {margin_label:<30}{:.6}", cert.closed_loop_margin);
    let verdict = if cert.is_certified(RESIDUAL_TOL) {
        "CERTIFIED"
    } else {
        "NOT CERTIFIED"
    };
    println!("  verdict               {verdict}");
}
