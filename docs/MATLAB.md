# MATLAB → tpt-control cheat sheet

Written for: control engineers who already script in MATLAB and want to port a
design to Rust.

The `tpt_control::matlab` module mirrors the MATLAB names and output order, so
most scripts translate line for line. All matrices are row-major `DMat<f64>`.

| MATLAB | tpt-control (Rust) | Notes |
|---|---|---|
| `[A,B,C,D] = tf2ss(num,den)` | `matlab::tf2ss(&num, &den)` | Controllable canonical form. Coefficients in descending powers of `s`. |
| `[Ad,Bd] = c2d(A,B,Ts)` | `matlab::c2d(&a, &b, ts)` | Exact zero-order hold (Van Loan). Tustin: `tpt_ctrl_core::discretize::tustin`. |
| `[K,S,e] = lqr(A,B,Q,R)` | `matlab::lqr(&a, &b, &q, &r)` | Continuous CARE. `e` = eigenvalues of `A-BK`. |
| `[K,S,e] = dlqr(A,B,Q,R)` | `matlab::dlqr(&a, &b, &q, &r)` | Discrete DARE (structure-preserving doubling). |
| `[L,P] = kalman(A,C,Qn,Rn)` | `matlab::kalman(&a, &c, &qn, &rn)` | Steady-state gain and covariance. Returns `(L, P)`. |
| `lqg(...)` | `tpt_ctrl_optimal::lqg` / `dlqg` | Separation principle; returns `LqgController`. |
| `eig(A)` | `tpt_ctrl_core::analysis::poles(&a)` | Returns `Vec<C64>`. |
| `ss(A,B,C,D)` | `DynSystem::new(a, b, c, d)` | Dimensions checked at construction. |
| `c2d` then `step` | `Continuous::discretize_zoh(dt)` then `Discrete::step(&x, &u)` | Const-generic variant with compile-time dimensions. |
| `isstable(A)` | `analysis::is_stable_continuous(&a, tol)` / `is_stable_discrete` | `tol` is the margin the poles must clear. |
| `ctrb(A,B)` / `obsv(A,C)` | `analysis::controllability_matrix` / `observability_matrix` | Plus `is_controllable`, `is_observable`. |
| `rank(M)` | `analysis::rank(&m, tol)` | Explicit tolerance, no hidden default. |
| `[Gm,Pm,...] = margin(G)` | `tpt_ctrl_classic::freq` | Gain and phase margins on a log grid. |
| `pid(Kp,Ki,Kd)` | `tpt_ctrl_classic::pid::PidGains::pid(kp, ki, kd, kaw, tau)` | Includes anti-windup gain and derivative filter. |
| `feedback(G,H)` | `TransferFunction::closed_loop()` | Unity feedback. |

## Worked example

MATLAB:

```matlab
A = [0 1; 0 0];  B = [0; 1];  Q = eye(2);  R = 1;
[K, S, e] = lqr(A, B, Q, R);
```

tpt-control:

```rust
use tpt_control::prelude::*;
use tpt_control::matlab;

let a = DMat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]);
let b = DMat::from_rows(&[&[0.0], &[1.0]]);
let q = DMat::identity(2);
let r = DMat::from_rows(&[&[1.0]]);
let (k, s, e) = matlab::lqr(&a, &b, &q, &r)?;
```

## Things that differ on purpose

* **Errors are values.** Every function returns `Result`; there is no silent
  `NaN` or warning printed to the console.
* **No hidden tolerances.** Stability and rank checks take an explicit `tol`.
* **Certificates are on demand.** MATLAB does not check a Riccati residual;
  `tpt_ctrl_optimal::certify_continuous` / `certify_discrete` do, returning a
  `Certificate` you can gate deployment on.
* **Index base is 0.** `K[(0, 1)]` in Rust is `K(1,2)` in MATLAB.
* **No toolbox object model.** There are no `ss`/`tf` objects with overloaded
  operators; models are plain matrices or `TransferFunction` values.

## Not yet covered

`lqi`, `place` (pole placement), `balreal`, `hinfsyn`, and Simulink-style block
diagrams. Pole placement is the most-requested gap; track it in `todo.md`.
