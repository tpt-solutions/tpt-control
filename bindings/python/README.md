# tpt-control (Python)

Python bindings for [tpt-control](../../README.md), a pure-Rust control systems
library with no LAPACK or solver dependencies. Matrices are lists of rows, so
there is no NumPy requirement.

```python
import tpt_control as tc

A = [[0.0, 1.0], [0.0, 0.0]]
B = [[0.0], [1.0]]
Q = [[1.0, 0.0], [0.0, 1.0]]
R = [[1.0]]

K, X = tc.lqr_gain(A, B, Q, R)          # K = [[1.0, 1.732...]]
cert = tc.certify(A, B, Q, R, X, K)      # continuous CARE check
assert cert.is_certified()
print(cert)                              # Certificate(riccati_residual=..., ...)
```

Functions: `lqr_gain`, `dlqr_gain`, `certify`, `c2d`, `tf2ss`, `eigenvalues`.
Errors raise `ValueError`. MATLAB name mapping: see `docs/MATLAB.md`.

Build locally with `maturin develop` (inside a virtualenv) and run `pytest tests`.
