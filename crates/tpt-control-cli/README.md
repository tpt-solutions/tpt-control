# tpt-control-cli (`tpt`)

Command-line tool for the [tpt-control](https://github.com/tpt-solutions/tpt-control) stack. Give it a plant file and it designs an LQR (continuous) or DLQR (discrete) state-feedback gain, prints the gain and the Riccati solution, and certifies the result. Its exit status can gate a CI job.

## Install

The crate is part of the workspace and is not on crates.io yet:

```sh
cargo install --path crates/tpt-control-cli
```

This installs a binary named `tpt`.

## Usage

```text
tpt lqr <plant.toml>     design LQR (or DLQR) and certify the result
tpt --help               show this message
```

### Plant file

`kind` is `"continuous"` or `"discrete"`. The matrices are row-major lists:

```toml
kind = "continuous"
a = [[0.0, 1.0], [0.0, 0.0]]   # n x n
b = [[0.0], [1.0]]             # n x m
q = [[1.0, 0.0], [0.0, 1.0]]   # n x n, state cost
r = [[1.0]]                    # m x m, input cost
```

### Example

```sh
tpt lqr crates/tpt-control-cli/plants/double_integrator.toml
```

```text
K (u = -K x):
  [    1.000000     1.732051]
X (Riccati solution):
  [    1.732051     1.000000]
  [    1.000000     1.732051]
certificate:
  riccati residual      1.160e-16  (tol 1e-8)
  closed-loop margin (-max Re λ)0.866025
  verdict               CERTIFIED
```

The certificate has two checks. The **riccati residual** is the relative size of the Riccati equation's error at the returned solution, and it must be at most `1e-8`. The **closed-loop margin** is how far the closed loop sits inside the stability boundary, and it must be strictly positive.

The sample plants in [`plants/`](https://github.com/tpt-solutions/tpt-control/tree/master/crates/tpt-control-cli/plants) cover the common cases:

| File | What it shows | Result |
|------|---------------|--------|
| `double_integrator.toml` | continuous LQR | certified |
| `sampled_double_integrator.toml` | discrete DLQR at 0.1 s | certified |
| `unstable_unobservable.toml` | unstabilizable plant, no stabilizing gain | error, exit 2 |

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Design certified |
| `1` | Design computed but the certificate failed |
| `2` | Bad input (unreadable file, malformed or non-rectangular matrix) or the solver failed |

Use it in CI to fail a build when a controller no longer certifies:

```sh
tpt lqr controllers/pitch.toml || { echo "gain not certified"; exit 1; }
```

## Tests

```sh
cargo test -p tpt-control-cli
```

## Status

Pre-1.0. The command set and output format may change between minor versions. See [CHANGELOG.md](CHANGELOG.md).

Dual-licensed under MIT OR Apache-2.0.
