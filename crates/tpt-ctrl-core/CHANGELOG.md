# Changelog

All notable changes to `tpt-ctrl-core` are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Runnable example `discretize_and_simulate`: ZOH vs. Tustin on a mass-spring system, with a simulated step response.

## [0.1.0] — 2026-10-06

### Added
- Const-generic dimensioned matrices (`Mat<T, R, C>`) and the runtime workhorse `DMat<T>` with the full operator set.
- From-scratch factorizations: LU with partial pivoting, Cholesky, Householder QR, Hessenberg reduction, real Schur via Francis double-shift QR.
- Pade-13 scaling-and-squaring matrix exponential.
- Lyapunov solvers: continuous (Bartels–Stewart over real Schur) and discrete (squaring/doubling).
- `StateSpace<T, N, M, P>` with continuous and discrete wrappers; exact zero-order-hold and Tustin discretization; RK4 and discrete simulation; poles, stability, controllability and observability analysis.
- Finite-difference Jacobian and gradient hooks.
- `no_std` + `alloc` support, with `libm` for the floating-point kernels.

[0.1.0]: https://github.com/tpt-solutions/tpt-control/releases/tag/v0.1.0
