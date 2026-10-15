# Changelog

All notable changes to the `tpt-control` Python package are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- PyO3 bindings published as the `tpt_control` module: `lqr_gain`, `dlqr_gain`, `certify`, `c2d`, `tf2ss`, `eigenvalues`, and the `Certificate` class.
- Matrices are lists of rows, so no NumPy dependency.
- MATLAB name mapping in `docs/MATLAB.md`.
- README and this changelog.
