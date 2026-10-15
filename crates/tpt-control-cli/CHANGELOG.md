# Changelog

All notable changes to `tpt-control-cli` are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `tpt lqr <plant.toml>`: LQR (continuous) or DLQR (discrete) design from a TOML plant file, printing the gain, the Riccati solution and a certificate.
- Exit codes `0` (certified), `1` (not certified), `2` (bad input or solver failure), for use in CI.
- Sample plants in `plants/`.
- README and this changelog.
