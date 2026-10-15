"""Smoke and correctness tests for the Python bindings (run with pytest)."""

import math

import pytest

import tpt_control as tc

DOUBLE_INTEGRATOR = dict(
    a=[[0.0, 1.0], [0.0, 0.0]],
    b=[[0.0], [1.0]],
    q=[[1.0, 0.0], [0.0, 1.0]],
    r=[[1.0]],
)


def test_continuous_lqr_matches_textbook_gain():
    k, x = tc.lqr_gain(**DOUBLE_INTEGRATOR)
    # Classic closed form for Q = I, R = 1: K = [1, sqrt(3)].
    assert k[0][0] == pytest.approx(1.0, abs=1e-6)
    assert k[0][1] == pytest.approx(math.sqrt(3.0), abs=1e-6)
    assert x[0][0] == pytest.approx(math.sqrt(3.0), abs=1e-6)


def test_certificate_passes_for_solver_output():
    k, x = tc.lqr_gain(**DOUBLE_INTEGRATOR)
    cert = tc.certify(**DOUBLE_INTEGRATOR, x=x, k=k)
    assert isinstance(cert, tc.Certificate)
    assert cert.is_certified()
    assert cert.stable
    assert cert.closed_loop_margin == pytest.approx(math.sqrt(3.0) / 2.0, abs=1e-6)


def test_certificate_rejects_corrupted_solution():
    k, x = tc.lqr_gain(**DOUBLE_INTEGRATOR)
    bad_x = [[v * 1.5 for v in row] for row in x]
    cert = tc.certify(**DOUBLE_INTEGRATOR, x=bad_x, k=k)
    assert not cert.is_certified()


def test_discrete_lqr_and_certificate():
    a, b = tc.c2d([[0.0, 1.0], [0.0, 0.0]], [[0.0], [1.0]], 0.1)
    q, r = [[1.0, 0.0], [0.0, 1.0]], [[1.0]]
    k, x = tc.dlqr_gain(a, b, q, r)
    cert = tc.certify(a, b, q, r, x, k, continuous=False)
    assert cert.is_certified()
    assert 0.0 < cert.closed_loop_margin < 1.0


def test_c2d_matches_exact_exponential():
    ad, bd = tc.c2d([[-1.0]], [[1.0]], 0.5)
    assert ad[0][0] == pytest.approx(math.exp(-0.5), abs=1e-12)
    assert bd[0][0] == pytest.approx(1.0 - math.exp(-0.5), abs=1e-12)


def test_tf2ss_first_order():
    a, b, c, d = tc.tf2ss([1.0], [1.0, 2.0])
    assert a[0][0] == pytest.approx(-2.0)
    assert b[0][0] == pytest.approx(1.0)
    assert c[0][0] == pytest.approx(1.0)
    assert d[0][0] == pytest.approx(0.0)


def test_eigenvalues_are_pairs():
    eig = tc.eigenvalues([[0.0, 1.0], [-2.0, -3.0]])
    assert sorted(round(re, 9) for re, _ in eig) == [-2.0, -1.0]


def test_ragged_matrix_raises_value_error():
    with pytest.raises(ValueError, match="not rectangular"):
        tc.lqr_gain([[1.0, 0.0], [0.0]], [[1.0], [0.0]], [[1.0]], [[1.0]])


def test_unstabilizable_plant_raises_value_error():
    # Unstable mode with no input authority.
    with pytest.raises(ValueError):
        tc.lqr_gain([[1.0, 0.0], [0.0, 1.0]], [[1.0], [0.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0]])
