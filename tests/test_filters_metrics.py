import numpy as np

from garch_groundroll import (agc, fk_fan_filter, fk_mask_filter, local_correlation,
                              ormsby_filter, snr_db, snr_report)
from garch_groundroll.filters import ormsby_weights


def test_ormsby_weights_trapezoid():
    w = ormsby_weights(np.array([0, 5, 10, 15, 20, 30, 40]), (5, 10, 20, 40))
    np.testing.assert_allclose(w, [0, 0, 1, 1, 1, 0.5, 0])


def test_ormsby_bandstop_and_complement():
    dt = 0.002
    t = np.arange(1000) * dt
    lo, hi = np.sin(2 * np.pi * 5 * t), np.sin(2 * np.pi * 60 * t)
    out = ormsby_filter((lo + hi)[:, None], dt, (0, 0, 15, 25), "bandstop")[:, 0]
    c = slice(100, 900)
    assert np.sqrt(np.mean((out[c] - hi[c]) ** 2)) < 0.05
    bp = ormsby_filter((lo + hi)[:, None], dt, (0, 0, 15, 25), "bandpass")[:, 0]
    np.testing.assert_allclose(bp + out, lo + hi, atol=1e-10)


def test_fk_mask_identity():
    d = np.random.default_rng(0).standard_normal((64, 32))
    np.testing.assert_allclose(fk_mask_filter(d, np.zeros_like(d)), d, atol=1e-12)
    np.testing.assert_allclose(fk_mask_filter(d, np.ones_like(d)), 0, atol=1e-10)


def test_fk_fan_keeps_fast_removes_slow():
    dt, dx, nt, nx = 0.004, 10.0, 256, 64
    t = np.arange(nt)[:, None] * dt
    x = np.arange(nx)[None, :] * dx
    fast = np.exp(-(((t - 0.4 - x / 5000) / 0.02) ** 2))
    slow = np.exp(-(((t - 0.1 - x / 300) / 0.05) ** 2))
    out = fk_fan_filter(fast + slow, dt, dx, 800, 1200)
    assert np.linalg.norm(out - fast) / np.linalg.norm(slow) < 0.35


def test_snr_and_report():
    assert abs(snr_db(np.ones(10), 0.1 * np.ones(10)) - 20) < 1e-12
    r = snr_report(np.ones(4), 0.5 * np.ones(4), np.ones(4) * 1.1, 0.4 * np.ones(4))
    assert r["gain_db"] > 0 and abs(r["gain_db"] - (r["snr_after_db"] - r["snr_before_db"])) < 1e-12


def test_local_correlation_of_identical_images():
    a = np.random.default_rng(1).standard_normal((50, 40))
    assert local_correlation(a, 3 * a + 1, 9).min() > 0.999


def test_agc_unit_rms():
    rng = np.random.default_rng(2)
    d = rng.standard_normal((2000, 3)) * np.linspace(1, 100, 2000)[:, None]
    out = agc(d, 100)
    assert 0.8 < np.sqrt(np.mean(out[200:-200] ** 2)) < 1.2
