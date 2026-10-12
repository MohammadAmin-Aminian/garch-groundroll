import numpy as np
import pytest

from garch_groundroll import fit_garch, fit_garch_traces, garch_conditional_variance, select_garch_order


def simulate(n, omega, a, b, seed):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    s2 = omega / (1 - a - b)
    for t in range(n):
        y[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + a * y[t] ** 2 + b * s2
    return y


def test_parameter_recovery():
    y = 7 * simulate(5000, 0.2, 0.12, 0.80, 1)
    f = fit_garch(y, 1, 1)
    assert abs(f.alpha[0] - 0.12) < 0.05
    assert abs(f.beta[0] - 0.80) < 0.08
    assert abs(f.omega / 49 - 0.2) < 0.15
    assert f.persistence < 1


def test_variance_recursion_matches_definition():
    y = np.array([1.0, -2.0, 0.5, 3.0])
    s2 = garch_conditional_variance(y, 0.1, [0.2], [0.7], backcast=1.0)
    ref, prev_s, prev_y = [], 1.0, 1.0
    for v in y:
        cur = 0.1 + 0.2 * prev_y + 0.7 * prev_s
        ref.append(cur)
        prev_s, prev_y = cur, v * v
    np.testing.assert_allclose(s2, ref, rtol=0, atol=1e-12)


def test_higher_order_recursion():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(50)
    om, al, be = 0.05, [0.1, 0.05], [0.5, 0.2, 0.1]
    s2 = garch_conditional_variance(y, om, al, be, backcast=1.0)
    y2 = np.concatenate([[1.0, 1.0], y ** 2])
    ss = [1.0, 1.0, 1.0]
    for t in range(50):
        v = om + al[0] * y2[t + 1] + al[1] * y2[t] + be[0] * ss[-1] + be[1] * ss[-2] + be[2] * ss[-3]
        ss.append(v)
    np.testing.assert_allclose(s2, ss[3:], atol=1e-12)


def test_scale_equivariance():
    y = simulate(1500, 0.1, 0.2, 0.7, 2)
    f1, f2 = fit_garch(y, 1, 1), fit_garch(1000 * y, 1, 1)
    np.testing.assert_allclose(f1.alpha, f2.alpha, atol=1e-4)
    np.testing.assert_allclose(f2.variance / 1e6, f1.variance, rtol=1e-3)
    assert abs((f2.loglik + y.size * np.log(1000)) - f1.loglik) < 1e-3


def test_zero_trace_is_handled():
    out = fit_garch_traces(np.zeros((100, 3)), 1, 1)
    assert np.all(out["std"] == 0) and not out["converged"].any()


def test_invalid_order():
    with pytest.raises(ValueError):
        fit_garch(np.ones(10), 1, 0)


def test_order_selection_prefers_true_model_with_bic():
    y = simulate(3000, 0.1, 0.15, 0.75, 4)
    sel = select_garch_order(y[:, None], orders=[(1, 1), (2, 2), (3, 3)], criterion="bic")
    assert sel["best_overall"] == (1, 1)
    # raw log-likelihood never decreases for nested models
    assert sel["loglik"][2, 0] >= sel["loglik"][0, 0] - 1e-3
