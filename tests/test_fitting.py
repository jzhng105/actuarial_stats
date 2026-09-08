"""Maximum-likelihood fits must recover the parameters they were built from."""

import math

import actstats
from actstats import actuarial


def _close(got, expected, rtol):
    assert abs(got - expected) <= rtol * abs(expected), f"{got!r} != {expected!r}"


def _sample(name, params, n=20000, seed=17):
    actstats.seed(seed)
    return getattr(actuarial, name)(*params).rvs(size=n)


def test_recovers_parameters():
    cases = [
        ("lognormal", (0.8, 0.45), 0.05),
        ("gamma", (2.5, 3.0), 0.08),
        ("gamma", (0.35, 12.0), 0.08),
        ("weibull", (1.7, 4.0), 0.06),
        ("pareto", (3.0, 500.0), 0.35),
        ("beta", (2.0, 5.0), 0.08),
        ("beta", (0.5, 0.8), 0.08),
        ("normal", (3.0, 2.0), 0.05),
        ("logistic", (-1.0, 0.7), 0.08),
        ("exponential", (4.0,), 0.05),
        ("poisson", (4.2,), 0.05),
    ]
    for name, params, tol in cases:
        data = _sample(name, params)
        fitted = getattr(actuarial, name).fit(data)
        assert len(fitted) == len(params)
        for got, expected in zip(fitted, params):
            _close(got, expected, tol)


def test_fit_is_a_stationary_point_of_the_likelihood():
    # Perturbing either parameter must lower the log-likelihood.
    for name, params in (
        ("gamma", (2.5, 3.0)),
        ("weibull", (1.7, 4.0)),
        ("pareto", (3.0, 500.0)),
        ("beta", (2.0, 5.0)),
        ("logistic", (-1.0, 0.7)),
        ("lognormal", (0.8, 0.45)),
    ):
        data = _sample(name, params, n=4000)
        fitted = getattr(actuarial, name).fit(data)
        family = getattr(actuarial, name)
        best = -family(*fitted).nnlf(data)
        for index in range(len(fitted)):
            for factor in (0.98, 1.02):
                nudged = list(fitted)
                nudged[index] *= factor
                try:
                    other = -family(*nudged).nnlf(data)
                except ValueError:
                    continue
                assert other <= best + 1e-6, (
                    f"{name}: perturbing parameter {index} improved the fit"
                )


def test_uniform_fit_is_the_range():
    data = [0.4, -1.0, 3.5, 2.2]
    assert actuarial.uniform.fit(data) == (-1.0, 3.5)


def test_negative_binomial_methods_agree_roughly():
    data = _sample("negative_binomial", (6.0, 0.35), n=30000)
    mle = actuarial.negative_binomial.fit(data)
    moments = actuarial.negative_binomial.fit(data, method="moments")
    _close(mle[0], 6.0, 0.15)
    _close(mle[1], 0.35, 0.1)
    _close(moments[0], mle[0], 0.15)
    _close(moments[1], mle[1], 0.1)


def test_negative_binomial_rejects_underdispersed_data():
    data = [3, 3, 3, 4, 3, 3, 4, 3]
    try:
        actuarial.negative_binomial.fit(data)
    except ValueError as exc:
        assert "over-dispersed" in str(exc)
        return
    raise AssertionError("expected ValueError")


def test_fit_rejects_out_of_support_data():
    for name, data in (
        ("lognormal", [1.0, -2.0]),
        ("gamma", [1.0, 0.0]),
        ("weibull", [1.0, -1.0]),
        ("beta", [0.5, 1.5]),
        ("poisson", [1, -1]),
    ):
        try:
            getattr(actuarial, name).fit(data)
        except ValueError:
            continue
        raise AssertionError(f"{name}: expected ValueError")


def test_fit_accepts_any_iterable():
    data = _sample("exponential", (2.0,), n=500)
    from_list = actuarial.exponential.fit(list(data))
    from_tuple = actuarial.exponential.fit(tuple(data))
    from_generator = actuarial.exponential.fit(v for v in data)
    assert from_list == from_tuple == from_generator


def test_round_trip_through_the_public_api():
    # The documented workflow: fit, rebuild, and check the fit is close.
    data = _sample("gamma", (2.0, 1500.0), n=8000)
    alpha, theta = actuarial.gamma.fit(data)
    fitted = actuarial.gamma(alpha, theta)
    statistic = fitted.ks_test(data)
    assert statistic < 0.02
    _, p_value = fitted.kstest(data)
    assert p_value > 0.01
