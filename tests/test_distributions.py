"""Distribution functions, checked against values SciPy produced."""

import math

from actstats import actuarial
from actstats.distributions import Gamma, LogNormal, Pareto, Poisson

from reference_values import REFERENCE

DISCRETE = {"poisson", "negative_binomial"}


def _close(got, expected, rtol=1e-10, atol=0.0):
    if math.isinf(expected):
        assert math.isinf(got) and (got > 0) == (expected > 0)
        return
    assert abs(got - expected) <= max(rtol * abs(expected), atol), (
        f"{got!r} != {expected!r}"
    )


def test_reference_values():
    for (name, params), ref in REFERENCE.items():
        dist = getattr(actuarial, name)(*params)
        mass = dist.pmf if name in DISCRETE else dist.pdf
        for x, expected in zip(ref["x"], ref["pdf"]):
            _close(mass(x), expected, 1e-10, atol=1e-300)
        for x, expected in zip(ref["x"], ref["cdf"]):
            _close(dist.cdf(x), expected, 1e-10, atol=1e-300)
        for x, expected in zip(ref["x"], ref["sf"]):
            _close(dist.sf(x), expected, 1e-10, atol=1e-300)
        for q, expected in zip(ref["q"], ref["ppf"]):
            _close(dist.ppf(q), expected, 1e-9, atol=1e-300)
        for q, expected in zip(ref["q"], ref["isf"]):
            _close(dist.isf(q), expected, 1e-9, atol=1e-300)
        _close(dist.mean(), ref["mean"], 1e-12)
        _close(dist.var(), ref["var"], 1e-12)
        _close(dist.median(), ref["median"], 1e-9)


def test_log_variants_match_logs():
    for (name, params), ref in REFERENCE.items():
        dist = getattr(actuarial, name)(*params)
        for x in ref["x"]:
            for direct, logged in (
                (dist.pdf(x), dist.logpdf(x)),
                (dist.cdf(x), dist.logcdf(x)),
                (dist.sf(x), dist.logsf(x)),
            ):
                if direct > 1e-290:
                    _close(logged, math.log(direct), 1e-9)
                else:
                    assert logged < -600.0


def test_cdf_sf_complement():
    for (name, params), ref in REFERENCE.items():
        dist = getattr(actuarial, name)(*params)
        for x in ref["x"]:
            _close(dist.cdf(x) + dist.sf(x), 1.0, 1e-12)


def test_ppf_cdf_roundtrip():
    for (name, params), ref in REFERENCE.items():
        if name in DISCRETE:
            continue
        dist = getattr(actuarial, name)(*params)
        for q in (1e-8, 0.001, 0.1, 0.5, 0.9, 0.999, 1 - 1e-8):
            _close(dist.cdf(dist.ppf(q)), q, 1e-8)
            _close(dist.sf(dist.isf(q)), q, 1e-8)


def test_discrete_ppf_definition():
    # ppf(q) is the smallest k with cdf(k) >= q.
    for name, params in (("poisson", (4.0,)), ("negative_binomial", (5.0, 0.4))):
        dist = getattr(actuarial, name)(*params)
        for q in (0.001, 0.05, 0.2, 0.5, 0.9, 0.99):
            k = dist.ppf(q)
            assert dist.cdf(k) >= q - 1e-15
            assert k == 0 or dist.cdf(k - 1) < q


def test_moments_match_simulation_free_identities():
    # var == E[X^2] - E[X]^2 wherever raw moments are available.
    for name, params in (
        ("lognormal", (0.5, 0.2)),
        ("gamma", (2.5, 3.0)),
        ("weibull", (1.5, 2.0)),
        ("pareto", (4.0, 100.0)),
        ("beta", (2.0, 5.0)),
        ("exponential", (3.0,)),
        ("uniform", (-1.0, 2.0)),
        ("normal", (1.0, 2.0)),
    ):
        dist = getattr(actuarial, name)(*params)
        m1, m2 = dist.moment(1), dist.moment(2)
        _close(m1, dist.mean(), 1e-11)
        _close(m2 - m1 * m1, dist.var(), 1e-9)


def test_pdf_integrates_to_the_cdf():
    # Simpson's rule over the middle 98% of each continuous distribution.
    for name, params in (
        ("lognormal", (0.5, 0.2)),
        ("gamma", (2.5, 3.0)),
        ("weibull", (1.5, 2.0)),
        ("pareto", (2.5, 100.0)),
        ("beta", (2.0, 5.0)),
        ("normal", (1.0, 2.0)),
        ("logistic", (0.5, 1.5)),
        ("exponential", (3.0,)),
    ):
        dist = getattr(actuarial, name)(*params)
        lo, hi = dist.ppf(0.01), dist.ppf(0.99)
        n = 400
        step = (hi - lo) / n
        total = dist.pdf(lo) + dist.pdf(hi)
        for i in range(1, n):
            total += (4.0 if i % 2 else 2.0) * dist.pdf(lo + i * step)
        integral = total * step / 3.0
        _close(integral, 0.98, 1e-5, atol=1e-5)


def test_pmf_sums_to_one():
    for name, params in (("poisson", (4.0,)), ("negative_binomial", (5.0, 0.4))):
        dist = getattr(actuarial, name)(*params)
        total = math.fsum(dist.pmf(k) for k in range(0, 400))
        _close(total, 1.0, 1e-12)


def test_vectorised_calls_return_samples():
    dist = actuarial.gamma(2.0, 3.0)
    values = dist.cdf([1.0, 2.0, 3.0])
    assert list(values) == [dist.cdf(1.0), dist.cdf(2.0), dist.cdf(3.0)]
    assert values.mean() > 0.0


def test_interval_and_stats():
    dist = actuarial.normal(0.0, 1.0)
    lo, hi = dist.interval(0.95)
    _close(lo, -1.959963984540054, 1e-10)
    _close(hi, 1.959963984540054, 1e-10)
    mean, var = dist.stats()
    assert (mean, var) == (0.0, 1.0)
    assert dist.stats("s") == 0.0


def test_tail_precision_beats_naive_complement():
    # sf() must not lose precision where 1 - cdf() would collapse to zero.
    dist = actuarial.normal(0.0, 1.0)
    assert dist.cdf(35.0) == 1.0  # 1 - cdf() would be exactly zero here
    assert 0.0 < dist.sf(35.0) < 1e-260
    _close(dist.sf(35.0), dist.cdf(-35.0), 1e-12)
    assert dist.isf(1e-300) > 37.0


def test_parameter_validation():
    for factory in (
        lambda: actuarial.lognormal(0.0, -1.0),
        lambda: actuarial.gamma(0.0, 1.0),
        lambda: actuarial.weibull(1.0, 0.0),
        lambda: actuarial.pareto(-1.0, 1.0),
        lambda: actuarial.beta(1.0, 0.0),
        lambda: actuarial.poisson(-1.0),
        lambda: actuarial.negative_binomial(1.0, 1.5),
        lambda: actuarial.uniform(1.0, 0.0),
    ):
        try:
            factory()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_quantile_range_is_checked():
    try:
        actuarial.gamma(2.0, 1.0).ppf(1.5)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_support_and_repr():
    assert LogNormal(0.0, 1.0).support() == (0.0, math.inf)
    assert Gamma(1.0, 1.0).support() == (0.0, math.inf)
    assert repr(Pareto(2.0, 3.0)) == "Pareto(alpha=2.0, beta=3.0)"
    assert Poisson(2.0) == Poisson(2.0)


def test_keyword_parameters():
    assert LogNormal(mu=0.5, sigma=0.2).params == (0.5, 0.2)
    assert Gamma(2.0, theta=4.0).params == (2.0, 4.0)


def test_nonhomogeneous_poisson():
    nhpp = actuarial.nonhomogeneous_poisson(10.0, 0.25, 0.0, 1.0)
    # Expected count over the full horizon: lambda0 * T when the seasonal term
    # integrates away over a whole cycle.
    _close(nhpp.mean(), 10.0, 1e-12)
    assert nhpp.rate(0.25) > nhpp.rate(0.75)
    events = nhpp.rvs(n_events=10)
    assert len(events) == 10
    assert events == sorted(events)
    assert all(0.0 <= t <= 1.0 for t in events)
    paths = nhpp.rvs(size=3)
    assert len(paths) == 3
