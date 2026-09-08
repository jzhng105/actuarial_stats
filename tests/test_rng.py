"""Sampling: reproducibility, stream structure and distributional sanity."""

import math

import actstats
from actstats import actuarial
from actstats.rng import RandomState

NAMES = [
    ("lognormal", (0.5, 0.2)),
    ("gamma", (2.5, 3.0)),
    ("gamma", (0.3, 4.0)),
    ("weibull", (1.5, 2.0)),
    ("pareto", (3.5, 100.0)),
    ("beta", (2.0, 5.0)),
    ("beta", (0.3, 0.4)),
    ("normal", (1.0, 2.0)),
    ("logistic", (0.5, 1.5)),
    ("exponential", (3.0,)),
    ("uniform", (-1.0, 2.0)),
    ("poisson", (0.4,)),
    ("poisson", (4.0,)),
    ("poisson", (5000.0,)),
    ("negative_binomial", (5.0, 0.4)),
    ("negative_binomial", (0.6, 0.25)),
]


def test_seeding_is_reproducible():
    for name, params in NAMES:
        dist = getattr(actuarial, name)(*params)
        actstats.seed(1234)
        first = list(dist.rvs(size=25))
        actstats.seed(1234)
        second = list(dist.rvs(size=25))
        assert first == second, name


def test_batched_and_sequential_draws_agree():
    # A batch must consume the same uniforms as the equivalent single draws,
    # so vectorising a simulation cannot change its results.
    for name, params in NAMES:
        dist = getattr(actuarial, name)(*params)
        actstats.seed(99)
        sequential = [dist.rvs() for _ in range(40)]
        actstats.seed(99)
        batched = list(dist.rvs(size=40))
        assert sequential == batched, name


def test_independent_random_states():
    a = RandomState(7)
    b = RandomState(7)
    dist = actuarial.lognormal(0.0, 1.0)
    assert list(dist.rvs(size=10, random_state=a)) == list(
        dist.rvs(size=10, random_state=b)
    )
    # A per-distribution stream is untouched by the module level one.
    own = RandomState(3)
    frozen = actuarial.gamma(2.0, 1.0, random_state=own)
    actstats.seed(0)
    first = list(frozen.rvs(size=5))
    own.seed(3)
    actstats.seed(12345)
    assert list(frozen.rvs(size=5)) == first


def test_random_state_round_trip():
    rng = RandomState(11)
    rng.standard_normal()  # leave a cached spare in place
    state = rng.get_state()
    expected = [rng.standard_normal() for _ in range(5)]
    rng.set_state(state)
    assert [rng.standard_normal() for _ in range(5)] == expected


def test_sample_moments_are_close_to_theory():
    for name, params in NAMES:
        dist = getattr(actuarial, name)(*params)
        mean = dist.mean()
        var = dist.var()
        if not math.isfinite(mean) or not math.isfinite(var) or var == 0.0:
            continue
        actstats.seed(20240908)
        n = 40000
        sample = dist.rvs(size=n)
        # Five standard errors of the sample mean, plus a floor for heavy tails.
        tolerance = 5.0 * math.sqrt(var / n) + 1e-9
        assert abs(sample.mean() - mean) < tolerance + 0.05 * abs(mean), name
        assert 0.7 < sample.var() / var < 1.4, name


def test_rvs_shapes():
    dist = actuarial.poisson(3.0)
    assert isinstance(dist.rvs(), int)
    assert len(dist.rvs(size=0)) == 0
    assert len(dist.rvs(size=7)) == 7


def test_discrete_draws_are_in_support():
    for name, params in (("poisson", (2.0,)), ("negative_binomial", (3.0, 0.5))):
        dist = getattr(actuarial, name)(*params)
        actstats.seed(5)
        for value in dist.rvs(size=2000):
            assert isinstance(value, int) and value >= 0


def test_continuous_draws_are_in_support():
    for name, params in NAMES:
        dist = getattr(actuarial, name)(*params)
        low, high = dist.support()
        actstats.seed(8)
        for value in dist.rvs(size=500):
            assert low <= value <= high, (name, value)


def test_generator_helpers():
    rng = RandomState(2)
    assert 0.0 <= rng.uniform(0.0, 1.0) <= 1.0
    assert rng.exponential(2.0) >= 0.0
    assert rng.standard_gamma(0.3) > 0.0
    assert 0.0 <= rng.beta(0.5, 0.5) <= 1.0
    assert rng.poisson(0.0) == 0
    assert rng.negative_binomial(2.0, 1.0) == 0
    assert rng.geometric(1.0) == 1
    assert rng.pareto(2.0, 5.0) >= 5.0
