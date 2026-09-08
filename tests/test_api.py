"""The public ``actuarial`` surface, including the pre-existing call styles."""

import math

import actstats
from actstats import Sample, actuarial


def test_frozen_and_unfrozen_agree():
    assert actuarial.lognormal.ppf(0.5, 2.0, 1.0) == actuarial.lognormal(2.0, 1.0).ppf(0.5)
    assert actuarial.poisson.pmf(3, 10.0) == actuarial.poisson(10.0).pmf(3)
    assert actuarial.lognormal.pdf(5.0, 2.0, 1.0) == actuarial.lognormal(2.0, 1.0).pdf(5.0)
    assert actuarial.lognormal.logpdf(5.0, 2.0, 1.0) == actuarial.lognormal(2.0, 1.0).logpdf(5.0)
    assert actuarial.gamma.cdf(4.0, 2.0, 3.0) == actuarial.gamma(2.0, 3.0).cdf(4.0)


def test_unfrozen_rvs_takes_parameters_first():
    actstats.seed(3)
    unfrozen = list(actuarial.lognormal.rvs(0.5, 0.2, 100))
    actstats.seed(3)
    frozen = list(actuarial.lognormal(0.5, 0.2).rvs(size=100))
    assert unfrozen == frozen


def test_defaults_when_no_parameters_given():
    assert actuarial.normal.mean() == 0.0
    assert actuarial.exponential.mean() == 1.0
    assert actuarial.uniform.mean() == 0.5


def test_np_rvs_is_an_alias_of_rvs():
    actstats.seed(11)
    legacy = list(actuarial.gamma(2.0, 3.0).np_rvs(size=50))
    actstats.seed(11)
    current = list(actuarial.gamma(2.0, 3.0).rvs(size=50))
    assert legacy == current


def test_samples_behave_like_arrays():
    actstats.seed(4)
    sample = actuarial.normal(5.0, 2.0).rvs(size=5000)
    assert isinstance(sample, Sample)
    assert abs(sample.mean() - 5.0) < 0.25
    assert abs(sample.std() - 2.0) < 0.25
    assert sample.min() <= sample.quantile(0.5) <= sample.max()
    doubled = sample * 2
    assert len(doubled) == len(sample)
    assert abs(doubled.mean() - 2 * sample.mean()) < 1e-9


def test_ks_test_and_kstest():
    actstats.seed(6)
    dist = actuarial.lognormal(0.5, 0.2)
    sample = dist.rvs(size=4000)
    statistic = dist.ks_test(sample)
    assert 0.0 < statistic < 0.05
    again, p_value = dist.kstest(sample)
    assert again == statistic
    assert 0.0 <= p_value <= 1.0
    # A badly mis-specified model should be rejected.
    wrong = actuarial.lognormal(3.0, 1.0)
    assert wrong.kstest(sample)[1] < 1e-6


def test_unknown_distribution_raises():
    try:
        actuarial.does_not_exist
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError")
    try:
        actstats.ActuarialDistribution("nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_repr_and_introspection():
    assert "lognormal" in repr(actuarial.lognormal)
    assert "unfrozen" in repr(actuarial.lognormal)
    frozen = actuarial.lognormal(0.5, 0.2)
    assert frozen.params == (0.5, 0.2)
    assert frozen.param_names == ("mu", "sigma")
    assert "gamma" in dir(actuarial)


def test_fit_from_the_namespace_and_from_a_frozen_instance():
    actstats.seed(2)
    data = actuarial.exponential(4.0).rvs(size=5000)
    assert actuarial.exponential.fit(data) == actuarial.exponential(1.0).fit(data)


def test_nonhomogeneous_poisson_legacy_signature():
    nhpp = actuarial.nonhomogeneous_poisson(10.0, 0.25, 0.0, 1.0)
    events = nhpp.rvs(size=2, n_events=10)
    assert len(events) == 2 and all(len(path) == 10 for path in events)
    legacy = nhpp.np_rvs(size=2, n_events=10)
    assert len(legacy) == 2
    dates = [actstats.fraction_to_date_full(t) for t in events[1]]
    assert len(dates) == 10


def test_module_exports():
    for name in (
        "actuarial",
        "LogNormal",
        "Gamma",
        "Poisson",
        "Sample",
        "seed",
        "kstest",
        "fraction_to_date_full",
    ):
        assert hasattr(actstats, name), name
