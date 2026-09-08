"""Worked examples for every distribution actstats supports.

Run it with ``python examples/examples.py``.  Nothing here imports NumPy or
SciPy: the theoretical moments are written out from the actuarial
parameterisation so the output doubles as a check on the library.
"""

import math

import actstats
from actstats import actuarial

SIZE = 100_000
actstats.seed(20240908)


def gamma_function(x):
    """``Gamma(x)`` from the standard library."""
    return math.gamma(x)


# name, parameters, theoretical mean, theoretical variance
CASES = [
    (
        "lognormal", (0.5, 0.2),
        lambda mu, sigma: math.exp(mu + sigma ** 2 / 2),
        lambda mu, sigma: (math.exp(sigma ** 2) - 1) * math.exp(2 * mu + sigma ** 2),
    ),
    (
        "gamma", (1.0, 2.0),
        lambda alpha, theta: alpha * theta,
        lambda alpha, theta: alpha * theta ** 2,
    ),
    (
        "weibull", (1.5, 1.0),
        lambda delta, beta: beta * gamma_function(1 + 1 / delta),
        lambda delta, beta: beta ** 2
        * (gamma_function(1 + 2 / delta) - gamma_function(1 + 1 / delta) ** 2),
    ),
    (
        "pareto", (5.0, 1.0),
        lambda alpha, beta: beta / (alpha - 1) if alpha > 1 else math.inf,
        lambda alpha, beta: (beta ** 2 * alpha) / ((alpha - 1) ** 2 * (alpha - 2))
        if alpha > 2
        else math.inf,
    ),
    (
        "beta", (1.0, 2.0),
        lambda alpha, beta: alpha / (alpha + beta),
        lambda alpha, beta: (alpha * beta)
        / ((alpha + beta) ** 2 * (alpha + beta + 1)),
    ),
    ("poisson", (5.0,), lambda lam: lam, lambda lam: lam),
    (
        "negative_binomial", (5.0, 0.5),
        lambda r, p: r * (1 - p) / p,
        lambda r, p: r * (1 - p) / p ** 2,
    ),
    ("normal", (0.0, 1.0), lambda mu, sigma: mu, lambda mu, sigma: sigma ** 2),
    (
        "logistic", (0.0, 1.0),
        lambda mu, s: mu,
        lambda mu, s: (math.pi ** 2 / 3) * s ** 2,
    ),
    ("exponential", (2.0,), lambda beta: beta, lambda beta: beta ** 2),
    (
        "uniform", (0.0, 1.0),
        lambda a, b: (a + b) / 2,
        lambda a, b: (b - a) ** 2 / 12,
    ),
]


def distribution_tour():
    """Simulate each distribution, then fit it back to its own sample."""
    print(f"Simulating {SIZE:,} variates per distribution\n")
    header = f"{'distribution':22s}{'mean (theory/sample)':>30s}{'variance (theory/sample)':>34s}"
    print(header)
    print("-" * len(header))
    for name, params, theoretical_mean, theoretical_var in CASES:
        dist = getattr(actuarial, name)(*params)
        sample = dist.rvs(size=SIZE)

        mean, variance = theoretical_mean(*params), theoretical_var(*params)
        # The library's own closed forms must agree with the formulas above.
        assert math.isclose(dist.mean(), mean, rel_tol=1e-9)
        assert math.isclose(dist.var(), variance, rel_tol=1e-9)

        label = f"{name}{params}"
        print(
            f"{label:22s}{mean:14.4f} /{sample.mean():13.4f}"
            f"{variance:16.4f} /{sample.var():15.4f}"
        )

    print("\nFitting each distribution back to its own sample")
    print(f"{'distribution':22s}{'true parameters':>28s}{'fitted':>30s}")
    print("-" * 80)
    for name, params, _, _ in CASES:
        sample = getattr(actuarial, name)(*params).rvs(size=SIZE)
        fitted = getattr(actuarial, name).fit(sample)
        true_text = ", ".join(f"{v:g}" for v in params)
        fitted_text = ", ".join(f"{v:.4f}" for v in fitted)
        print(f"{name:22s}{'(' + true_text + ')':>28s}{'(' + fitted_text + ')':>30s}")


def quantiles_and_goodness_of_fit():
    """Tail quantiles and a Kolmogorov-Smirnov check, the way pricing work uses them."""
    print("\nSeverity quantiles for lognormal(mu=9, sigma=1.3)")
    severity = actuarial.lognormal(9.0, 1.3)
    for level in (0.5, 0.9, 0.99, 0.995, 0.999):
        print(f"  {level:>6.3%} VaR  {severity.ppf(level):15,.0f}")
    # isf() is the accurate way to ask for a far tail: ppf(1 - 1e-9) has already
    # lost most of its significant digits by the time it is called.
    print(f"  1-in-1e9 loss {severity.isf(1e-9):15,.0f}")

    losses = severity.rvs(size=20_000)
    mu, sigma = actuarial.lognormal.fit(losses)
    fitted = actuarial.lognormal(mu, sigma)
    statistic, p_value = fitted.kstest(losses)
    print(f"\n  fitted mu={mu:.4f} sigma={sigma:.4f}")
    print(f"  KS statistic {statistic:.5f}, p-value {p_value:.3f}")


def collective_risk_model():
    """Aggregate losses from a frequency-severity model, one trial at a time."""
    print("\nCollective risk model: Poisson(3.7) claims, lognormal(9, 1.3) severity")
    frequency = actuarial.poisson(3.7)
    severity = actuarial.lognormal(9.0, 1.3)
    trials = 50_000

    totals = actstats.Sample(
        sum(severity.rvs(size=n)) if (n := frequency.rvs()) else 0.0
        for _ in range(trials)
    )
    print(f"  mean aggregate loss   {totals.mean():15,.0f}")
    print(f"  standard deviation    {totals.std():15,.0f}")
    for level in (0.9, 0.99, 0.999):
        print(f"  {level:.1%} aggregate VaR   {totals.quantile(level):15,.0f}")
    # TVaR: the average of the worst 1% of trials.
    ordered = sorted(totals)
    tail = ordered[int(0.99 * trials):]
    print(f"  99% TVaR              {sum(tail) / len(tail):15,.0f}")


def seasonal_claim_arrivals():
    """A nonhomogeneous Poisson process with a seasonal claim rate."""
    print("\nSeasonal claim arrivals: lambda(t) = 10 * (1 + 0.25 sin(2 pi t))")
    nhpp = actuarial.nonhomogeneous_poisson(10.0, 0.25, 0.0, 1.0)
    print(f"  expected events over the year: {nhpp.mean():.2f}")

    events = nhpp.rvs()  # thinning, until the horizon
    print(f"  one simulated year produced {len(events)} events")
    for time_fraction in events[:5]:
        print(f"    {actstats.fraction_to_date_full(time_fraction)}")

    # Or condition on a known number of events and place them by intensity.
    conditioned = nhpp.rvs(size=2, n_events=10)
    print(f"  two conditioned paths of 10 events: {len(conditioned)} paths")
    first_half = sum(1 for t in nhpp.rvs(n_events=10_000) if t < 0.5)
    print(f"  share of events in the first half-year: {first_half / 10_000:.3f}")


def reproducibility():
    """Seeding makes a run repeatable, and batching never changes it."""
    print("\nReproducibility")
    dist = actuarial.gamma(2.0, 3.0)
    actstats.seed(7)
    one_at_a_time = [dist.rvs() for _ in range(5)]
    actstats.seed(7)
    in_one_batch = list(dist.rvs(size=5))
    print(f"  sequential == batched: {one_at_a_time == in_one_batch}")

    # An independent stream, useful when a model needs reproducible sub-models.
    own_stream = actstats.RandomState(42)
    print(f"  private stream draw:  {dist.rvs(random_state=own_stream):.4f}")


if __name__ == "__main__":
    distribution_tour()
    quantiles_and_goodness_of_fit()
    collective_risk_model()
    seasonal_claim_arrivals()
    reproducibility()
