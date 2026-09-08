"""Goodness-of-fit statistics."""

from __future__ import annotations

import math

__all__ = ["ks_statistic", "kolmogorov_sf", "kstest"]


def ks_statistic(data, cdf):
    """One-sample Kolmogorov-Smirnov statistic ``D``.

    ``cdf`` is called once per observation, so it may be any callable -- a
    fitted distribution's ``cdf``, or a closure over one.
    """
    values = sorted(float(v) for v in data)
    n = len(values)
    if n == 0:
        raise ValueError("no observations supplied")
    inv_n = 1.0 / n
    d = 0.0
    for i, value in enumerate(values):
        f = cdf(value)
        above = (i + 1) * inv_n - f  # step above the empirical CDF
        below = f - i * inv_n  # step below it
        if above > d:
            d = above
        if below > d:
            d = below
    return d


def kolmogorov_sf(x):
    """Survival function of the Kolmogorov distribution, ``Q(x)``.

    ``Q(x) = 2 * sum_{k>=1} (-1)^(k-1) exp(-2 k^2 x^2)``, the limiting
    distribution of ``sqrt(n) * D``.
    """
    if x <= 0.0:
        return 1.0
    if x < 1.18:
        # The series above converges slowly for small x; use the theta-function
        # transform, which converges instantly there.
        y = math.exp(-1.233700550136170 / (x * x))  # pi^2 / 8
        w = math.sqrt(2.0 * math.pi) / x
        total = y + y ** 9 + y ** 25 + y ** 49
        return 1.0 - w * total
    total = 0.0
    for k in range(1, 101):
        term = math.exp(-2.0 * k * k * x * x)
        total += term if k % 2 else -term
        if term < 1e-18:
            break
    value = 2.0 * total
    return min(max(value, 0.0), 1.0)


def kstest(data, cdf):
    """One-sample KS test: returns ``(statistic, p_value)``.

    The p-value uses the asymptotic Kolmogorov distribution with Stephens'
    small-sample correction, which is accurate for ``n`` of roughly 20 or more.
    """
    values = list(data)
    n = len(values)
    d = ks_statistic(values, cdf)
    root_n = math.sqrt(n)
    return d, kolmogorov_sf((root_n + 0.12 + 0.11 / root_n) * d)
