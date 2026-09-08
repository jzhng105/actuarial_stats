"""Regenerate ``tests/reference_values.py`` from SciPy.

The test suite compares the pure-Python implementations against values SciPy
produced once, so the tests themselves need no dependencies.  Run this script
in an environment that *does* have SciPy when a new distribution or reference
point is added::

    python scripts/regenerate_reference.py > tests/reference_values.py
"""

from __future__ import annotations

import math

import scipy.special as ss
import scipy.stats as st

CASES = [
    ("lognormal", (0.5, 0.2), st.lognorm(0.2, 0, math.exp(0.5))),
    ("gamma", (2.5, 3.0), st.gamma(2.5, 0, 3.0)),
    ("weibull", (1.5, 2.0), st.weibull_min(1.5, 0, 2.0)),
    ("pareto", (2.5, 100.0), st.lomax(2.5, 0, 100.0)),
    ("beta", (2.0, 5.0), st.beta(2.0, 5.0)),
    ("normal", (1.0, 2.0), st.norm(1.0, 2.0)),
    ("logistic", (0.5, 1.5), st.logistic(0.5, 1.5)),
    ("exponential", (3.0,), st.expon(0, 3.0)),
    ("uniform", (-1.0, 2.0), st.uniform(-1.0, 3.0)),
    ("poisson", (4.0,), st.poisson(4.0)),
    ("negative_binomial", (5.0, 0.4), st.nbinom(5.0, 0.4)),
]
CONTINUOUS_X = [0.05, 0.5, 1.5, 4.0, 25.0]
DISCRETE_X = [0, 1, 3, 8, 15]
QUANTILES = [0.001, 0.05, 0.5, 0.95, 0.999]
DISCRETE = {"poisson", "negative_binomial"}


def main():
    print('"""Reference values for the test suite.')
    print()
    print(f"Generated from SciPy {st.__name__ and __import__('scipy').__version__} by")
    print("``scripts/regenerate_reference.py`` so that the tests can check the")
    print("pure-Python implementations to full double precision without SciPy being")
    print("installed.")
    print('"""')
    print()
    print("#: ``(name, params) -> {method: values}`` computed with SciPy.")
    print("REFERENCE = {")
    for name, params, dist in CASES:
        discrete = name in DISCRETE
        xs = DISCRETE_X if discrete else CONTINUOUS_X
        mass = dist.pmf if discrete else dist.pdf
        print(f"    ({name!r}, {params!r}): {{")
        print(f"        'x': {xs!r},")
        print(f"        'pdf': {[float(mass(x)) for x in xs]!r},")
        print(f"        'cdf': {[float(dist.cdf(x)) for x in xs]!r},")
        print(f"        'sf': {[float(dist.sf(x)) for x in xs]!r},")
        print(f"        'q': {QUANTILES!r},")
        print(f"        'ppf': {[float(dist.ppf(q)) for q in QUANTILES]!r},")
        print(f"        'isf': {[float(dist.isf(q)) for q in QUANTILES]!r},")
        print(f"        'mean': {float(dist.mean())!r},")
        print(f"        'var': {float(dist.var())!r},")
        print(f"        'median': {float(dist.median())!r},")
        print("    },")
    print("}")
    print()
    print("#: ``function -> [(args..., expected)]`` for actstats.special.")
    print("SPECIAL = {")
    gammainc = [(0.5, 0.3), (1.0, 1.0), (2.5, 4.0), (30.0, 25.0), (0.05, 1e-3), (500.0, 520.0)]
    print(f"    'gammainc': {[(a, x, float(ss.gammainc(a, x))) for a, x in gammainc]!r},")
    gammaincc = [(0.5, 0.3), (2.5, 20.0), (30.0, 80.0), (3.0, 0.01)]
    print(f"    'gammaincc': {[(a, x, float(ss.gammaincc(a, x))) for a, x in gammaincc]!r},")
    betainc = [(0.5, 0.5, 0.3), (2.0, 3.0, 0.7), (10.0, 1.0, 0.9), (0.1, 20.0, 0.01)]
    print(f"    'betainc': {[(a, b, x, float(ss.betainc(a, b, x))) for a, b, x in betainc]!r},")
    print(f"    'ndtr': {[(x, float(ss.ndtr(x))) for x in [-6.0, -1.96, 0.0, 0.5, 3.0, 8.0]]!r},")
    ps = [1e-12, 0.001, 0.025, 0.5, 0.975, 1 - 1e-12]
    print(f"    'ndtri': {[(p, float(ss.ndtri(p))) for p in ps]!r},")
    xs = [0.1, 0.5, 1.0, 2.5, 10.0, 1000.0, -1.5]
    print(f"    'digamma': {[(x, float(ss.digamma(x))) for x in xs]!r},")
    xs = [0.1, 0.5, 1.0, 2.5, 10.0, 1000.0]
    print(f"    'trigamma': {[(x, float(ss.polygamma(1, x))) for x in xs]!r},")
    pairs = [(0.5, 0.3), (2.5, 0.95), (30.0, 0.001), (0.05, 1e-8)]
    print(f"    'gammaincinv': {[(a, p, float(ss.gammaincinv(a, p))) for a, p in pairs]!r},")
    pairs = [(0.5, 0.3), (2.5, 0.001), (30.0, 1e-9)]
    print(f"    'gammainccinv': {[(a, q, float(ss.gammainccinv(a, q))) for a, q in pairs]!r},")
    triples = [(0.5, 0.5, 0.3), (2.0, 3.0, 0.99), (10.0, 1.0, 1e-6)]
    print(f"    'betaincinv': {[(a, b, p, float(ss.betaincinv(a, b, p))) for a, b, p in triples]!r},")
    print("}")


if __name__ == "__main__":
    main()
