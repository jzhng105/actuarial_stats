"""Pure-Python special functions.

This module provides the handful of special functions the distributions in
:mod:`actstats` need (incomplete gamma, incomplete beta, the normal integral
and their inverses).  Everything here is implemented on top of the standard
library :mod:`math` module so that ``actstats`` has no third-party
dependencies.

The implementations follow the classical continued-fraction / series
expansions (Numerical Recipes, Cephes) and are accurate to close to double
precision over the ranges used by the distribution code.
"""

from __future__ import annotations

import math

__all__ = [
    "log_gamma",
    "log_beta",
    "log_binom",
    "digamma",
    "trigamma",
    "gammainc",
    "gammaincc",
    "gammaincinv",
    "gammainccinv",
    "betainc",
    "betaincinv",
    "betainccinv",
    "ndtr",
    "ndtri",
]

_EPS = 2.220446049250313e-16
_TINY = 1e-300
_LOG_MIN_SUBNORMAL = math.log(5e-324)
# Iteration caps for the series / continued fractions below.  They converge in
# a handful of terms for everyday parameters; the generous cap only matters for
# very large shape parameters, where convergence is O(sqrt(a)).
_MAX_ITER = 100000

log_gamma = math.lgamma


def log_beta(a: float, b: float) -> float:
    """Natural logarithm of the beta function ``B(a, b)``."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def log_binom(n: float, k: float) -> float:
    """Natural logarithm of the binomial coefficient ``C(n, k)``."""
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


# ---------------------------------------------------------------------------
# Polygamma
# ---------------------------------------------------------------------------

def digamma(x: float) -> float:
    """Digamma function ``psi(x) = d/dx log Gamma(x)``."""
    if x <= 0.0 and x == math.floor(x):
        raise ValueError("digamma is undefined at non-positive integers")
    if x < 0.0:
        # Reflection formula: psi(1 - x) - psi(x) = pi * cot(pi * x)
        return digamma(1.0 - x) - math.pi / math.tan(math.pi * x)

    result = 0.0
    while x < 12.0:
        result -= 1.0 / x
        x += 1.0

    inv = 1.0 / x
    inv2 = inv * inv
    result += math.log(x) - 0.5 * inv
    result -= inv2 * (
        1.0 / 12.0
        - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 / 132.0)))
    )
    return result


def trigamma(x: float) -> float:
    """Trigamma function ``psi'(x)``."""
    if x <= 0.0 and x == math.floor(x):
        raise ValueError("trigamma is undefined at non-positive integers")
    if x < 0.0:
        # psi'(x) + psi'(1 - x) = pi^2 / sin^2(pi x)
        s = math.sin(math.pi * x)
        return (math.pi * math.pi) / (s * s) - trigamma(1.0 - x)

    result = 0.0
    while x < 12.0:
        result += 1.0 / (x * x)
        x += 1.0

    inv = 1.0 / x
    inv2 = inv * inv
    result += inv * (
        1.0
        + 0.5 * inv
        + inv2 * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 * (1.0 / 42.0 - inv2 / 30.0)))
    )
    return result



# ---------------------------------------------------------------------------
# Root finding shared by the quantile functions
# ---------------------------------------------------------------------------

def _bracket_mid(lo, hi, x):
    """Next bisection point for the bracket ``(lo, hi)`` around ``x``."""
    if not math.isfinite(hi):
        return max(2.0 * x, 2.0 * lo, 1e-300)
    if lo <= 0.0:
        if hi < 1e-4:
            # Arithmetic bisection needs ~1000 steps to reach the subnormal
            # range; bisecting the exponent instead gets there in ~10.
            return math.exp(0.5 * (math.log(hi) + _LOG_MIN_SUBNORMAL))
        return 0.5 * hi
    if hi > lo * 1e4:
        # Brackets spanning many orders of magnitude close far faster when the
        # exponent is bisected rather than the value.
        return math.sqrt(lo) * math.sqrt(hi)
    return 0.5 * (lo + hi)


def _safe_newton(cdf, log_pdf, target, x, lo, hi, decreasing=False, max_iter=120):
    """Invert a monotone CDF with Newton steps guarded by bisection.

    ``cdf`` and ``log_pdf`` describe the distribution, ``target`` is the
    probability to invert and ``(lo, hi)`` brackets the root.  Newton steps
    that would leave the bracket fall back first to a step in ``log x`` --
    which stays well scaled where the density over- or underflows -- and then
    to bisection, so the iteration always converges.  Set ``decreasing`` when
    ``cdf`` is a survival function.
    """
    sign = -1.0 if decreasing else 1.0
    for _ in range(max_iter):
        if not lo < x < hi:
            x = _bracket_mid(lo, hi, x)
            if not lo < x < hi:
                # The bracket has collapsed to neighbouring floats.
                return x
        err = cdf(x) - target
        if err == 0.0:
            return x
        if err * sign > 0.0:
            hi = x
        else:
            lo = x

        lp = log_pdf(x)
        new_x = x
        newton = False
        if -740.0 < lp < 709.0:
            pdf = math.exp(lp)
            if pdf > 0.0:
                newton = True
                new_x = x - sign * err / pdf
        if new_x == x and x > 0.0:
            # Newton in log-space: d(cdf)/d(log x) = x * pdf.  This stays well
            # scaled where the density itself over- or underflows.
            scaled = lp + math.log(x)
            if -740.0 < scaled < 709.0:
                step = sign * err / math.exp(scaled)
                if -700.0 < step < 700.0:
                    newton = True
                    new_x = x * math.exp(-step)

        if newton and new_x == x:
            # The correction is below the last representable digit.
            return x
        if newton and not math.isfinite(hi) and new_x > 16.0 * x > 0.0:
            # Do not let a vanishing density fling the iterate to infinity
            # while the upper end of the bracket is still open.
            new_x = 16.0 * x
        if not newton or not lo < new_x < hi:
            new_x = _bracket_mid(lo, hi, x)
            if new_x == x or not lo < new_x < hi:
                return x
        elif abs(new_x - x) <= 4.0 * _EPS * abs(x):
            return new_x
        x = new_x
    return x


# ---------------------------------------------------------------------------
# Incomplete gamma
# ---------------------------------------------------------------------------

def _gamma_series(a: float, x: float) -> float:
    """Series expansion for the regularized lower incomplete gamma P(a, x)."""
    ap = a
    total = 1.0 / a
    term = total
    for _ in range(_MAX_ITER):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a: float, x: float) -> float:
    """Continued fraction for the regularized upper incomplete gamma Q(a, x)."""
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b if b != 0.0 else 1.0 / _TINY
    h = d
    for i in range(1, _MAX_ITER + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammainc(a: float, x: float) -> float:
    """Regularized lower incomplete gamma ``P(a, x)``."""
    if a <= 0.0:
        raise ValueError("gammainc requires a > 0")
    if x < 0.0:
        raise ValueError("gammainc requires x >= 0")
    if x == 0.0:
        return 0.0
    if math.isinf(x):
        return 1.0
    if x < a + 1.0:
        return _gamma_series(a, x)
    return 1.0 - _gamma_cf(a, x)


def gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma ``Q(a, x) = 1 - P(a, x)``."""
    if a <= 0.0:
        raise ValueError("gammaincc requires a > 0")
    if x < 0.0:
        raise ValueError("gammaincc requires x >= 0")
    if x == 0.0:
        return 1.0
    if math.isinf(x):
        return 0.0
    if x < a + 1.0:
        return 1.0 - _gamma_series(a, x)
    return _gamma_cf(a, x)


def _gammainc_start(a: float, p: float) -> float:
    """Initial guess for the ``P(a, x) = p`` root."""
    if a > 1.0:
        z = ndtri(p)
        t = 1.0 - 1.0 / (9.0 * a) + z / math.sqrt(9.0 * a)
        x = a * t * t * t
        if x <= 0.0:
            x = 0.5 * a
        return x
    # For a <= 1 the leading small-x behaviour is P(a, x) ~ x^a / (a Gamma(a)),
    # which is an excellent start whenever p is small.
    log_x = (math.log(p) + math.lgamma(a + 1.0)) / a
    if log_x < -700.0:
        return math.exp(max(log_x, _LOG_MIN_SUBNORMAL))
    x = math.exp(log_x)
    return x if x > 0.0 else 5e-324


def gammaincinv(a: float, p: float) -> float:
    """Inverse of :func:`gammainc` with respect to ``x``."""
    if a <= 0.0:
        raise ValueError("gammaincinv requires a > 0")
    if not 0.0 <= p <= 1.0:
        raise ValueError("gammaincinv requires 0 <= p <= 1")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return math.inf
    if p > 0.5:
        # Solving the upper tail keeps full relative precision there.
        return gammainccinv(a, 1.0 - p)

    log_norm = math.lgamma(a)

    def _log_pdf(t):
        return (a - 1.0) * math.log(t) - t - log_norm

    return _safe_newton(
        lambda t: gammainc(a, t), _log_pdf, p, _gammainc_start(a, p), 0.0, math.inf
    )


def gammainccinv(a: float, q: float) -> float:
    """Inverse of :func:`gammaincc` with respect to ``x``.

    Preferred over ``gammaincinv(a, 1 - q)`` when ``q`` is small, because the
    upper tail keeps its relative precision.
    """
    if a <= 0.0:
        raise ValueError("gammainccinv requires a > 0")
    if not 0.0 <= q <= 1.0:
        raise ValueError("gammainccinv requires 0 <= q <= 1")
    if q == 0.0:
        return math.inf
    if q == 1.0:
        return 0.0
    if q > 0.5:
        return gammaincinv(a, 1.0 - q)

    log_norm = math.lgamma(a)

    def _log_pdf(t):
        return (a - 1.0) * math.log(t) - t - log_norm

    return _safe_newton(
        lambda t: gammaincc(a, t), _log_pdf, q, _gammainc_start(a, 1.0 - q),
        0.0, math.inf, decreasing=True,
    )


# ---------------------------------------------------------------------------
# Incomplete beta
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)``."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError("betainc requires a > 0 and b > 0")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betainc_start(a: float, b: float, p: float) -> float:
    """Initial guess for the ``I_x(a, b) = p`` root."""
    if a >= 1.0 and b >= 1.0:
        # Numerical Recipes / AS 109 normal-based start.
        pp = p if p < 0.5 else 1.0 - p
        t = math.sqrt(-2.0 * math.log(pp))
        x = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t
        if p < 0.5:
            x = -x
        al = (x * x - 3.0) / 6.0
        h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0))
        w = (x * math.sqrt(al + h) / h) - (
            1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)
        ) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h))
        try:
            x = a / (a + b * math.exp(2.0 * w))
        except OverflowError:
            x = 0.0
    else:
        # Leading small-x behaviour: I_x(a, b) ~ x^a / (a B(a, b)).
        log_x = (math.log(p) + math.log(a) + log_beta(a, b)) / a
        x = math.exp(log_x) if log_x > _LOG_MIN_SUBNORMAL else 5e-324
    if not 0.0 < x < 1.0 or x != x:
        x = 0.5
    return x


def betaincinv(a: float, b: float, p: float) -> float:
    """Inverse of :func:`betainc` with respect to ``x``."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError("betaincinv requires a > 0 and b > 0")
    if not 0.0 <= p <= 1.0:
        raise ValueError("betaincinv requires 0 <= p <= 1")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    if p > 0.5 and betainc(a, b, 0.5) < p:
        # The root lies above 1/2.  I_x(a, b) = 1 - I_{1-x}(b, a), and solving
        # for the reflected root keeps the iteration away from the flat region
        # of the CDF.  The median test matters for very small ``a``, where the
        # distribution piles up at zero and even p > 1/2 has a tiny root.
        return 1.0 - betaincinv(b, a, 1.0 - p)

    log_norm = log_beta(a, b)

    def _log_pdf(t):
        return (a - 1.0) * math.log(t) + (b - 1.0) * math.log1p(-t) - log_norm

    return _safe_newton(
        lambda t: betainc(a, b, t), _log_pdf, p, _betainc_start(a, b, p), 0.0, 1.0
    )


def betainccinv(a: float, b: float, q: float) -> float:
    """Value of ``x`` with ``1 - I_x(a, b) = q``.

    Preferred over ``betaincinv(a, b, 1 - q)`` for small ``q``.
    """
    if q == 0.0:
        return 1.0
    if q == 1.0:
        return 0.0
    if q > 0.5:
        return betaincinv(a, b, 1.0 - q)
    return 1.0 - betaincinv(b, a, q)


# ---------------------------------------------------------------------------
# Normal integral
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)


def ndtr(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * math.erfc(-x / _SQRT2)


_NDTRI_A = (
    -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
    1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
)
_NDTRI_B = (
    -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
    6.680131188771972e+01, -1.328068155288572e+01,
)
_NDTRI_C = (
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
    -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
)
_NDTRI_D = (
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
    3.754408661907416e+00,
)
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def ndtri(p: float) -> float:
    """Inverse of :func:`ndtr` (the standard normal quantile function).

    Uses Acklam's rational approximation refined by one Halley step, which
    brings the result to full double precision.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("ndtri requires 0 <= p <= 1")
    if p == 0.0:
        return -math.inf
    if p == 1.0:
        return math.inf

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_NDTRI_C[0] * q + _NDTRI_C[1]) * q + _NDTRI_C[2]) * q + _NDTRI_C[3]) * q
              + _NDTRI_C[4]) * q + _NDTRI_C[5]) / (
             (((_NDTRI_D[0] * q + _NDTRI_D[1]) * q + _NDTRI_D[2]) * q + _NDTRI_D[3]) * q + 1.0)
    elif p <= _P_HIGH:
        q = p - 0.5
        r = q * q
        x = (((((_NDTRI_A[0] * r + _NDTRI_A[1]) * r + _NDTRI_A[2]) * r + _NDTRI_A[3]) * r
              + _NDTRI_A[4]) * r + _NDTRI_A[5]) * q / (
             ((((_NDTRI_B[0] * r + _NDTRI_B[1]) * r + _NDTRI_B[2]) * r + _NDTRI_B[3]) * r
              + _NDTRI_B[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log1p(-p))
        x = -(((((_NDTRI_C[0] * q + _NDTRI_C[1]) * q + _NDTRI_C[2]) * q + _NDTRI_C[3]) * q
               + _NDTRI_C[4]) * q + _NDTRI_C[5]) / (
              (((_NDTRI_D[0] * q + _NDTRI_D[1]) * q + _NDTRI_D[2]) * q + _NDTRI_D[3]) * q + 1.0)

    # One Halley refinement step against the (exact) erfc.  Above the median
    # the comparison is made on the upper tail, where both sides stay small and
    # the subtraction keeps its significant digits.
    if p > 0.5:
        err = (1.0 - p) - 0.5 * math.erfc(x / _SQRT2)
    else:
        err = 0.5 * math.erfc(-x / _SQRT2) - p
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    if pdf > 0.0:
        u = err / pdf
        x -= u / (1.0 + 0.5 * x * u)
    return x
