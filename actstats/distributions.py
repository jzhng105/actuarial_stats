"""Probability distributions in actuarial parameterisation, in pure Python.

Each class here is self-contained: densities, distribution functions,
quantiles, moments, maximum-likelihood fitting and sampling are all written
against the standard library plus :mod:`actstats.special`.

The classes are parameterised the way actuaries write them (see the table in
the README), which is often *not* how SciPy writes them -- ``pareto`` here is
the two-parameter Pareto supported on ``[0, inf)`` (SciPy's ``lomax``), and
``gamma`` takes a scale rather than a rate.
"""

from __future__ import annotations

import math
from bisect import bisect_left

from .optimize import brentq, bracket_root, minimize_scalar
from .rng import _resolve
from .sample import Sample
from .special import (
    betainc,
    betaincinv,
    betainccinv,
    digamma,
    gammainc,
    gammaincc,
    gammaincinv,
    gammainccinv,
    log_beta,
    ndtr,
    ndtri,
    trigamma,
)

__all__ = [
    "Distribution",
    "LogNormal",
    "Gamma",
    "Weibull",
    "Pareto",
    "Beta",
    "Poisson",
    "NegativeBinomial",
    "Normal",
    "Logistic",
    "Exponential",
    "Uniform",
    "NonhomogeneousPoisson",
]

_INF = math.inf
_LOG = math.log
_EXP = math.exp
_SQRT = math.sqrt
_LOG_2PI = math.log(2.0 * math.pi)
_LOG_SQRT_2PI = 0.5 * _LOG_2PI

#: Sentinel meaning "the sampling table has not been built yet".
_UNBUILT = object()


def _as_float_list(data) -> list:
    """Coerce ``data`` to a list of floats, accepting any iterable."""
    if isinstance(data, (int, float)):
        return [float(data)]
    values = [float(v) for v in data]
    if not values:
        raise ValueError("no observations supplied")
    return values


class Distribution:
    """Base class: turns scalar kernels into the usual distribution API.

    Subclasses implement the scalar ``_pdf``/``_cdf``/``_ppf``/``_rvs1``
    kernels; everything public here vectorises over any iterable input and
    returns a :class:`~actstats.sample.Sample`.
    """

    #: Names of the actuarial parameters, in positional order.
    param_names: tuple = ()
    #: Default parameter values, in the same order.
    param_defaults: tuple = ()
    #: ``False`` for lattice distributions (Poisson, negative binomial).
    continuous: bool = True

    def __init__(self, *params, random_state=None, **kwargs):
        values = list(params)
        if len(values) > len(self.param_names):
            raise TypeError(
                f"{type(self).__name__} takes at most {len(self.param_names)} "
                f"parameters, got {len(values)}"
            )
        for index in range(len(values), len(self.param_names)):
            name = self.param_names[index]
            if name in kwargs:
                values.append(kwargs.pop(name))
            else:
                values.append(self.param_defaults[index])
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
        self.params = tuple(float(v) for v in values)
        self._random_state = random_state
        self._validate()
        self._prepare()

    # -- hooks ------------------------------------------------------------
    def _validate(self):
        """Raise :class:`ValueError` if the parameters are out of range."""

    def _prepare(self):
        """Cache anything derived from the parameters."""

    def _pdf(self, x):
        return _EXP(self._logpdf(x))

    def _logpdf(self, x):
        p = self._pdf(x)
        return _LOG(p) if p > 0.0 else -_INF

    def _cdf(self, x):
        raise NotImplementedError

    def _sf(self, x):
        return 1.0 - self._cdf(x)

    def _ppf(self, q):
        raise NotImplementedError

    def _isf(self, q):
        return self._ppf(1.0 - q)

    def _rvs1(self, rng):
        return self._ppf(rng.random())

    # -- introspection ----------------------------------------------------
    @property
    def name(self):
        """Lower-case name of the distribution."""
        return type(self).__name__.lower()

    def support(self):
        """``(lower, upper)`` bounds of the distribution's support."""
        return (0.0, _INF)

    def __repr__(self):
        args = ", ".join(
            f"{n}={v!r}" for n, v in zip(self.param_names, self.params)
        )
        return f"{type(self).__name__}({args})"

    def __eq__(self, other):
        return type(self) is type(other) and self.params == other.params

    def __hash__(self):
        return hash((type(self).__name__, self.params))

    # -- vectorisation ----------------------------------------------------
    @staticmethod
    def _map(func, x):
        """Apply a scalar kernel to a number or to any iterable of numbers."""
        if isinstance(x, (int, float)):
            return func(float(x))
        try:
            values = iter(x)
        except TypeError:
            return func(float(x))
        return Sample([func(float(v)) for v in values])

    # -- public API -------------------------------------------------------
    def pdf(self, x):
        """Probability density (continuous) or mass (discrete) at ``x``."""
        return self._map(self._pdf, x)

    def logpdf(self, x):
        """Log of :meth:`pdf`, computed without intermediate underflow."""
        return self._map(self._logpdf, x)

    def cdf(self, x):
        """Cumulative distribution function ``P(X <= x)``."""
        return self._map(self._cdf, x)

    def logcdf(self, x):
        """Log of :meth:`cdf`."""
        return self._map(lambda v: _log_or_ninf(self._cdf(v)), x)

    def sf(self, x):
        """Survival function ``P(X > x)``, accurate in the upper tail."""
        return self._map(self._sf, x)

    def logsf(self, x):
        """Log of :meth:`sf`."""
        return self._map(lambda v: _log_or_ninf(self._sf(v)), x)

    def ppf(self, q):
        """Quantile function: the inverse of :meth:`cdf`."""
        return self._map(self._checked_ppf, q)

    def isf(self, q):
        """Inverse survival function: the value exceeded with probability ``q``."""
        return self._map(self._checked_isf, q)

    def _checked_ppf(self, q):
        if not 0.0 <= q <= 1.0:
            raise ValueError("quantiles must lie in [0, 1]")
        return self._ppf(q)

    def _checked_isf(self, q):
        if not 0.0 <= q <= 1.0:
            raise ValueError("quantiles must lie in [0, 1]")
        return self._isf(q)

    def rvs(self, size=None, random_state=None):
        """Draw random variates.

        With ``size=None`` a single value is returned, otherwise a
        :class:`~actstats.sample.Sample` of that length.
        """
        rng = _resolve(random_state if random_state is not None else self._random_state)
        if size is None:
            return self._rvs1(rng)
        n = int(size)
        if n < 0:
            raise ValueError("size must be non-negative")
        return Sample(self._rvs_many(rng, n))

    def _rvs_many(self, rng, n):
        """Draw ``n`` variates.

        Subclasses override this with a tighter loop where that pays; a batch
        always consumes the same underlying uniforms as ``n`` single draws, so
        batching changes the speed of a simulation and not its results.
        """
        draw = self._rvs1
        return [draw(rng) for _ in range(n)]

    # -- moments ----------------------------------------------------------
    def mean(self):
        """Expected value."""
        raise NotImplementedError

    def var(self):
        """Variance."""
        raise NotImplementedError

    def std(self):
        """Standard deviation."""
        v = self.var()
        return _SQRT(v) if v == v and v != _INF else v

    def median(self):
        """Median."""
        return self._ppf(0.5)

    def stats(self, moments="mv"):
        """Selected moments, mirroring SciPy's ``stats`` accessor.

        ``m`` mean, ``v`` variance, ``s`` skewness, ``k`` excess kurtosis.
        """
        out = []
        for code in moments:
            if code == "m":
                out.append(self.mean())
            elif code == "v":
                out.append(self.var())
            elif code == "s":
                out.append(self.skewness())
            elif code == "k":
                out.append(self.kurtosis())
            else:
                raise ValueError(f"unknown moment code: {code!r}")
        return tuple(out) if len(out) != 1 else out[0]

    def moment(self, n):
        """``n``-th raw moment ``E[X^n]``."""
        raise NotImplementedError(
            f"raw moments are not implemented for {type(self).__name__}"
        )

    def skewness(self):
        """Skewness, from the first three raw moments."""
        m1, m2, m3 = self.moment(1), self.moment(2), self.moment(3)
        var = m2 - m1 * m1
        return (m3 - 3.0 * m1 * var - m1 ** 3) / var ** 1.5

    def kurtosis(self):
        """Excess kurtosis, from the first four raw moments."""
        m1, m2, m3, m4 = (self.moment(k) for k in (1, 2, 3, 4))
        var = m2 - m1 * m1
        central4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4
        return central4 / (var * var) - 3.0

    def interval(self, confidence=0.95):
        """Equal-tailed interval containing ``confidence`` of the mass."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        tail = 0.5 * (1.0 - confidence)
        return (self._ppf(tail), self._isf(tail))

    # -- inference --------------------------------------------------------
    @classmethod
    def fit(cls, data, **kwargs):
        """Maximum-likelihood estimate of the parameters, as a tuple."""
        raise NotImplementedError(f"fit is not implemented for {cls.__name__}")

    def logpmf(self, x):
        """Alias of :meth:`logpdf` for lattice distributions."""
        if self.continuous:
            raise AttributeError(f"{type(self).__name__} is continuous; use logpdf")
        return self.logpdf(x)

    def pmf(self, x):
        """Alias of :meth:`pdf` for lattice distributions."""
        if self.continuous:
            raise AttributeError(f"{type(self).__name__} is continuous; use pdf")
        return self.pdf(x)

    def nnlf(self, data):
        """Negative log-likelihood of ``data`` under the fitted parameters."""
        return -math.fsum(self._logpdf(float(v)) for v in _as_float_list(data))


def _log_or_ninf(p):
    return _LOG(p) if p > 0.0 else -_INF


class _DiscreteDistribution(Distribution):
    """Shared machinery for lattice distributions on ``0, 1, 2, ...``.

    Frozen lattice distributions cache a table of cumulative probabilities the
    first time they are sampled.  Every later draw is then one uniform plus a
    binary search -- markedly cheaper than the rejection samplers, and it makes
    the cost independent of the mean.
    """

    continuous = False

    #: Largest table that will be built; beyond this the rejection sampler wins.
    _max_table = 4096

    def _table_probabilities(self):
        """Yield ``P(X = k)`` for ``k = 0, 1, 2, ...`` (subclass hook)."""
        raise NotImplementedError

    def _build_table(self):
        """Cumulative probabilities for inversion sampling, or ``None``."""
        table = []
        total = 0.0
        limit = self._max_table
        for probability in self._table_probabilities():
            total += probability
            table.append(total)
            if total >= 1.0 - 1e-15:
                return table
            if len(table) >= limit:
                return None
        # The generator stopped early, which means the mass function underflows
        # at zero; inversion is not usable for such a large mean.
        return table if table else None

    def _get_table(self):
        table = self._table
        if table is _UNBUILT:
            table = self._build_table()
            self._table = table
        return table

    def _rvs1(self, rng):
        table = self._get_table()
        if table is None:
            return self._rvs_reject(rng)
        u = rng.random()
        k = bisect_left(table, u)
        return k if k < len(table) else int(self._ppf(u))

    def _rvs_many(self, rng, n):
        table = self._get_table()
        if table is None:
            draw = self._rvs_reject
            return [draw(rng) for _ in range(n)]
        random = rng.random
        size = len(table)
        search = bisect_left
        ppf = self._ppf
        out = []
        add = out.append
        for _ in range(n):
            u = random()
            k = search(table, u)
            add(k if k < size else int(ppf(u)))
        return out

    def _rvs_reject(self, rng):
        """Sampler used when the distribution is too spread out to tabulate."""
        raise NotImplementedError

    def pmf(self, x):
        """Probability mass function."""
        return self._map(self._pdf, x)

    def logpmf(self, x):
        """Log of :meth:`pmf`."""
        return self._map(self._logpdf, x)

    def pdf(self, x):
        """Alias of :meth:`pmf` (SciPy keeps ``pdf`` off discrete laws)."""
        return self.pmf(x)

    def logpdf(self, x):
        """Alias of :meth:`logpmf`."""
        return self.logpmf(x)

    def _ppf(self, q):
        """Smallest ``k`` with ``cdf(k) >= q``, found by search from the mean."""
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return _INF
        k = self._ppf_guess(q)
        if self._cdf(k) >= q:
            while k > 0.0 and self._cdf(k - 1.0) >= q:
                k -= 1.0
            return k
        while self._cdf(k) < q:
            k += 1.0
        return k

    def _ppf_guess(self, q):
        """Normal-approximation starting point for the quantile search."""
        mu = self.mean()
        sd = self.std()
        if not (sd == sd and sd < _INF):
            return 0.0
        k = math.floor(mu + ndtri(q) * sd)
        return float(k) if k > 0.0 else 0.0

    def support(self):
        return (0.0, _INF)


# ---------------------------------------------------------------------------
# Severity distributions
# ---------------------------------------------------------------------------

class LogNormal(Distribution):
    """Lognormal severity ``lognormal(mu, sigma)``.

    ``mu`` and ``sigma`` are the mean and standard deviation of ``log X``.
    """

    param_names = ("mu", "sigma")
    param_defaults = (0.0, 1.0)

    def _validate(self):
        if self.params[1] <= 0.0:
            raise ValueError("sigma must be positive")

    def _prepare(self):
        self.mu, self.sigma = self.params
        self._log_norm = _LOG(self.sigma) + _LOG_SQRT_2PI

    def _logpdf(self, x):
        if x <= 0.0:
            return -_INF
        z = (_LOG(x) - self.mu) / self.sigma
        return -0.5 * z * z - self._log_norm - _LOG(x)

    def _cdf(self, x):
        if x <= 0.0:
            return 0.0
        return ndtr((_LOG(x) - self.mu) / self.sigma)

    def _sf(self, x):
        if x <= 0.0:
            return 1.0
        return ndtr(-(_LOG(x) - self.mu) / self.sigma)

    def _ppf(self, q):
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return _INF
        return _EXP(self.mu + self.sigma * ndtri(q))

    def _isf(self, q):
        if q <= 0.0:
            return _INF
        if q >= 1.0:
            return 0.0
        return _EXP(self.mu - self.sigma * ndtri(q))

    def _rvs1(self, rng):
        return _EXP(self.mu + self.sigma * rng.standard_normal())

    def _rvs_many(self, rng, n):
        mu, sigma, exp = self.mu, self.sigma, _EXP
        return [exp(mu + sigma * z) for z in rng.standard_normals(n)]

    def moment(self, n):
        return _EXP(n * self.mu + 0.5 * n * n * self.sigma * self.sigma)

    def mean(self):
        return _EXP(self.mu + 0.5 * self.sigma * self.sigma)

    def var(self):
        s2 = self.sigma * self.sigma
        return math.expm1(s2) * _EXP(2.0 * self.mu + s2)

    @classmethod
    def fit(cls, data, **kwargs):
        """Closed-form MLE from the logged observations."""
        values = _as_float_list(data)
        if any(v <= 0.0 for v in values):
            raise ValueError("lognormal data must be strictly positive")
        logs = [_LOG(v) for v in values]
        n = len(logs)
        mu = math.fsum(logs) / n
        var = math.fsum((v - mu) ** 2 for v in logs) / n
        return (mu, _SQRT(var))


class Gamma(Distribution):
    """Gamma severity ``gamma(alpha, theta)`` with shape ``alpha``, scale ``theta``."""

    param_names = ("alpha", "theta")
    param_defaults = (1.0, 1.0)

    def _validate(self):
        if self.params[0] <= 0.0:
            raise ValueError("alpha must be positive")
        if self.params[1] <= 0.0:
            raise ValueError("theta must be positive")

    def _prepare(self):
        self.alpha, self.theta = self.params
        self._log_norm = math.lgamma(self.alpha) + self.alpha * _LOG(self.theta)

    def _logpdf(self, x):
        if x < 0.0:
            return -_INF
        if x == 0.0:
            if self.alpha == 1.0:
                return -self._log_norm
            return -_INF if self.alpha > 1.0 else _INF
        return (self.alpha - 1.0) * _LOG(x) - x / self.theta - self._log_norm

    def _cdf(self, x):
        if x <= 0.0:
            return 0.0
        return gammainc(self.alpha, x / self.theta)

    def _sf(self, x):
        if x <= 0.0:
            return 1.0
        return gammaincc(self.alpha, x / self.theta)

    def _ppf(self, q):
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return _INF
        return self.theta * gammaincinv(self.alpha, q)

    def _isf(self, q):
        if q <= 0.0:
            return _INF
        if q >= 1.0:
            return 0.0
        return self.theta * gammainccinv(self.alpha, q)

    def _rvs1(self, rng):
        return self.theta * rng.standard_gamma(self.alpha)

    def _rvs_many(self, rng, n):
        theta, alpha, draw = self.theta, self.alpha, rng.standard_gamma
        return [theta * draw(alpha) for _ in range(n)]

    def moment(self, n):
        return self.theta ** n * _EXP(
            math.lgamma(self.alpha + n) - math.lgamma(self.alpha)
        )

    def mean(self):
        return self.alpha * self.theta

    def var(self):
        return self.alpha * self.theta * self.theta

    @classmethod
    def fit(cls, data, **kwargs):
        """MLE with the location held at zero.

        Solves ``log(alpha) - digamma(alpha) = log(mean) - mean(log x)`` by
        Newton's method from Minka's starting value.
        """
        values = _as_float_list(data)
        if any(v <= 0.0 for v in values):
            raise ValueError("gamma data must be strictly positive")
        n = len(values)
        mean = math.fsum(values) / n
        mean_log = math.fsum(_LOG(v) for v in values) / n
        s = _LOG(mean) - mean_log
        if s <= 0.0:
            # Degenerate sample (all observations equal); no finite MLE.
            raise ValueError("gamma fit requires variation in the data")
        alpha = (3.0 - s + _SQRT((3.0 - s) ** 2 + 24.0 * s)) / (12.0 * s)
        for _ in range(100):
            f = _LOG(alpha) - digamma(alpha) - s
            fprime = 1.0 / alpha - trigamma(alpha)
            step = f / fprime
            new_alpha = alpha - step
            if new_alpha <= 0.0:
                new_alpha = alpha / 2.0
            if abs(new_alpha - alpha) <= 1e-14 * alpha:
                alpha = new_alpha
                break
            alpha = new_alpha
        return (alpha, mean / alpha)


class Weibull(Distribution):
    """Weibull severity ``weibull(delta, beta)`` with shape ``delta``, scale ``beta``."""

    param_names = ("delta", "beta")
    param_defaults = (1.0, 1.0)

    def _validate(self):
        if self.params[0] <= 0.0:
            raise ValueError("delta must be positive")
        if self.params[1] <= 0.0:
            raise ValueError("beta must be positive")

    def _prepare(self):
        self.delta, self.beta = self.params

    def _logpdf(self, x):
        if x < 0.0:
            return -_INF
        d, b = self.delta, self.beta
        if x == 0.0:
            return _LOG(d / b) if d == 1.0 else (-_INF if d > 1.0 else _INF)
        z = x / b
        return _LOG(d / b) + (d - 1.0) * _LOG(z) - z ** d

    def _cdf(self, x):
        if x <= 0.0:
            return 0.0
        return -math.expm1(-((x / self.beta) ** self.delta))

    def _sf(self, x):
        if x <= 0.0:
            return 1.0
        return _EXP(-((x / self.beta) ** self.delta))

    def _ppf(self, q):
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return _INF
        return self.beta * (-math.log1p(-q)) ** (1.0 / self.delta)

    def _isf(self, q):
        if q <= 0.0:
            return _INF
        if q >= 1.0:
            return 0.0
        return self.beta * (-_LOG(q)) ** (1.0 / self.delta)

    def _rvs1(self, rng):
        return self.beta * (-_LOG(1.0 - rng.random())) ** (1.0 / self.delta)

    def _rvs_many(self, rng, n):
        beta, inv_delta, log, random = self.beta, 1.0 / self.delta, _LOG, rng.random
        return [beta * (-log(1.0 - random())) ** inv_delta for _ in range(n)]

    def moment(self, n):
        return self.beta ** n * _EXP(math.lgamma(1.0 + n / self.delta))

    def mean(self):
        return self.beta * math.gamma(1.0 + 1.0 / self.delta)

    def var(self):
        g1 = math.gamma(1.0 + 1.0 / self.delta)
        g2 = math.gamma(1.0 + 2.0 / self.delta)
        return self.beta * self.beta * (g2 - g1 * g1)

    @classmethod
    def fit(cls, data, **kwargs):
        """MLE with the location held at zero (Newton on the shape equation)."""
        values = _as_float_list(data)
        if any(v <= 0.0 for v in values):
            raise ValueError("weibull data must be strictly positive")
        n = len(values)
        logs = [_LOG(v) for v in values]
        mean_log = math.fsum(logs) / n

        def score(delta):
            # d/d(delta) of the profile log-likelihood.
            powers = [v ** delta for v in values]
            total = math.fsum(powers)
            if total <= 0.0:
                return -_INF
            weighted = math.fsum(p * lg for p, lg in zip(powers, logs))
            return weighted / total - 1.0 / delta - mean_log

        lo, hi = bracket_root(score, 1.0)
        delta = brentq(score, lo, hi)
        beta = (math.fsum(v ** delta for v in values) / n) ** (1.0 / delta)
        return (delta, beta)


class Pareto(Distribution):
    """Two-parameter Pareto severity ``pareto(alpha, beta)``.

    Supported on ``[0, inf)`` with ``S(x) = (1 + x / beta) ** -alpha``; this is
    the Pareto of Bahnemann's monograph, and SciPy's ``lomax``.
    """

    param_names = ("alpha", "beta")
    param_defaults = (1.0, 1.0)

    def _validate(self):
        if self.params[0] <= 0.0:
            raise ValueError("alpha must be positive")
        if self.params[1] <= 0.0:
            raise ValueError("beta must be positive")

    def _prepare(self):
        self.alpha, self.beta = self.params

    def _logpdf(self, x):
        if x < 0.0:
            return -_INF
        a, b = self.alpha, self.beta
        return _LOG(a / b) - (a + 1.0) * math.log1p(x / b)

    def _cdf(self, x):
        if x <= 0.0:
            return 0.0
        return -math.expm1(-self.alpha * math.log1p(x / self.beta))

    def _sf(self, x):
        if x <= 0.0:
            return 1.0
        return _EXP(-self.alpha * math.log1p(x / self.beta))

    def _ppf(self, q):
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return _INF
        return self.beta * math.expm1(-math.log1p(-q) / self.alpha)

    def _isf(self, q):
        if q <= 0.0:
            return _INF
        if q >= 1.0:
            return 0.0
        return self.beta * math.expm1(-_LOG(q) / self.alpha)

    def _rvs1(self, rng):
        return self.beta * ((1.0 - rng.random()) ** (-1.0 / self.alpha) - 1.0)

    def _rvs_many(self, rng, n):
        beta, power, random = self.beta, -1.0 / self.alpha, rng.random
        return [beta * ((1.0 - random()) ** power - 1.0) for _ in range(n)]

    def moment(self, n):
        if n >= self.alpha:
            return _INF
        return self.beta ** n * _EXP(
            math.lgamma(n + 1.0) + math.lgamma(self.alpha - n) - math.lgamma(self.alpha)
        )

    def mean(self):
        if self.alpha <= 1.0:
            return _INF
        return self.beta / (self.alpha - 1.0)

    def var(self):
        a, b = self.alpha, self.beta
        if a <= 2.0:
            return _INF
        return (b * b * a) / ((a - 1.0) ** 2 * (a - 2.0))

    @classmethod
    def fit(cls, data, **kwargs):
        """MLE with the location held at zero.

        The shape has a closed form given the scale, so the scale is found by
        maximising the profile likelihood.
        """
        values = _as_float_list(data)
        if any(v < 0.0 for v in values):
            raise ValueError("pareto data must be non-negative")
        n = len(values)
        mean = math.fsum(values) / n

        def neg_profile(log_beta_):
            beta = _EXP(log_beta_)
            s = math.fsum(math.log1p(v / beta) for v in values)
            if s <= 0.0:
                return _INF
            alpha = n / s
            # -log L, dropping terms that do not depend on beta.
            return -(n * _LOG(alpha) - n * _LOG(beta) - (alpha + 1.0) * s)

        lo = _LOG(mean) - 12.0
        hi = _LOG(mean) + 12.0
        log_beta_hat = minimize_scalar(neg_profile, lo, hi)
        beta = _EXP(log_beta_hat)
        alpha = n / math.fsum(math.log1p(v / beta) for v in values)
        return (alpha, beta)


class Beta(Distribution):
    """Beta distribution ``beta(alpha, beta)`` on ``(0, 1)``."""

    param_names = ("alpha", "beta")
    param_defaults = (1.0, 1.0)

    def _validate(self):
        if self.params[0] <= 0.0 or self.params[1] <= 0.0:
            raise ValueError("alpha and beta must be positive")

    def _prepare(self):
        self.alpha, self.beta = self.params
        self._log_norm = log_beta(self.alpha, self.beta)

    def support(self):
        return (0.0, 1.0)

    def _logpdf(self, x):
        if x < 0.0 or x > 1.0:
            return -_INF
        a, b = self.alpha, self.beta
        if x == 0.0:
            return -_INF if a > 1.0 else (_INF if a < 1.0 else -self._log_norm)
        if x == 1.0:
            return -_INF if b > 1.0 else (_INF if b < 1.0 else -self._log_norm)
        return (a - 1.0) * _LOG(x) + (b - 1.0) * math.log1p(-x) - self._log_norm

    def _cdf(self, x):
        return betainc(self.alpha, self.beta, x)

    def _sf(self, x):
        return betainc(self.beta, self.alpha, 1.0 - x)

    def _ppf(self, q):
        return betaincinv(self.alpha, self.beta, q)

    def _isf(self, q):
        return betainccinv(self.alpha, self.beta, q)

    def _rvs1(self, rng):
        return rng.beta(self.alpha, self.beta)

    def _rvs_many(self, rng, n):
        a, b, draw = self.alpha, self.beta, rng.beta
        return [draw(a, b) for _ in range(n)]

    def moment(self, n):
        a, b = self.alpha, self.beta
        return _EXP(
            math.lgamma(a + n) - math.lgamma(a) + math.lgamma(a + b) - math.lgamma(a + b + n)
        )

    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    def var(self):
        a, b = self.alpha, self.beta
        s = a + b
        return (a * b) / (s * s * (s + 1.0))

    @classmethod
    def fit(cls, data, **kwargs):
        """MLE on ``(0, 1)`` by two-dimensional Newton on the digamma system."""
        values = _as_float_list(data)
        if any(not 0.0 < v < 1.0 for v in values):
            raise ValueError("beta data must lie strictly inside (0, 1)")
        n = len(values)
        mean_log = math.fsum(_LOG(v) for v in values) / n
        mean_log1m = math.fsum(math.log1p(-v) for v in values) / n
        mean = math.fsum(values) / n
        var = math.fsum((v - mean) ** 2 for v in values) / n
        # Method-of-moments start.
        common = mean * (1.0 - mean) / var - 1.0 if var > 0.0 else 1.0
        a = max(mean * common, 1e-3)
        b = max((1.0 - mean) * common, 1e-3)
        for _ in range(200):
            psi_ab = digamma(a + b)
            g1 = psi_ab - digamma(a) + mean_log
            g2 = psi_ab - digamma(b) + mean_log1m
            tri_ab = trigamma(a + b)
            h11 = tri_ab - trigamma(a)
            h22 = tri_ab - trigamma(b)
            h12 = tri_ab
            det = h11 * h22 - h12 * h12
            if det == 0.0:
                break
            da = (g1 * h22 - g2 * h12) / det
            db = (g2 * h11 - g1 * h12) / det
            step = 1.0
            while step > 1e-8 and (a - step * da <= 0.0 or b - step * db <= 0.0):
                step *= 0.5
            new_a = a - step * da
            new_b = b - step * db
            converged = abs(new_a - a) <= 1e-12 * a and abs(new_b - b) <= 1e-12 * b
            a, b = new_a, new_b
            if converged:
                break
        return (a, b)


class Normal(Distribution):
    """Normal distribution ``normal(mu, sigma)``."""

    param_names = ("mu", "sigma")
    param_defaults = (0.0, 1.0)

    def _validate(self):
        if self.params[1] <= 0.0:
            raise ValueError("sigma must be positive")

    def _prepare(self):
        self.mu, self.sigma = self.params
        self._log_norm = _LOG(self.sigma) + _LOG_SQRT_2PI

    def support(self):
        return (-_INF, _INF)

    def _logpdf(self, x):
        z = (x - self.mu) / self.sigma
        return -0.5 * z * z - self._log_norm

    def _cdf(self, x):
        return ndtr((x - self.mu) / self.sigma)

    def _sf(self, x):
        return ndtr(-(x - self.mu) / self.sigma)

    def _ppf(self, q):
        return self.mu + self.sigma * ndtri(q)

    def _isf(self, q):
        return self.mu - self.sigma * ndtri(q)

    def _rvs1(self, rng):
        return self.mu + self.sigma * rng.standard_normal()

    def _rvs_many(self, rng, n):
        mu, sigma = self.mu, self.sigma
        return [mu + sigma * z for z in rng.standard_normals(n)]

    def mean(self):
        return self.mu

    def var(self):
        return self.sigma * self.sigma

    def skewness(self):
        return 0.0

    def kurtosis(self):
        return 0.0

    def moment(self, n):
        # Raw moments via the central-moment expansion.
        total = 0.0
        for k in range(n + 1):
            if (n - k) % 2:
                continue
            central = _double_factorial(n - k - 1) * self.sigma ** (n - k)
            total += math.comb(n, k) * self.mu ** k * central
        return total

    @classmethod
    def fit(cls, data, **kwargs):
        """Closed-form MLE (mean and population standard deviation)."""
        values = _as_float_list(data)
        n = len(values)
        mu = math.fsum(values) / n
        var = math.fsum((v - mu) ** 2 for v in values) / n
        return (mu, _SQRT(var))


def _double_factorial(n):
    """``n!!`` for odd ``n``; ``1`` for ``n <= 0``."""
    result = 1.0
    while n > 1:
        result *= n
        n -= 2
    return result


class Logistic(Distribution):
    """Logistic distribution ``logistic(mu, s)``."""

    param_names = ("mu", "s")
    param_defaults = (0.0, 1.0)

    def _validate(self):
        if self.params[1] <= 0.0:
            raise ValueError("s must be positive")

    def _prepare(self):
        self.mu, self.s = self.params

    def support(self):
        return (-_INF, _INF)

    def _logpdf(self, x):
        z = (x - self.mu) / self.s
        az = z if z >= 0.0 else -z
        # -|z| - 2 log(1 + exp(-|z|)) is the overflow-free form.
        return -az - 2.0 * math.log1p(_EXP(-az)) - _LOG(self.s)

    def _cdf(self, x):
        z = (x - self.mu) / self.s
        if z >= 0.0:
            return 1.0 / (1.0 + _EXP(-z))
        e = _EXP(z)
        return e / (1.0 + e)

    def _sf(self, x):
        return self._cdf(2.0 * self.mu - x)

    def _ppf(self, q):
        if q <= 0.0:
            return -_INF
        if q >= 1.0:
            return _INF
        return self.mu + self.s * (_LOG(q) - math.log1p(-q))

    def _isf(self, q):
        return 2.0 * self.mu - self._ppf(q)

    def _rvs1(self, rng):
        return rng.logistic(self.mu, self.s)

    def _rvs_many(self, rng, n):
        mu, scale, log, random = self.mu, self.s, _LOG, rng.random
        out = []
        add = out.append
        for _ in range(n):
            u = random()
            while u <= 0.0:
                u = random()
            add(mu + scale * log(u / (1.0 - u)))
        return out

    def mean(self):
        return self.mu

    def var(self):
        return (math.pi * math.pi / 3.0) * self.s * self.s

    def skewness(self):
        return 0.0

    def kurtosis(self):
        return 1.2

    @classmethod
    def fit(cls, data, **kwargs):
        """MLE by Newton iteration on the location and scale."""
        values = _as_float_list(data)
        n = len(values)
        mu = math.fsum(values) / n
        var = math.fsum((v - mu) ** 2 for v in values) / n
        s = _SQRT(3.0 * var) / math.pi if var > 0.0 else 1.0
        for _ in range(200):
            g_mu = 0.0
            g_s = 0.0
            for v in values:
                z = (v - mu) / s
                w = 1.0 / (1.0 + _EXP(-z)) if z >= 0.0 else _EXP(z) / (1.0 + _EXP(z))
                g_mu += 2.0 * w - 1.0
                g_s += z * (2.0 * w - 1.0) - 1.0
            g_mu /= s
            g_s /= s
            # Fisher information for the logistic (per observation).
            i_mu = n / (3.0 * s * s)
            i_s = n * (math.pi * math.pi / 9.0 + 1.0 / 3.0) / (s * s)
            new_mu = mu + g_mu / i_mu
            new_s = s + g_s / i_s
            if new_s <= 0.0:
                new_s = s / 2.0
            if abs(new_mu - mu) <= 1e-12 * (abs(mu) + 1.0) and abs(new_s - s) <= 1e-12 * s:
                mu, s = new_mu, new_s
                break
            mu, s = new_mu, new_s
        return (mu, s)


class Exponential(Distribution):
    """Exponential severity ``exponential(beta)`` with mean ``beta``."""

    param_names = ("beta",)
    param_defaults = (1.0,)

    def _validate(self):
        if self.params[0] <= 0.0:
            raise ValueError("beta must be positive")

    def _prepare(self):
        self.beta = self.params[0]

    def _logpdf(self, x):
        if x < 0.0:
            return -_INF
        return -x / self.beta - _LOG(self.beta)

    def _cdf(self, x):
        if x <= 0.0:
            return 0.0
        return -math.expm1(-x / self.beta)

    def _sf(self, x):
        if x <= 0.0:
            return 1.0
        return _EXP(-x / self.beta)

    def _ppf(self, q):
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return _INF
        return -self.beta * math.log1p(-q)

    def _isf(self, q):
        if q <= 0.0:
            return _INF
        if q >= 1.0:
            return 0.0
        return -self.beta * _LOG(q)

    def _rvs1(self, rng):
        return -self.beta * _LOG(1.0 - rng.random())

    def _rvs_many(self, rng, n):
        beta, log, random = self.beta, _LOG, rng.random
        return [-beta * log(1.0 - random()) for _ in range(n)]

    def moment(self, n):
        return self.beta ** n * math.gamma(n + 1.0)

    def mean(self):
        return self.beta

    def var(self):
        return self.beta * self.beta

    @classmethod
    def fit(cls, data, **kwargs):
        """Closed-form MLE with the location held at zero."""
        values = _as_float_list(data)
        if any(v < 0.0 for v in values):
            raise ValueError("exponential data must be non-negative")
        return (math.fsum(values) / len(values),)


class Uniform(Distribution):
    """Uniform distribution ``uniform(a, b)`` on ``[a, b]``."""

    param_names = ("a", "b")
    param_defaults = (0.0, 1.0)

    def _validate(self):
        if self.params[1] <= self.params[0]:
            raise ValueError("b must be greater than a")

    def _prepare(self):
        self.a, self.b = self.params
        self._width = self.b - self.a

    def support(self):
        return (self.a, self.b)

    def _logpdf(self, x):
        if x < self.a or x > self.b:
            return -_INF
        return -_LOG(self._width)

    def _cdf(self, x):
        if x <= self.a:
            return 0.0
        if x >= self.b:
            return 1.0
        return (x - self.a) / self._width

    def _ppf(self, q):
        return self.a + q * self._width

    def _isf(self, q):
        return self.b - q * self._width

    def _rvs1(self, rng):
        return self.a + self._width * rng.random()

    def _rvs_many(self, rng, n):
        a, width, random = self.a, self._width, rng.random
        return [a + width * random() for _ in range(n)]

    def moment(self, n):
        return (self.b ** (n + 1) - self.a ** (n + 1)) / ((n + 1) * self._width)

    def mean(self):
        return 0.5 * (self.a + self.b)

    def var(self):
        return self._width * self._width / 12.0

    def skewness(self):
        return 0.0

    def kurtosis(self):
        return -1.2

    @classmethod
    def fit(cls, data, **kwargs):
        """MLE: the observed range."""
        values = _as_float_list(data)
        return (min(values), max(values))


# ---------------------------------------------------------------------------
# Frequency distributions
# ---------------------------------------------------------------------------

class Poisson(_DiscreteDistribution):
    """Poisson frequency ``poisson(lambda_)``."""

    param_names = ("lambda_",)
    param_defaults = (1.0,)

    def _validate(self):
        if self.params[0] < 0.0:
            raise ValueError("lambda_ must be non-negative")

    def _prepare(self):
        self.lambda_ = self.params[0]
        self._log_lambda = _LOG(self.lambda_) if self.lambda_ > 0.0 else -_INF
        self._table = _UNBUILT

    def _logpdf(self, k):
        if k < 0.0 or k != math.floor(k):
            return -_INF
        if self.lambda_ == 0.0:
            return 0.0 if k == 0.0 else -_INF
        return -self.lambda_ + k * self._log_lambda - math.lgamma(k + 1.0)

    def _cdf(self, k):
        k = math.floor(k)
        if k < 0.0:
            return 0.0
        if self.lambda_ == 0.0:
            return 1.0
        return gammaincc(k + 1.0, self.lambda_)

    def _sf(self, k):
        k = math.floor(k)
        if k < 0.0:
            return 1.0
        if self.lambda_ == 0.0:
            return 0.0
        return gammainc(k + 1.0, self.lambda_)

    def _table_probabilities(self):
        lam = self.lambda_
        probability = _EXP(-lam)
        if probability == 0.0:
            return  # the mean is so large that P(X = 0) underflows
        k = 0
        while True:
            yield probability
            k += 1
            probability *= lam / k

    def _rvs_reject(self, rng):
        return rng.poisson(self.lambda_)

    def mean(self):
        return self.lambda_

    def var(self):
        return self.lambda_

    def skewness(self):
        return 1.0 / _SQRT(self.lambda_)

    def kurtosis(self):
        return 1.0 / self.lambda_

    @classmethod
    def fit(cls, data, **kwargs):
        """MLE: the sample mean."""
        values = _as_float_list(data)
        if any(v < 0.0 for v in values):
            raise ValueError("poisson data must be non-negative counts")
        return (math.fsum(values) / len(values),)


class NegativeBinomial(_DiscreteDistribution):
    """Negative binomial frequency ``negative_binomial(r, p)``.

    ``X`` counts the failures before the ``r``-th success, so ``r`` may be any
    positive real number (the Polya form).
    """

    param_names = ("r", "p")
    param_defaults = (1.0, 0.5)

    def _validate(self):
        if self.params[0] <= 0.0:
            raise ValueError("r must be positive")
        if not 0.0 < self.params[1] <= 1.0:
            raise ValueError("p must lie in (0, 1]")

    def _prepare(self):
        self.r, self.p = self.params
        self._log_p = _LOG(self.p)
        self._log_q = math.log1p(-self.p) if self.p < 1.0 else -_INF
        self._table = _UNBUILT

    def _logpdf(self, k):
        if k < 0.0 or k != math.floor(k):
            return -_INF
        if self.p == 1.0:
            return 0.0 if k == 0.0 else -_INF
        return (
            math.lgamma(k + self.r)
            - math.lgamma(self.r)
            - math.lgamma(k + 1.0)
            + self.r * self._log_p
            + k * self._log_q
        )

    def _cdf(self, k):
        k = math.floor(k)
        if k < 0.0:
            return 0.0
        return betainc(self.r, k + 1.0, self.p)

    def _sf(self, k):
        k = math.floor(k)
        if k < 0.0:
            return 1.0
        return betainc(k + 1.0, self.r, 1.0 - self.p)

    def _table_probabilities(self):
        r, p = self.r, self.p
        q = 1.0 - p
        probability = p ** r
        if probability == 0.0:
            return  # P(X = 0) underflows; fall back to the mixture sampler
        k = 0
        while True:
            yield probability
            probability *= (k + r) * q / (k + 1)
            k += 1

    def _rvs_reject(self, rng):
        return rng.negative_binomial(self.r, self.p)

    def mean(self):
        return self.r * (1.0 - self.p) / self.p

    def var(self):
        return self.r * (1.0 - self.p) / (self.p * self.p)

    def skewness(self):
        q = 1.0 - self.p
        return (2.0 - self.p) / _SQRT(self.r * q)

    def kurtosis(self):
        q = 1.0 - self.p
        return (self.p * self.p - 6.0 * self.p + 6.0) / (self.r * q)

    @classmethod
    def fit(cls, data, method="mle", **kwargs):
        """Fit ``(r, p)``.

        ``method='mle'`` (the default) maximises the likelihood; ``'moments'``
        matches the sample mean and variance.  Both need over-dispersed data:
        a sample whose variance does not exceed its mean has no negative
        binomial fit, and Poisson should be used instead.
        """
        values = _as_float_list(data)
        if any(v < 0.0 for v in values):
            raise ValueError("negative binomial data must be non-negative counts")
        n = len(values)
        mean = math.fsum(values) / n
        var = math.fsum((v - mean) ** 2 for v in values) / n
        if mean <= 0.0:
            raise ValueError("negative binomial fit requires a positive mean")
        if var <= mean:
            raise ValueError(
                "variance must exceed the mean for a negative binomial fit; "
                "the data are not over-dispersed"
            )
        r_mom = mean * mean / (var - mean)
        if method == "moments":
            return (r_mom, r_mom / (r_mom + mean))
        if method != "mle":
            raise ValueError("method must be 'mle' or 'moments'")

        # Profile likelihood: given r, p_hat = r / (r + mean).
        def score(r):
            return (
                math.fsum(digamma(v + r) for v in values) / n
                - digamma(r)
                + _LOG(r / (r + mean))
            )

        try:
            lo, hi = bracket_root(score, r_mom)
            r = brentq(score, lo, hi)
        except (RuntimeError, ValueError):
            r = r_mom
        return (r, r / (r + mean))


# ---------------------------------------------------------------------------
# Nonhomogeneous Poisson process
# ---------------------------------------------------------------------------

class NonhomogeneousPoisson(Distribution):
    """Nonhomogeneous Poisson process with a seasonal rate.

    The intensity is ``lambda(t) = lambda0 * (1 + alpha * sin(2*pi*t + phase))``
    on ``[0, T]``.  ``rvs`` returns event *times*: thinning by default, or the
    order-statistics construction when ``n_events`` is given.
    """

    param_names = ("lambda0", "alpha", "phase", "T")
    param_defaults = (10.0, 0.5, 0.0, 1.0)
    continuous = True

    def __init__(self, *params, num_points=10000, random_state=None, **kwargs):
        self.num_points = int(num_points)
        super().__init__(*params, random_state=random_state, **kwargs)

    def _validate(self):
        if self.params[0] < 0.0:
            raise ValueError("lambda0 must be non-negative")
        if self.params[3] <= 0.0:
            raise ValueError("T must be positive")
        if self.num_points < 2:
            raise ValueError("num_points must be at least 2")

    def _prepare(self):
        self.lambda0, self.alpha, self.phase, self.T = self.params
        self.lambda_max = self.lambda0 * (1.0 + abs(self.alpha))
        # Grid of the cumulative intensity, used to invert it by interpolation.
        n = self.num_points
        dt = self.T / (n - 1)
        self.dt = dt
        two_pi = 2.0 * math.pi
        lam0, a, ph = self.lambda0, self.alpha, self.phase
        grid = [i * dt for i in range(n)]
        rates = [lam0 * (1.0 + a * math.sin(two_pi * t + ph)) for t in grid]
        cumulative = []
        running = 0.0
        for rate in rates:
            running += rate
            cumulative.append(running * dt)
        total = cumulative[-1]
        self.t_grid = grid
        self.rates = rates
        self.cum_intensity = cumulative
        self.F = [c / total for c in cumulative] if total > 0.0 else [0.0] * n

    def support(self):
        return (0.0, self.T)

    def rate(self, t):
        """Instantaneous intensity at time ``t``."""
        return self._map(
            lambda v: self.lambda0 * (1.0 + self.alpha * math.sin(2.0 * math.pi * v + self.phase)),
            t,
        )

    def _pdf(self, t):
        return self.lambda0 * (1.0 + self.alpha * math.sin(2.0 * math.pi * t + self.phase))

    def pdf(self, t):
        """Instantaneous rate at ``t``.

        A process has no density in the usual sense; this reports the
        intensity, matching the behaviour of the original implementation.
        """
        return self.rate(t)

    def expected_events(self, t=None):
        """Expected number of events in ``[0, t]`` (default the whole horizon)."""
        if t is None:
            t = self.T
        two_pi = 2.0 * math.pi
        lam0, a, ph = self.lambda0, self.alpha, self.phase
        return lam0 * t + (lam0 * a / two_pi) * (math.cos(ph) - math.cos(two_pi * t + ph))

    def mean(self):
        """Expected number of events over ``[0, T]``."""
        return self.expected_events()

    def var(self):
        """Variance of the event count (equal to the mean for a Poisson process)."""
        return self.expected_events()

    def _interp(self, u):
        """Invert the normalised cumulative intensity by linear interpolation."""
        grid_f = self.F
        lo, hi = 0, len(grid_f) - 1
        if u <= grid_f[0]:
            return self.t_grid[0]
        if u >= grid_f[hi]:
            return self.t_grid[hi]
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if grid_f[mid] <= u:
                lo = mid
            else:
                hi = mid
        f_lo, f_hi = grid_f[lo], grid_f[hi]
        if f_hi == f_lo:
            return self.t_grid[lo]
        weight = (u - f_lo) / (f_hi - f_lo)
        return self.t_grid[lo] + weight * (self.t_grid[hi] - self.t_grid[lo])

    def _simulate_once(self, rng, n_events=None):
        if n_events is not None:
            us = sorted(rng.random() for _ in range(int(n_events)))
            return Sample([self._interp(u) for u in us])
        # Thinning (Lewis-Shedler).
        events = []
        t = 0.0
        lam_max = self.lambda_max
        if lam_max <= 0.0:
            return Sample()
        random = rng.random
        two_pi = 2.0 * math.pi
        lam0, a, ph = self.lambda0, self.alpha, self.phase
        horizon = self.T
        while True:
            t -= _LOG(1.0 - random()) / lam_max
            if t >= horizon:
                return Sample(events)
            if random() * lam_max < lam0 * (1.0 + a * math.sin(two_pi * t + ph)):
                events.append(t)

    def rvs(self, size=1, n_events=None, random_state=None):
        """Simulate event times.

        ``size=1`` returns a single :class:`~actstats.sample.Sample` of event
        times; a larger ``size`` returns a list of such samples.
        """
        rng = _resolve(random_state if random_state is not None else self._random_state)
        if size == 1:
            return self._simulate_once(rng, n_events)
        return [self._simulate_once(rng, n_events) for _ in range(int(size))]

    def _cdf(self, t):
        """Fraction of the horizon's expected events that fall in ``[0, t]``."""
        if t <= 0.0:
            return 0.0
        if t >= self.T:
            return 1.0
        return self.expected_events(t) / self.expected_events(self.T)

    def _ppf(self, q):
        return self._interp(q)

    @classmethod
    def fit(cls, data, **kwargs):
        raise NotImplementedError(
            "fitting a nonhomogeneous Poisson process is not supported"
        )
