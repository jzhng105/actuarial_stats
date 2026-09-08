"""Pure-Python random variate generation.

Everything is built on :class:`random.Random` (the standard library's
Mersenne Twister), so ``actstats`` can simulate without NumPy.  The generators
are the same algorithms NumPy uses -- Marsaglia-Tsang for the gamma,
Hoermann's PTRS for the Poisson -- so the statistical quality matches; only the
underlying bit stream differs.

A single module level generator backs the convenience functions, and
:func:`seed` makes a run reproducible::

    >>> import actstats
    >>> actstats.seed(42)

Pass ``random_state=`` to any distribution to keep an independent stream
instead.
"""

from __future__ import annotations

import math
import random as _random_module

__all__ = ["RandomState", "default_rng", "get_random_state", "seed"]

_LOG = math.log
_SQRT = math.sqrt
_EXP = math.exp
_LGAMMA = math.lgamma
_TWO_PI = 2.0 * math.pi


class RandomState:
    """Random variate generator for the distributions in :mod:`actstats`.

    Parameters
    ----------
    seed:
        Optional seed, or an existing :class:`random.Random` instance to draw
        from.
    """

    __slots__ = ("_rng", "random", "_spare")

    def __init__(self, seed=None):
        if isinstance(seed, _random_module.Random):
            self._rng = seed
        else:
            self._rng = _random_module.Random(seed)
        # The bound method is hoisted once: the sampling loops below call it
        # millions of times and the attribute lookup is a measurable cost.
        self.random = self._rng.random
        self._spare = None

    # -- plumbing ---------------------------------------------------------
    def seed(self, value=None):
        """Reseed the underlying generator."""
        self._rng.seed(value)
        self._spare = None

    def get_state(self):
        """Return the internal generator state (see :meth:`set_state`)."""
        return (self._rng.getstate(), self._spare)

    def set_state(self, state):
        """Restore a state captured with :meth:`get_state`."""
        inner, spare = state
        self._rng.setstate(inner)
        self._spare = spare

    # -- continuous variates ---------------------------------------------
    def uniform(self, low=0.0, high=1.0):
        """Uniform variate on ``[low, high)``."""
        return low + (high - low) * self.random()

    def standard_normal(self):
        """Standard normal variate (Box-Muller, keeping the spare)."""
        z = self._spare
        if z is not None:
            self._spare = None
            return z
        random = self.random
        radius = _SQRT(-2.0 * _LOG(1.0 - random()))
        theta = _TWO_PI * random()
        self._spare = radius * math.sin(theta)
        return radius * math.cos(theta)

    def standard_normals(self, n):
        """``n`` standard normal variates, generated in one tight loop.

        Drawing in bulk consumes exactly the same underlying uniforms as ``n``
        successive :meth:`standard_normal` calls, so batching a simulation
        never changes its results -- only its speed.
        """
        out = []
        add = out.append
        remaining = n
        z = self._spare
        if z is not None:
            self._spare = None
            add(z)
            remaining -= 1
        random = self.random
        sqrt = _SQRT
        log = _LOG
        cos = math.cos
        sin = math.sin
        two_pi = _TWO_PI
        while remaining >= 2:
            radius = sqrt(-2.0 * log(1.0 - random()))
            theta = two_pi * random()
            add(radius * cos(theta))
            add(radius * sin(theta))
            remaining -= 2
        if remaining == 1:
            radius = sqrt(-2.0 * log(1.0 - random()))
            theta = two_pi * random()
            add(radius * cos(theta))
            self._spare = radius * sin(theta)
        return out

    def gauss(self, mu=0.0, sigma=1.0):
        """Normal variate with mean ``mu`` and standard deviation ``sigma``."""
        return mu + sigma * self.standard_normal()

    def normal(self, mu=0.0, sigma=1.0):
        """Normal variate with mean ``mu`` and standard deviation ``sigma``."""
        return mu + sigma * self.standard_normal()

    def lognormal(self, mu=0.0, sigma=1.0):
        """Lognormal variate whose log has mean ``mu`` and sd ``sigma``."""
        return _EXP(mu + sigma * self.standard_normal())

    def standard_exponential(self):
        """Exponential variate with unit mean."""
        return -_LOG(1.0 - self.random())

    def exponential(self, scale=1.0):
        """Exponential variate with mean ``scale``."""
        return -scale * _LOG(1.0 - self.random())

    def standard_gamma(self, shape):
        """Gamma variate with unit scale (Marsaglia-Tsang)."""
        if shape <= 0.0:
            raise ValueError("shape must be positive")
        random = self.random
        if shape < 1.0:
            # Boost the shape and scale the result back down.
            u = 1.0 - random()
            return self.standard_gamma(shape + 1.0) * u ** (1.0 / shape)
        normal = self.standard_normal
        d = shape - 1.0 / 3.0
        c = 1.0 / _SQRT(9.0 * d)
        while True:
            x = normal()
            v = 1.0 + c * x
            if v <= 0.0:
                continue
            v = v * v * v
            u = random()
            x2 = x * x
            if u < 1.0 - 0.0331 * x2 * x2:
                return d * v
            if _LOG(u) < 0.5 * x2 + d * (1.0 - v + _LOG(v)):
                return d * v

    def gamma(self, shape, scale=1.0):
        """Gamma variate with the given shape and scale."""
        return scale * self.standard_gamma(shape)

    def beta(self, a, b):
        """Beta variate on ``(0, 1)``."""
        if a <= 0.0 or b <= 0.0:
            raise ValueError("beta requires a > 0 and b > 0")
        if a <= 1.0 and b <= 1.0:
            # Johnk's algorithm avoids the underflow that the gamma ratio can
            # hit when both shapes are small.
            random = self.random
            while True:
                u = random()
                v = random()
                x = u ** (1.0 / a)
                y = v ** (1.0 / b)
                s = x + y
                if s <= 1.0:
                    if s > 0.0:
                        return x / s
                    # Both terms underflowed; fall back to logs.
                    lx = _LOG(u) / a
                    ly = _LOG(v) / b
                    return _EXP(lx - (lx if lx > ly else ly)) / (
                        _EXP(lx - (lx if lx > ly else ly))
                        + _EXP(ly - (lx if lx > ly else ly))
                    )
        g1 = self.standard_gamma(a)
        g2 = self.standard_gamma(b)
        return g1 / (g1 + g2)

    def weibull(self, c, scale=1.0):
        """Weibull variate with shape ``c`` and scale ``scale``."""
        return scale * (-_LOG(1.0 - self.random())) ** (1.0 / c)

    def lomax(self, alpha, scale=1.0):
        """Lomax (Pareto type II) variate, i.e. a Pareto shifted to zero."""
        return scale * ((1.0 - self.random()) ** (-1.0 / alpha) - 1.0)

    def pareto(self, alpha, scale=1.0):
        """Classical (type I) Pareto variate supported on ``[scale, inf)``."""
        return scale * (1.0 - self.random()) ** (-1.0 / alpha)

    def logistic(self, loc=0.0, scale=1.0):
        """Logistic variate."""
        u = self.random()
        while u <= 0.0:
            u = self.random()
        return loc + scale * _LOG(u / (1.0 - u))

    # -- discrete variates ------------------------------------------------
    def poisson(self, mu):
        """Poisson variate.

        Uses the product-of-uniforms method for small means and Hoermann's
        transformed rejection (PTRS) for large ones, which keeps the cost
        constant as the mean grows.
        """
        if mu < 0.0:
            raise ValueError("mu must be non-negative")
        if mu == 0.0:
            return 0
        if mu < 10.0:
            random = self.random
            enmu = _EXP(-mu)
            x = 0
            prod = random()
            while prod > enmu:
                x += 1
                prod *= random()
            return x
        return self._poisson_ptrs(mu)

    def _poisson_ptrs(self, mu):
        random = self.random
        smu = _SQRT(mu)
        b = 0.931 + 2.53 * smu
        a = -0.059 + 0.02483 * b
        inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
        v_r = 0.9277 - 3.6224 / (b - 2.0)
        log_mu = _LOG(mu)
        while True:
            u = random() - 0.5
            v = random()
            us = 0.5 - (u if u >= 0.0 else -u)
            k = int((2.0 * a / us + b) * u + mu + 0.43)
            if us >= 0.07 and v <= v_r:
                return k
            if k < 0 or (us < 0.013 and v > us):
                continue
            if _LOG(v * inv_alpha / (a / (us * us) + b)) <= (
                -mu + k * log_mu - _LGAMMA(k + 1.0)
            ):
                return k

    def negative_binomial(self, n, p):
        """Number of failures before the ``n``-th success.

        Drawn as a gamma-Poisson mixture, which is exact and stays fast for
        large ``n``.
        """
        if n <= 0.0:
            raise ValueError("n must be positive")
        if not 0.0 < p <= 1.0:
            raise ValueError("p must lie in (0, 1]")
        if p == 1.0:
            return 0
        lam = self.standard_gamma(n) * (1.0 - p) / p
        return self.poisson(lam)

    def geometric(self, p):
        """Number of trials up to and including the first success."""
        if not 0.0 < p <= 1.0:
            raise ValueError("p must lie in (0, 1]")
        if p == 1.0:
            return 1
        u = 1.0 - self.random()
        return int(math.ceil(_LOG(u) / math.log1p(-p)))


_default_state = RandomState()


def default_rng(seed=None):
    """Return a fresh :class:`RandomState`, optionally seeded."""
    return RandomState(seed)


def get_random_state():
    """Return the module level generator used when none is supplied."""
    return _default_state


def seed(value=None):
    """Seed the module level generator, making a simulation reproducible."""
    _default_state.seed(value)


def _resolve(random_state):
    """Normalise a ``random_state`` argument to a :class:`RandomState`."""
    if random_state is None:
        return _default_state
    if isinstance(random_state, RandomState):
        return random_state
    return RandomState(random_state)
