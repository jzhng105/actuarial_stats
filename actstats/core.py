"""The ``actuarial`` front end.

``actstats.actuarial`` exposes every distribution under its actuarial name and
parameterisation.  Distributions can be used frozen (parameters supplied up
front) or unfrozen (parameters passed to each call), the two styles SciPy users
will recognise::

    dist = actuarial.lognormal(0.5, 0.2)   # frozen
    dist.rvs(size=1000)
    dist.ppf(0.99)

    actuarial.lognormal.rvs(0.5, 0.2, 1000)   # unfrozen: parameters first
    actuarial.lognormal.ppf(0.99, 0.5, 0.2)   # unfrozen: parameters last

Nothing here depends on NumPy or SciPy; see :mod:`actstats.distributions` for
the implementations.
"""

from __future__ import annotations

from .distributions import (
    Beta,
    Exponential,
    Gamma,
    LogNormal,
    Logistic,
    NegativeBinomial,
    NonhomogeneousPoisson,
    Normal,
    Pareto,
    Poisson,
    Uniform,
    Weibull,
)
from .goodness import ks_statistic, kstest
from .utils.utils import fraction_to_date_full

__all__ = [
    "ActuarialClass",
    "ActuarialDistribution",
    "actuarial",
    "fraction_to_date_full",
]

# Kept as a module level alias so ``from actstats.core import NHPPDistribution``
# still resolves for code written against earlier releases.
NHPPDistribution = NonhomogeneousPoisson


class ActuarialDistribution:
    """A distribution in actuarial parameterisation.

    Supported distributions:

    - ``lognormal(mu, sigma)``      -- ``mu``, ``sigma`` of ``log X``
    - ``gamma(alpha, theta)``       -- shape, scale
    - ``weibull(delta, beta)``      -- shape, scale
    - ``pareto(alpha, beta)``       -- two-parameter Pareto on ``[0, inf)``
    - ``beta(alpha, beta)``
    - ``poisson(lambda_)``
    - ``negative_binomial(r, p)``   -- failures before the ``r``-th success
    - ``normal(mu, sigma)``
    - ``logistic(mu, s)``
    - ``exponential(beta)``         -- mean ``beta``
    - ``uniform(a, b)``
    - ``nonhomogeneous_poisson(lambda0, alpha, phase, T)``
    """

    _distributions = {
        "lognormal": LogNormal,
        "gamma": Gamma,
        "weibull": Weibull,
        "pareto": Pareto,
        "beta": Beta,
        "poisson": Poisson,
        "negative_binomial": NegativeBinomial,
        "normal": Normal,
        "logistic": Logistic,
        "exponential": Exponential,
        "uniform": Uniform,
        "nonhomogeneous_poisson": NonhomogeneousPoisson,
    }

    def __init__(self, name, *args, random_state=None, **kwargs):
        if name not in self._distributions:
            raise ValueError(f"Unsupported distribution: {name}")
        self.name = name
        self.dist_class = self._distributions[name]
        #: ``True`` while no parameters have been supplied, in which case each
        #: call must carry them (SciPy's unfrozen style).
        self.used_default_params = not args and not kwargs
        self.dist = self.dist_class(*args, random_state=random_state, **kwargs)

    # -- parameters -------------------------------------------------------
    @property
    def params(self):
        """The distribution parameters, in the order they are declared."""
        return self.dist.params

    @property
    def param_names(self):
        """Names of the distribution parameters."""
        return self.dist_class.param_names

    def __repr__(self):
        if self.used_default_params:
            return f"<actuarial.{self.name} (unfrozen)>"
        return f"<actuarial.{self.name}{self.dist.params}>"

    # -- inference --------------------------------------------------------
    def fit(self, data, **kwargs):
        """Fit the distribution to ``data`` and return actuarial parameters.

        The estimates are maximum likelihood with the location held at zero,
        so they come back on the same scale the constructor takes::

            alpha, theta = actuarial.gamma.fit(losses)
            fitted = actuarial.gamma(alpha, theta)
        """
        return self.dist_class.fit(data, **kwargs)

    def ks_test(self, data):
        """Kolmogorov-Smirnov statistic of ``data`` against this distribution."""
        try:
            return ks_statistic(data, self.dist.cdf)
        except Exception as exc:  # pragma: no cover - defensive, as before
            raise RuntimeError(f"Error computing KS statistic: {exc}") from exc

    def kstest(self, data):
        """Kolmogorov-Smirnov test: returns ``(statistic, p_value)``."""
        return kstest(data, self.dist.cdf)

    # -- sampling ---------------------------------------------------------
    def np_rvs(self, size=None, **kwargs):
        """Deprecated alias of :meth:`rvs`.

        Sampling used to be routed through NumPy, and this name distinguished
        it from the SciPy path.  Both are now the same pure-Python sampler, so
        ``rvs`` is preferred; this alias is kept so existing scripts run
        unchanged.
        """
        return self.dist.rvs(size=size, **kwargs)

    # -- dispatch ---------------------------------------------------------
    def __getattr__(self, name):
        """Forward to the underlying distribution, frozen or not."""
        if name.startswith("__"):
            raise AttributeError(name)
        dist = self.__dict__.get("dist")
        if dist is None:
            raise AttributeError(name)
        attr = getattr(dist, name)
        if not callable(attr):
            return attr

        def method(*args, **kwargs):
            if self.used_default_params and args:
                # Unfrozen call: peel the distribution parameters off the
                # arguments.  ``rvs`` and ``stats`` take them first, every
                # other function takes its own argument first.
                n_params = len(self.dist_class.param_names)
                if name in ("rvs", "stats"):
                    params, rest = args[:n_params], args[n_params:]
                else:
                    params, rest = args[-n_params:], args[:-n_params]
                frozen = self.dist_class(*params)
                return getattr(frozen, name)(*rest, **kwargs)
            return attr(*args, **kwargs)

        method.__name__ = name
        method.__doc__ = attr.__doc__
        return method

    def __call__(self, *args, **kwargs):
        """Freeze the distribution with the given parameters."""
        return ActuarialDistribution(self.name, *args, **kwargs)

    def __dir__(self):
        return sorted(set(list(super().__dir__()) + dir(self.dist)))


class ActuarialClass:
    """Namespace giving every supported distribution by name."""

    def __getattr__(self, name):
        """Return an unfrozen :class:`ActuarialDistribution` called ``name``."""
        if name not in ActuarialDistribution._distributions:
            raise AttributeError(f"'ActuarialClass' has no attribute '{name}'")
        return ActuarialDistribution(name)

    def __dir__(self):
        return sorted(ActuarialDistribution._distributions)

    def __repr__(self):
        names = ", ".join(sorted(ActuarialDistribution._distributions))
        return f"<actuarial: {names}>"


#: The entry point used throughout the documentation and examples.
actuarial = ActuarialClass()
