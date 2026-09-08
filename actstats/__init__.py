"""actstats: statistical distributions in actuarial conventions.

The package is written in pure Python -- installing it pulls in nothing else --
and covers the severity, frequency and process models actuaries reach for,
parameterised the way the actuarial literature writes them.
"""

__version__ = "0.2.0"
__author__ = "Juntao Zhang, Tingting Shi"
__email__ = "jzhng106@gmail.com"

from .core import ActuarialClass, ActuarialDistribution, actuarial
from .distributions import (
    Beta,
    Distribution,
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
from .rng import RandomState, default_rng, get_random_state, seed
from .sample import Sample
from .utils.utils import fraction_to_date_full

__all__ = [
    "__version__",
    "actuarial",
    "ActuarialClass",
    "ActuarialDistribution",
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
    "RandomState",
    "default_rng",
    "get_random_state",
    "seed",
    "Sample",
    "ks_statistic",
    "kstest",
    "fraction_to_date_full",
]
