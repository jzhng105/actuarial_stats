"""The container returned by ``rvs`` and friends.

:class:`Sample` is a plain ``list`` subclass, so it prints, slices, pickles and
iterates like any Python sequence, while still offering the handful of array
methods simulation code reaches for (``mean``, ``var``, ``std``, ...).  That
keeps ``dist.rvs(size=n).mean()`` working without a NumPy array.
"""

from __future__ import annotations

import math
from bisect import bisect_left

__all__ = ["Sample"]


class Sample(list):
    """A list of simulated values with the usual summary statistics."""

    __slots__ = ()

    # -- summaries --------------------------------------------------------
    def sum(self):
        """Total of the sample."""
        return math.fsum(self)

    def mean(self):
        """Arithmetic mean."""
        n = len(self)
        if n == 0:
            raise ValueError("mean of an empty sample")
        return math.fsum(self) / n

    def var(self, ddof=0):
        """Variance; ``ddof=0`` (the population form) matches NumPy's default."""
        n = len(self)
        if n - ddof <= 0:
            raise ValueError("not enough observations for the requested ddof")
        mu = math.fsum(self) / n
        return math.fsum((x - mu) ** 2 for x in self) / (n - ddof)

    def std(self, ddof=0):
        """Standard deviation."""
        return math.sqrt(self.var(ddof))

    def min(self):
        """Smallest value."""
        return min(self)

    def max(self):
        """Largest value."""
        return max(self)

    def quantile(self, q):
        """Empirical quantile using linear interpolation (NumPy's default)."""
        if not self:
            raise ValueError("quantile of an empty sample")
        if isinstance(q, (int, float)):
            return self._quantile(sorted(self), float(q))
        ordered = sorted(self)
        return Sample(self._quantile(ordered, float(p)) for p in q)

    def percentile(self, p):
        """Empirical percentile; ``p`` is on a 0-100 scale."""
        if isinstance(p, (int, float)):
            return self.quantile(float(p) / 100.0)
        return self.quantile([float(v) / 100.0 for v in p])

    def median(self):
        """Empirical median."""
        return self.quantile(0.5)

    @staticmethod
    def _quantile(ordered, q):
        if not 0.0 <= q <= 1.0:
            raise ValueError("quantiles must lie in [0, 1]")
        pos = q * (len(ordered) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        frac = pos - lo
        return ordered[lo] + frac * (ordered[hi] - ordered[lo])

    def ecdf(self, x):
        """Empirical CDF evaluated at ``x``."""
        ordered = sorted(self)
        return bisect_left(ordered, x, lo=0, hi=len(ordered)) / len(ordered)

    # -- conversions ------------------------------------------------------
    def tolist(self):
        """Return the values as a plain ``list``."""
        return list(self)

    def to_numpy(self, dtype=None):
        """Convert to a NumPy array.

        ``actstats`` never needs NumPy, but this bridge is here for callers who
        want to hand a simulation off to the wider scientific stack.
        """
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "to_numpy() requires NumPy, which actstats does not install"
            ) from exc
        return np.asarray(self, dtype=dtype)

    # -- arithmetic -------------------------------------------------------
    def _binary(self, other, op):
        if isinstance(other, (int, float)):
            return Sample(op(x, other) for x in self)
        other = list(other)
        if len(other) != len(self):
            raise ValueError("operands have different lengths")
        return Sample(op(x, y) for x, y in zip(self, other))

    def __add__(self, other):
        return self._binary(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self._binary(other, lambda x, y: y + x)

    def __sub__(self, other):
        return self._binary(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self._binary(other, lambda x, y: y - x)

    def __mul__(self, other):
        return self._binary(other, lambda x, y: x * y)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, lambda x, y: x / y)

    def __rtruediv__(self, other):
        return self._binary(other, lambda x, y: y / x)

    def __pow__(self, other):
        return self._binary(other, lambda x, y: x ** y)

    def __neg__(self):
        return Sample(-x for x in self)

    def __repr__(self):
        if len(self) > 8:
            head = ", ".join(repr(v) for v in self[:4])
            tail = ", ".join(repr(v) for v in self[-2:])
            return f"Sample([{head}, ..., {tail}], n={len(self)})"
        return f"Sample({list.__repr__(self)})"
