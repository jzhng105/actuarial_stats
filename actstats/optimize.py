"""Small root-finding and optimisation helpers used by the fitting code."""

from __future__ import annotations

import math

__all__ = ["brentq", "bracket_root", "minimize_scalar"]

_GOLDEN = 0.3819660112501051


def bracket_root(func, x0, factor=2.0, max_iter=60):
    """Expand outwards from ``x0`` until ``func`` changes sign.

    Returns ``(lo, hi)`` with ``func(lo)`` and ``func(hi)`` of opposite sign.
    """
    lo = hi = float(x0)
    f_lo = f_hi = func(lo)
    if f_lo == 0.0:
        return lo, hi
    for _ in range(max_iter):
        lo = lo / factor if lo > 0.0 else lo * factor - 1.0
        f_lo = func(lo)
        if f_lo * f_hi <= 0.0:
            return lo, hi
        hi = hi * factor if hi > 0.0 else hi / factor + 1.0
        f_hi = func(hi)
        if f_lo * f_hi <= 0.0:
            return lo, hi
    raise RuntimeError("could not bracket a root")


def brentq(func, a, b, xtol=1e-13, rtol=8.9e-16, max_iter=200):
    """Brent's method: find a root of ``func`` inside ``[a, b]``.

    ``func(a)`` and ``func(b)`` must have opposite signs.
    """
    fa = func(a)
    fb = func(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("the root is not bracketed by [a, b]")

    c, fc = a, fa
    d = e = b - a
    for _ in range(max_iter):
        if fb * fc > 0.0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol = 2.0 * rtol * abs(b) + 0.5 * xtol
        m = 0.5 * (c - b)
        if abs(m) <= tol or fb == 0.0:
            return b
        if abs(e) < tol or abs(fa) <= abs(fb):
            d = e = m  # bisection
        else:
            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e = d
                d = p / q  # interpolation accepted
            else:
                d = e = m
        a, fa = b, fb
        b += d if abs(d) > tol else (tol if m > 0.0 else -tol)
        fb = func(b)
    return b


def minimize_scalar(func, a, b, xtol=1e-10, max_iter=200):
    """Golden-section / parabolic search for the minimum of ``func`` on ``[a, b]``."""
    x = w = v = a + _GOLDEN * (b - a)
    fx = fw = fv = func(x)
    d = e = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        tol1 = xtol * abs(x) + 1e-300
        tol2 = 2.0 * tol1
        if abs(x - mid) <= tol2 - 0.5 * (b - a):
            return x
        use_golden = True
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            prev_e, e = e, d
            if abs(p) < abs(0.5 * q * prev_e) and q * (a - x) < p < q * (b - x):
                d = p / q
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = tol1 if x < mid else -tol1
                use_golden = False
        if use_golden:
            e = (b - x) if x < mid else (a - x)
            d = _GOLDEN * e
        u = x + (d if abs(d) >= tol1 else (tol1 if d > 0.0 else -tol1))
        fu = func(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x
