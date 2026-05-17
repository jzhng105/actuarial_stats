"""
Robust distribution fitting for actuarial severity and frequency models.

SciPy's generic ``dist.fit`` relies on MLE through a single unguided
optimization run from hand-picked starting values. It fails silently or
returns poor fits surprisingly often for the distributions that matter most
in actuarial work: heavy-tailed severities (Pareto, GPD), bounded
distributions (beta near the boundaries) and shifted/scaled lognormals.

This module addresses those shortcomings with:

* a ``method`` switch -- maximum likelihood (``"mle"``), method of moments
  (``"mom"``) and L-moments (``"lmoments"``);
* multi-start MLE optimization seeded from MoM / L-moment / SciPy estimates
  so the optimizer does not get trapped or fail silently;
* proper parameter standard errors from the observed Fisher information;
* profile-likelihood confidence intervals (reliable for skewed / bounded
  parameters where Wald intervals are not);
* goodness-of-fit diagnostics (log-likelihood, AIC, BIC, KS, Anderson-Darling).

All estimation is performed directly in *actuarial* parameter space, so the
reported standard errors and intervals refer to the actuarial parameters the
caller actually uses -- no delta-method bookkeeping required.
"""

import warnings
from collections import namedtuple

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize, least_squares

INF = np.inf

_Meta = namedtuple("_Meta", ["param_names", "bounds", "discrete", "support"])

# Actuarial parameter metadata, keyed by actuarial distribution name.
# ``bounds`` are the admissible ranges for each actuarial parameter.
# ``support`` constrains the data: 'pos', 'unit', 'real' or 'count'.
_META = {
    "lognormal":         _Meta(["mu", "sigma"],  [(-INF, INF), (1e-12, INF)],       False, "pos"),
    "gamma":             _Meta(["alpha", "theta"], [(1e-12, INF), (1e-12, INF)],     False, "pos"),
    "weibull":           _Meta(["delta", "beta"], [(1e-12, INF), (1e-12, INF)],      False, "pos"),
    "pareto":            _Meta(["alpha", "beta"], [(1e-12, INF), (1e-12, INF)],      False, "pos"),
    "beta":              _Meta(["alpha", "beta"], [(1e-12, INF), (1e-12, INF)],      False, "unit"),
    "exponential":       _Meta(["beta"],          [(1e-12, INF)],                    False, "pos"),
    "normal":            _Meta(["mu", "sigma"],   [(-INF, INF), (1e-12, INF)],       False, "real"),
    "logistic":          _Meta(["mu", "s"],       [(-INF, INF), (1e-12, INF)],       False, "real"),
    "uniform":           _Meta(["a", "b"],        [(-INF, INF), (-INF, INF)],        False, "real"),
    "gpd":               _Meta(["xi", "beta"],    [(-INF, INF), (1e-12, INF)],       False, "pos"),
    "poisson":           _Meta(["lambda"],        [(1e-12, INF)],                    True,  "count"),
    "negative_binomial": _Meta(["r", "p"],        [(1e-9, INF), (1e-9, 1 - 1e-9)],   True,  "count"),
}

# Extra keyword arguments for the SciPy ``fit`` call (used only as one of the
# multi-start seeds). Severity distributions fix the location at zero.
_SCIPY_FIT_KW = {
    "lognormal": {"floc": 0}, "gamma": {"floc": 0}, "weibull": {"floc": 0},
    "pareto": {"floc": 0}, "exponential": {"floc": 0}, "gpd": {"floc": 0},
    "beta": {"floc": 0, "fscale": 1},
}

_METHOD_ALIASES = {
    "mle": "mle", "ml": "mle", "maximum_likelihood": "mle",
    "mom": "mom", "moments": "mom", "method_of_moments": "mom",
    "lmoments": "lmoments", "lmom": "lmoments", "l-moments": "lmoments",
    "lm": "lmoments",
}


# --------------------------------------------------------------------------
# Sample L-moments
# --------------------------------------------------------------------------
def sample_lmoments(data, nmom=4):
    """Return the first ``nmom`` sample L-moments of ``data``.

    L-moments are computed from unbiased probability-weighted moments and are
    far more robust to heavy tails and outliers than ordinary moments, which
    is why they are widely used for actuarial severity fitting.
    """
    x = np.sort(np.asarray(data, dtype=float))
    n = x.size
    nmom = int(min(nmom, n))
    if nmom < 1:
        raise ValueError("Need at least one observation for L-moments.")
    idx = np.arange(1, n + 1)
    b = np.zeros(nmom)
    b[0] = x.mean()
    for r in range(1, nmom):
        w = np.ones(n)
        for k in range(r):
            w = w * (idx - 1 - k) / (n - 1 - k)
        b[r] = np.sum(w * x) / n
    lmom = np.zeros(nmom)
    lmom[0] = b[0]
    if nmom > 1:
        lmom[1] = 2 * b[1] - b[0]
    if nmom > 2:
        lmom[2] = 6 * b[2] - 6 * b[1] + b[0]
    if nmom > 3:
        lmom[3] = 20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]
    return lmom


# --------------------------------------------------------------------------
# Data validation
# --------------------------------------------------------------------------
def _validate(name, data, meta):
    if data.size < 2:
        raise ValueError("Need at least 2 observations to fit a distribution.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Data contains NaN or infinite values.")
    if meta.support == "pos" and np.any(data <= 0):
        raise ValueError(
            f"'{name}' is supported on (0, inf) but the data minimum is "
            f"{data.min():.6g}. Remove non-positive values or pick another "
            f"distribution.")
    if meta.support == "unit" and (np.any(data <= 0) or np.any(data >= 1)):
        raise ValueError(
            f"'{name}' is supported on (0, 1) but the data range is "
            f"[{data.min():.6g}, {data.max():.6g}]. Rescale the data first.")
    if meta.support == "count" and (
            np.any(data < 0) or np.any(data != np.round(data))):
        raise ValueError(f"'{name}' requires non-negative integer counts.")


# --------------------------------------------------------------------------
# Method of moments
# --------------------------------------------------------------------------
def _mom_estimate(name, data):
    """Closed-form method-of-moments estimate in actuarial parameters."""
    m = float(np.mean(data))
    v = max(float(np.var(data)), 1e-12)

    if name == "lognormal":
        s2 = np.log1p(v / m ** 2)
        return (np.log(m) - s2 / 2.0, np.sqrt(s2))
    if name == "gamma":
        return (m * m / v, v / m)
    if name == "weibull":
        from scipy.special import gamma as gammafn
        from scipy.optimize import brentq
        cv2 = v / m ** 2

        def f(d):
            return gammafn(1 + 2 / d) / gammafn(1 + 1 / d) ** 2 - 1 - cv2

        try:
            delta = brentq(f, 0.05, 50.0)
        except Exception:
            delta = 1.0
        return (delta, m / gammafn(1 + 1 / delta))
    if name == "pareto":
        cv2 = v / m ** 2
        alpha = 2 * cv2 / (cv2 - 1) if cv2 > 1.001 else 10.0
        return (alpha, m * (alpha - 1))
    if name == "beta":
        t = max(m * (1 - m) / v - 1, 1e-6)
        return (m * t, (1 - m) * t)
    if name == "exponential":
        return (m,)
    if name == "normal":
        return (m, np.sqrt(v))
    if name == "logistic":
        return (m, np.sqrt(3 * v) / np.pi)
    if name == "uniform":
        hw = np.sqrt(3 * v)
        return (m - hw, m + hw)
    if name == "gpd":
        cv2 = v / m ** 2
        xi = min(0.5 * (1 - 1 / cv2), 0.49)
        return (xi, m * (1 - xi))
    if name == "poisson":
        return (m,)
    if name == "negative_binomial":
        if v > m:
            return (m * m / (v - m), m / v)
        return (max(m, 1.0), 0.5)
    raise ValueError(f"Unsupported distribution: {name}")


# --------------------------------------------------------------------------
# L-moments
# --------------------------------------------------------------------------
def _lmom_numeric(scipy_dist, to_scipy, meta, x0, target_l1, target_l2):
    """Generic L-moment matching via numerical integration of the quantile
    function. Used when no closed-form L-moment estimator is available."""
    nodes, gw = np.polynomial.legendre.leggauss(128)
    u = 0.5 * (nodes + 1.0)
    gw = 0.5 * gw
    p1 = 2 * u - 1.0

    def resid(theta):
        try:
            frozen = scipy_dist(*to_scipy(*theta))
            q = frozen.ppf(u)
            if not np.all(np.isfinite(q)):
                return [1e8, 1e8]
            return [np.sum(gw * q) - target_l1,
                    np.sum(gw * q * p1) - target_l2]
        except Exception:
            return [1e8, 1e8]

    lo = [b[0] for b in meta.bounds]
    hi = [b[1] for b in meta.bounds]
    res = least_squares(resid, np.asarray(x0, float), bounds=(lo, hi))
    return tuple(res.x)


def _lmom_estimate(name, data, scipy_dist, to_scipy, meta):
    """L-moment estimate in actuarial parameters."""
    if meta.discrete:
        raise ValueError(
            "L-moment estimation is not defined for discrete distributions; "
            "use method='mle' or method='mom'.")
    lmom = sample_lmoments(data, 4)
    l1, l2 = float(lmom[0]), float(lmom[1])

    if name == "normal":
        return (l1, l2 * np.sqrt(np.pi))
    if name == "logistic":
        return (l1, l2)
    if name == "exponential":
        return (l1,)
    if name == "uniform":
        return (l1 - 3 * l2, l1 + 3 * l2)
    if name == "lognormal":
        loglmom = sample_lmoments(np.log(data), 2)
        return (float(loglmom[0]), float(loglmom[1]) * np.sqrt(np.pi))
    if name == "gamma":
        tau = l2 / l1
        if tau < 0.5:
            z = np.pi * tau * tau
            alpha = (1 - 0.3080 * z) / (z - 0.05812 * z ** 2 + 0.01765 * z ** 3)
        else:
            z = 1 - tau
            alpha = (0.7213 * z - 0.5947 * z ** 2) / (
                1 - 2.1817 * z + 1.2113 * z ** 2)
        return (alpha, l1 / alpha)
    if name == "gpd":
        t3 = float(lmom[2]) / l2
        k = (1 - 3 * t3) / (1 + t3)
        return (-k, l1 * (1 + k))
    if name == "pareto":
        t3 = float(lmom[2]) / l2
        k = (1 - 3 * t3) / (1 + t3)
        if k < -1e-6:               # heavy-tailed: closed form applies
            return (-1.0 / k, -l1 * (1 + k) / k)
        return _lmom_numeric(scipy_dist, to_scipy, meta,
                             _mom_estimate(name, data), l1, l2)
    if name in ("weibull", "beta"):
        return _lmom_numeric(scipy_dist, to_scipy, meta,
                             _mom_estimate(name, data), l1, l2)
    raise ValueError(f"Unsupported distribution: {name}")


# --------------------------------------------------------------------------
# Numerical helpers
# --------------------------------------------------------------------------
def _numerical_hessian(func, x):
    """Central-difference Hessian of a scalar function at ``x``."""
    x = np.asarray(x, float)
    n = x.size
    step = np.maximum(np.abs(x), 1e-3) * 1e-4
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ei[i] = step[i]
            ej = np.zeros(n); ej[j] = step[j]
            fpp = func(x + ei + ej)
            fpm = func(x + ei - ej)
            fmp = func(x - ei + ej)
            fmm = func(x - ei - ej)
            val = (fpp - fpm - fmp + fmm) / (4 * step[i] * step[j])
            hess[i, j] = hess[j, i] = val
    return hess


def _covariance_from_hessian(hess):
    """Invert the observed-information matrix to a covariance matrix."""
    if not np.all(np.isfinite(hess)):
        return None
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(cov)):
        return None
    return cov


def _anderson_darling(data, frozen):
    """Anderson-Darling statistic for a fully specified continuous fit.

    The AD statistic weights the tails more heavily than KS, so it is the more
    informative goodness-of-fit measure for heavy-tailed severity models.
    """
    x = np.sort(np.asarray(data, float))
    n = x.size
    cdf = np.clip(frozen.cdf(x), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    s = np.sum((2 * i - 1) * (np.log(cdf) + np.log(1 - cdf[::-1])))
    return -n - s / n


# --------------------------------------------------------------------------
# Fit result container
# --------------------------------------------------------------------------
class FitResult:
    """Outcome of a distribution fit, with diagnostics and inference tools."""

    def __init__(self, name, method, scipy_dist, to_scipy, params, data, meta,
                 loglik, cov=None, success=True, message="",
                 n_starts=0, optimizer="", nll=None):
        self.name = name
        self.method = method
        self.param_names = list(meta.param_names)
        self.params_tuple = tuple(float(p) for p in params)
        self.params = dict(zip(self.param_names, self.params_tuple))
        self.scipy_params = tuple(to_scipy(*self.params_tuple))
        self.n = int(np.asarray(data).size)
        self.k = len(self.param_names)
        self.loglik = float(loglik)
        self.aic = 2 * self.k - 2 * self.loglik
        self.bic = self.k * np.log(self.n) - 2 * self.loglik
        # small-sample corrected AIC
        denom = self.n - self.k - 1
        self.aicc = (self.aic + 2 * self.k * (self.k + 1) / denom
                     if denom > 0 else np.inf)
        self.cov = cov
        if cov is not None:
            diag = np.diag(cov)
            self.se_array = np.where(diag > 0, np.sqrt(np.abs(diag)), np.nan)
        else:
            self.se_array = np.full(self.k, np.nan)
        self.se = dict(zip(self.param_names, self.se_array))
        self.success = bool(success)
        self.message = message
        self.n_starts = n_starts
        self.optimizer = optimizer
        self._scipy_dist = scipy_dist
        self._to_scipy = to_scipy
        self._meta = meta
        self._data = np.asarray(data, float)
        self._nll = nll
        self.ks_stat = np.nan
        self.ks_pvalue = np.nan
        self.ad_stat = np.nan
        self._compute_gof()

    # -- diagnostics -------------------------------------------------------
    @property
    def dist(self):
        """A frozen SciPy distribution with the fitted parameters."""
        return self._scipy_dist(*self.scipy_params)

    def _compute_gof(self):
        try:
            frozen = self.dist
            ks = stats.kstest(self._data, frozen.cdf)
            self.ks_stat = float(ks.statistic)
            self.ks_pvalue = float(ks.pvalue)
            if not self._meta.discrete:
                self.ad_stat = float(_anderson_darling(self._data, frozen))
        except Exception:
            pass

    def _param_index(self, param):
        if isinstance(param, str):
            if param not in self.param_names:
                raise KeyError(
                    f"Unknown parameter '{param}'. Choose from "
                    f"{self.param_names}.")
            return self.param_names.index(param)
        idx = int(param)
        if not 0 <= idx < self.k:
            raise IndexError(f"Parameter index {idx} out of range.")
        return idx

    # -- inference ---------------------------------------------------------
    def confidence_intervals(self, alpha=0.05, method="wald"):
        """Confidence intervals for every fitted parameter.

        ``method='wald'`` uses the standard-error normal approximation.
        ``method='profile'`` inverts the profile likelihood -- slower but
        reliable for skewed or near-boundary parameters.
        """
        z = stats.norm.ppf(1 - alpha / 2)
        out = {}
        if method == "wald":
            for nm in self.param_names:
                s = self.se.get(nm, np.nan)
                p = self.params[nm]
                if s is None or not np.isfinite(s):
                    out[nm] = (np.nan, np.nan)
                else:
                    out[nm] = (p - z * s, p + z * s)
        elif method == "profile":
            for nm in self.param_names:
                out[nm] = self.profile_likelihood(nm, alpha=alpha)["ci"]
        else:
            raise ValueError("method must be 'wald' or 'profile'.")
        return out

    def profile_likelihood(self, param, num=41, span=4.0, alpha=0.05):
        """Profile-likelihood curve and confidence interval for one parameter.

        Returns a dict with the parameter grid, the profile log-likelihood,
        the likelihood-ratio deviance, the chi-square threshold and the
        resulting confidence interval ``ci``.
        """
        if self.method != "mle" or self._nll is None:
            raise RuntimeError(
                "Profile likelihood is only available for method='mle'.")
        idx = self._param_index(param)
        mle = np.array(self.params_tuple, float)
        s = self.se_array[idx]
        if not (np.isfinite(s) and s > 0):
            s = abs(mle[idx]) * 0.25 + 1e-3
        lo_b, hi_b = self._meta.bounds[idx]
        grid = np.linspace(mle[idx] - span * s, mle[idx] + span * s, num)
        lo_b = lo_b if np.isfinite(lo_b) else -INF
        hi_b = hi_b if np.isfinite(hi_b) else INF
        grid = grid[(grid > lo_b) & (grid < hi_b)]
        grid = np.unique(np.append(grid, mle[idx]))
        others = [j for j in range(self.k) if j != idx]

        prof_ll = []
        for g in grid:
            if not others:
                prof_ll.append(-self._nll([g]))
                continue

            def obj(free, g=g):
                full = np.empty(self.k)
                full[idx] = g
                for j, v in zip(others, free):
                    full[j] = v
                return self._nll(full)

            x0 = mle[others]
            best = obj(x0)
            try:
                res = minimize(obj, x0, method="Nelder-Mead")
                if np.isfinite(res.fun) and res.fun < best:
                    best = res.fun
            except Exception:
                pass
            prof_ll.append(-best)

        prof_ll = np.array(prof_ll)
        llmax = prof_ll.max()
        deviance = 2 * (llmax - prof_ll)
        threshold = stats.chi2.ppf(1 - alpha, 1)
        ci = _crossing(grid, deviance, threshold)
        return {"param": self.param_names[idx], "grid": grid,
                "loglik": prof_ll, "deviance": deviance,
                "threshold": threshold, "ci": ci, "alpha": alpha}

    # -- reporting ---------------------------------------------------------
    def summary(self, alpha=0.05):
        """Human-readable summary of the fit."""
        lines = []
        lines.append(f"Distribution : {self.name}")
        lines.append(f"Method       : {self.method.upper()}   "
                     f"n = {self.n}   success = {self.success}")
        if self.message:
            lines.append(f"Note         : {self.message}")
        lines.append("-" * 60)
        header = f"{'parameter':<14}{'estimate':>14}{'std.err':>14}"
        if self.method == "mle":
            header += f"{'  ' + str(int((1-alpha)*100)) + '% CI':>18}"
        lines.append(header)
        ci = (self.confidence_intervals(alpha) if self.method == "mle"
              else None)
        for nm in self.param_names:
            est = self.params[nm]
            se = self.se.get(nm, np.nan)
            se_txt = f"{se:>14.6g}" if np.isfinite(se) else f"{'n/a':>14}"
            row = f"{nm:<14}{est:>14.6g}{se_txt}"
            if ci is not None:
                lo, hi = ci[nm]
                if np.isfinite(lo) and np.isfinite(hi):
                    row += f"   [{lo:.5g}, {hi:.5g}]"
                else:
                    row += "   [n/a]"
            lines.append(row)
        lines.append("-" * 60)
        lines.append(f"log-likelihood : {self.loglik:.4f}")
        lines.append(f"AIC : {self.aic:.4f}    AICc : {self.aicc:.4f}    "
                     f"BIC : {self.bic:.4f}")
        gof = f"KS stat : {self.ks_stat:.4f} (p = {self.ks_pvalue:.4f})"
        if np.isfinite(self.ad_stat):
            gof += f"    AD stat : {self.ad_stat:.4f}"
        lines.append(gof)
        if self.method == "mle":
            lines.append(f"starts tried : {self.n_starts}    "
                         f"optimizer : {self.optimizer}")
        return "\n".join(lines)

    def __repr__(self):
        params = ", ".join(f"{k}={v:.5g}"
                            for k, v in self.params.items())
        return (f"FitResult({self.name}, method={self.method}, {params}, "
                f"loglik={self.loglik:.3f}, aic={self.aic:.2f})")


def _crossing(grid, deviance, threshold):
    """Locate where the deviance curve crosses ``threshold`` around its min."""
    imin = int(np.argmin(deviance))
    lo = hi = np.nan

    def interp(x0, y0, x1, y1):
        if y1 == y0:
            return x0
        return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)

    for i in range(imin, 0, -1):
        if (deviance[i] - threshold) * (deviance[i - 1] - threshold) <= 0:
            lo = interp(grid[i - 1], deviance[i - 1], grid[i], deviance[i])
            break
    for i in range(imin, len(grid) - 1):
        if (deviance[i] - threshold) * (deviance[i + 1] - threshold) <= 0:
            hi = interp(grid[i], deviance[i], grid[i + 1], deviance[i + 1])
            break
    return (lo, hi)


# --------------------------------------------------------------------------
# Fitters
# --------------------------------------------------------------------------
def _make_nll(scipy_dist, to_scipy, data, meta):
    """Build the negative log-likelihood in actuarial parameter space."""
    logf = scipy_dist.logpmf if meta.discrete else scipy_dist.logpdf
    bounds = meta.bounds

    def nll(theta):
        theta = np.asarray(theta, float)
        for v, (lo, hi) in zip(theta, bounds):
            if not np.isfinite(v) or v < lo or v > hi:
                return INF
        try:
            total = -np.sum(logf(data, *to_scipy(*theta)))
        except Exception:
            return INF
        return total if np.isfinite(total) else INF

    return nll


def _candidate_starts(name, scipy_dist, to_scipy, from_scipy, data, meta):
    """Collect a diverse set of starting points for multi-start MLE."""
    starts = []

    def add(point):
        if point is None:
            return
        arr = np.asarray(point, float)
        if arr.size == len(meta.param_names) and np.all(np.isfinite(arr)):
            starts.append(arr)

    try:
        add(_mom_estimate(name, data))
    except Exception:
        pass
    try:
        add(_lmom_estimate(name, data, scipy_dist, to_scipy, meta))
    except Exception:
        pass
    if not meta.discrete:
        try:
            kw = _SCIPY_FIT_KW.get(name, {})
            add(from_scipy(scipy_dist.fit(data, **kw)))
        except Exception:
            pass

    # Perturb the base seeds so the optimizer explores more of the surface.
    perturbed = []
    for base in list(starts):
        for factor in (0.5, 2.0):
            cand = base.copy()
            for j, (lo, hi) in enumerate(meta.bounds):
                if lo >= 0:                       # positive-scale parameter
                    cand[j] = base[j] * factor
            perturbed.append(cand)
    starts.extend(perturbed)
    return starts


def _fit_mle(name, scipy_dist, to_scipy, from_scipy, data, meta):
    # Uniform MLE is the sample range -- the likelihood is not differentiable,
    # so it is handled in closed form.
    if name == "uniform":
        a, b = float(np.min(data)), float(np.max(data))
        nll = _make_nll(scipy_dist, to_scipy, data, meta)
        loglik = -nll((a, b))
        return FitResult(name, "mle", scipy_dist, to_scipy, (a, b), data, meta,
                         loglik, cov=None, success=True,
                         message="closed-form MLE (sample range)",
                         n_starts=1, optimizer="closed-form", nll=nll)

    nll = _make_nll(scipy_dist, to_scipy, data, meta)
    starts = _candidate_starts(name, scipy_dist, to_scipy, from_scipy,
                               data, meta)
    if not starts:                                 # last-resort generic seed
        starts = [np.ones(len(meta.param_names))]

    finite_bounds = [(None if not np.isfinite(lo) else lo,
                      None if not np.isfinite(hi) else hi)
                     for lo, hi in meta.bounds]

    best_val, best_x, best_opt = INF, None, ""
    n_tried = 0
    for x0 in starts:
        if not np.isfinite(nll(x0)):
            continue
        n_tried += 1
        if np.isfinite(nll(x0)) and nll(x0) < best_val:
            best_val, best_x, best_opt = nll(x0), np.asarray(x0, float), "seed"
        for opt in ("Nelder-Mead", "L-BFGS-B"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if opt == "L-BFGS-B":
                        res = minimize(nll, x0, method=opt,
                                       bounds=finite_bounds)
                    else:
                        res = minimize(nll, x0, method=opt)
            except Exception:
                continue
            val = nll(res.x)
            if np.isfinite(val) and val < best_val:
                best_val, best_x, best_opt = val, np.asarray(res.x, float), opt

    success = best_x is not None and np.isfinite(best_val)
    if not success:
        # Fall back to the best available moment estimate so the caller still
        # receives usable parameters instead of a silent failure.
        try:
            best_x = np.asarray(_mom_estimate(name, data), float)
        except Exception:
            best_x = np.ones(len(meta.param_names))
        message = ("MLE optimization did not converge from any start; "
                   "returning method-of-moments estimate instead.")
        loglik = -nll(best_x)
        return FitResult(name, "mle", scipy_dist, to_scipy, best_x, data, meta,
                         loglik if np.isfinite(loglik) else -INF,
                         cov=None, success=False, message=message,
                         n_starts=n_tried, optimizer="failed", nll=nll)

    hess = _numerical_hessian(nll, best_x)
    cov = _covariance_from_hessian(hess)
    message = ""
    if cov is None:
        message = ("Observed information matrix was not invertible; standard "
                   "errors are unavailable (try profile_likelihood).")
    return FitResult(name, "mle", scipy_dist, to_scipy, best_x, data, meta,
                     -best_val, cov=cov, success=True, message=message,
                     n_starts=n_tried, optimizer=best_opt, nll=nll)


def _fit_moment_based(method, name, scipy_dist, to_scipy, data, meta):
    if method == "mom":
        params = _mom_estimate(name, data)
    else:
        params = _lmom_estimate(name, data, scipy_dist, to_scipy, meta)
    success = np.all(np.isfinite(np.asarray(params, float)))
    nll = _make_nll(scipy_dist, to_scipy, data, meta)
    loglik = -nll(params)
    message = ""
    if not np.isfinite(loglik):
        # The moment estimate is well defined but the implied support does
        # not cover every observation, so likelihood-based metrics are not.
        message = ("Some observations fall outside the support implied by "
                   "the moment estimate; log-likelihood/AIC/BIC are not "
                   "available. Use method='mle' for likelihood inference.")
    return FitResult(name, method, scipy_dist, to_scipy, params, data, meta,
                     loglik if np.isfinite(loglik) else -INF,
                     cov=None, success=success, message=message,
                     n_starts=0, optimizer="closed-form")


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def robust_fit(name, scipy_dist, to_scipy, from_scipy, data,
               method="mle", alpha=0.05):
    """Fit an actuarial distribution robustly.

    Parameters
    ----------
    name : str
        Actuarial distribution name (e.g. ``"lognormal"``, ``"gpd"``).
    scipy_dist : scipy.stats distribution
        The underlying SciPy distribution object.
    to_scipy, from_scipy : callable
        Converters between actuarial and SciPy parameter conventions.
    data : array-like
        Observed sample.
    method : {"mle", "mom", "lmoments"}
        Estimation method. MLE uses multi-start optimization and reports
        standard errors; MoM and L-moments are closed-form / fast.
    alpha : float
        Significance level used for the summary's confidence intervals.

    Returns
    -------
    FitResult
    """
    if name not in _META:
        raise ValueError(f"Robust fitting is not supported for '{name}'.")
    meta = _META[name]
    data = np.asarray(data, dtype=float).ravel()
    _validate(name, data, meta)

    key = _METHOD_ALIASES.get(str(method).lower())
    if key is None:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'mle', 'mom' or 'lmoments'.")

    if key == "mle":
        return _fit_mle(name, scipy_dist, to_scipy, from_scipy, data, meta)
    return _fit_moment_based(key, name, scipy_dist, to_scipy, data, meta)
