"""
Robust distribution fitting examples.

Demonstrates the fitting engine that addresses the weaknesses of SciPy's
bare ``dist.fit``: a method switch (MLE / MoM / L-moments), multi-start
optimization, parameter standard errors, profile-likelihood confidence
intervals and goodness-of-fit diagnostics.
"""
import numpy as np
from actstats import actuarial

np.random.seed(42)

# ----------------------------------------------------------------------
# 1. Heavy-tailed severity (Pareto) -- where SciPy's fit is least reliable
# ----------------------------------------------------------------------
losses = actuarial.pareto(2.5, 1000).rvs(size=3000)

for method in ("mle", "mom", "lmoments"):
    res = actuarial.pareto.fit(losses, method=method, full_output=True)
    print(f"pareto [{method:9s}] -> {res.params}")

# Full diagnostics object
res = actuarial.pareto.fit(losses, full_output=True)
print()
print(res.summary())

# ----------------------------------------------------------------------
# 2. Generalized Pareto distribution (peaks-over-threshold tail)
# ----------------------------------------------------------------------
tail = actuarial.gpd(0.3, 500).rvs(size=2000)
res = actuarial.gpd.fit(tail, full_output=True)
print()
print(res.summary())

# ----------------------------------------------------------------------
# 3. Profile-likelihood confidence interval (reliable near boundaries)
# ----------------------------------------------------------------------
claims = actuarial.gamma(2.0, 3.0).rvs(size=500)
res = actuarial.gamma.fit(claims, full_output=True)
print()
print("gamma Wald CIs   :", res.confidence_intervals(method="wald"))
print("gamma profile CIs:", res.confidence_intervals(method="profile"))

prof = res.profile_likelihood("alpha")
print(f"profile-likelihood CI for alpha: {prof['ci']}")

# ----------------------------------------------------------------------
# 4. Model selection across candidate severity distributions via AIC
# ----------------------------------------------------------------------
data = actuarial.lognormal(8.0, 1.2).rvs(size=1500)
print()
for name in ("lognormal", "gamma", "weibull", "pareto"):
    res = getattr(actuarial, name).fit(data, full_output=True)
    print(f"{name:10s}  AIC={res.aic:12.2f}  BIC={res.bic:12.2f}  "
          f"KS={res.ks_stat:.4f}")

# Backward-compatible call: returns just the parameter tuple
print()
print("plain tuple result:", actuarial.lognormal.fit(data))
