# actuarial_stats
**Of the Actuary, By the Actuary, For the Actuary**

`actstats` is a Python library that gives you the standard statistical
distributions in the parameterisations actuaries actually use — and it does so
in **pure Python, with no dependencies at all**.

```bash
pip install actstats     # installs nothing else
```

```python
import actstats
from actstats import actuarial

actstats.seed(42)

severity = actuarial.lognormal(9.0, 1.3)   # mu, sigma of log X
losses   = severity.rvs(size=100_000)      # a Sample: .mean(), .var(), .quantile()

severity.ppf(0.995)                        # 230,632  (99.5% VaR)
severity.isf(1e-9)                         # 19,720,103  (accurate far tail)
mu, sigma = actuarial.lognormal.fit(losses)
statistic, p_value = actuarial.lognormal(mu, sigma).kstest(losses)
```

---

## 🔢 Distributions

| Distribution             | Actuarial parameters | Mean                        | Variance                              | SciPy equivalent             |
| ------------------------ | -------------------- | --------------------------- | ------------------------------------- | ---------------------------- |
| `lognormal`              | (μ, σ)               | exp(μ + σ²/2)               | (exp(σ²) − 1)·exp(2μ + σ²)            | `lognorm(s=σ, scale=exp(μ))` |
| `gamma`                  | (α, θ)               | α·θ                         | α·θ²                                  | `gamma(a=α, scale=θ)`        |
| `weibull`                | (δ, β)               | β·Γ(1 + 1/δ)                | β²·[Γ(1 + 2/δ) − Γ(1 + 1/δ)²]         | `weibull_min(c=δ, scale=β)`  |
| `pareto`                 | (α, β)               | β / (α − 1),  α > 1         | β²·α / [(α − 1)²·(α − 2)],  α > 2     | `lomax(c=α, scale=β)`        |
| `beta`                   | (α, β)               | α / (α + β)                 | αβ / [(α + β)²·(α + β + 1)]           | `beta(a=α, b=β)`             |
| `poisson`                | (λ)                  | λ                           | λ                                     | `poisson(mu=λ)`              |
| `negative_binomial`      | (r, p)               | r·(1 − p) / p               | r·(1 − p) / p²                        | `nbinom(n=r, p=p)`           |
| `normal`                 | (μ, σ)               | μ                           | σ²                                    | `norm(loc=μ, scale=σ)`       |
| `logistic`               | (μ, s)               | μ                           | (π² / 3)·s²                           | `logistic(loc=μ, scale=s)`   |
| `exponential`            | (β)                  | β                           | β²                                    | `expon(scale=β)`             |
| `uniform`                | (a, b)               | (a + b) / 2                 | (b − a)² / 12                         | `uniform(loc=a, scale=b−a)`  |
| `nonhomogeneous_poisson` | (λ₀, α, ϕ, T)        | λ₀·T + (λ₀·α / 2π)·[cos(ϕ) − cos(2πT + ϕ)] | — | custom `NonhomogeneousPoisson` |

The SciPy column is there for reference and migration; nothing in `actstats`
imports it. Note that `pareto` is the two-parameter Pareto of the actuarial
literature (supported on `[0, ∞)`), which is SciPy's `lomax`, not its `pareto`.

## 📐 What each distribution gives you

| | |
| --- | --- |
| Densities | `pdf` / `pmf`, `logpdf` / `logpmf` |
| Distribution functions | `cdf`, `logcdf`, `sf`, `logsf` |
| Quantiles | `ppf`, `isf`, `median`, `interval(confidence)` |
| Moments | `mean`, `var`, `std`, `moment(n)`, `stats("mvsk")` |
| Simulation | `rvs(size=...)`, `random_state=...` |
| Inference | `fit(data)`, `nnlf(data)`, `ks_test(data)`, `kstest(data)` |

Every function accepts a scalar or any iterable, and returns a float or a
`Sample` accordingly. Both call styles work, as before:

```python
dist = actuarial.lognormal(0.5, 0.2)   # frozen
dist.ppf(0.5)

actuarial.lognormal.rvs(0.5, 0.2, 1000)   # unfrozen: parameters first
actuarial.lognormal.ppf(0.5, 0.5, 0.2)    # unfrozen: parameters last
```

### Working in the tail

For return periods, ask for the tail directly instead of subtracting from one.
`sf` and `isf` keep their significant digits where `1 - cdf(x)` has already
collapsed to zero:

```python
dist = actuarial.normal(0.0, 1.0)
dist.cdf(35.0)       # 1.0 exactly, so 1 - cdf() would be 0
dist.sf(35.0)        # 1.1249107064725534e-268

dist.isf(1e-12)      # 7.034483825301132  <- the exact 1-in-1e12 quantile
dist.ppf(1 - 1e-12)  # 7.034486910047836  <- 1 - 1e-12 lost digits before the call

severity.isf(1e-6)   # the 1-in-a-million loss
```

### Samples

`rvs` returns a `Sample`, a `list` subclass that carries the summary methods
simulation code reaches for:

```python
losses = actuarial.gamma(2.0, 1500.0).rvs(size=50_000)
losses.mean(), losses.std(), losses.var()
losses.quantile(0.995), losses.percentile([90, 99])
losses.min(), losses.max(), losses.sum()
losses.to_numpy()          # optional bridge, only if you have NumPy
```

Arithmetic on a `Sample` is element-wise (`losses * 1.05` inflates every
value), not the `list` operations of the same name.

### Reproducibility

```python
actstats.seed(42)                     # the module level stream
own = actstats.RandomState(7)         # or an independent one
actuarial.poisson(3.7).rvs(size=100, random_state=own)
```

A batch consumes exactly the same underlying uniforms as the equivalent single
draws, so `[d.rvs() for _ in range(n)]` and `d.rvs(size=n)` return identical
values from the same seed. Batching a simulation changes its speed, never its
results.

## 🧪 Fitting

`fit` returns maximum-likelihood estimates as a tuple, in the same order the
constructor takes them, with the location held at zero:

```python
alpha, theta = actuarial.gamma.fit(losses)
fitted = actuarial.gamma(alpha, theta)
fitted.kstest(losses)          # (statistic, p-value)
```

| Distribution | Method |
| --- | --- |
| `lognormal`, `normal`, `exponential`, `uniform`, `poisson` | closed form |
| `gamma` | Newton on `log α − ψ(α) = log x̄ − mean(log x)`, from Minka's start |
| `weibull` | Brent on the profile score for the shape |
| `pareto` | profile likelihood in the scale, closed form for the shape |
| `beta` | two-dimensional Newton on the digamma system |
| `logistic` | Fisher scoring |
| `negative_binomial` | MLE by default, or `method="moments"` |

These agree with SciPy's own MLEs to the digits printed by
`examples/examples.py`. Fitting a negative binomial to data that is not
over-dispersed now raises a clear error rather than returning nonsense.

---

## ⚡ Removing the NumPy and SciPy dependency

Earlier releases wrapped `scipy.stats` and `numpy.random`. Everything is now
implemented directly, in four layers:

1. **`actstats.special`** — the special functions the distribution functions
   need: the regularised incomplete gamma (`gammainc`, `gammaincc`) by series
   and continued fraction, the regularised incomplete beta (`betainc`) by
   Lentz's continued fraction, the normal integral from `math.erfc`, and
   `digamma`/`trigamma` by asymptotic expansion with recurrence shifting.
   Their inverses use Newton's method safeguarded by bisection, with the step
   taken in `log x` where the density over- or underflows, and separate
   upper-tail entry points (`gammainccinv`, `betainccinv`) so far quantiles
   keep their precision.
2. **`actstats.rng`** — variate generation on `random.Random`: Box-Muller with
   a cached spare for normals, Marsaglia-Tsang for the gamma, Jöhnk for small
   beta shapes, Hörmann's PTRS for large-mean Poissons, and a gamma-Poisson
   mixture for the negative binomial. These are the same algorithms NumPy
   uses, so the statistical quality matches; only the bit stream differs.
3. **`actstats.distributions`** — one class per distribution, holding scalar
   kernels that the base class vectorises. Frozen distributions cache whatever
   their parameters imply (log-normalising constants, and for lattice
   distributions a table of cumulative probabilities built on first use, which
   turns each later draw into one uniform plus a binary search).
4. **`actstats.optimize`** — Brent root-finding and golden-section
   minimisation for the maximum-likelihood fits.

Accuracy is checked against SciPy: across wide parameter sweeps the special
functions agree to about 1e-11 relative, and every distribution function,
quantile and moment in `tests/reference_values.py` matches to 1e-9 or better.
The samplers pass Kolmogorov-Smirnov and chi-square tests against their own
distribution functions and two-sample tests against NumPy's generators.

### Is it faster?

For everything except large vectorised batches, yes — often by a lot, because
SciPy's frozen distributions pay a large fixed cost per call. From
`benchmarks/benchmark.py` (CPython 3.11, SciPy 1.17, NumPy 2.4):

| benchmark | actstats | scipy/numpy | |
| --- | ---: | ---: | --- |
| `lognormal.pdf(x)` scalar | 1.98 µs | 127.69 µs | **65× faster** |
| `lognormal.cdf(x)` scalar | 1.89 µs | 75.23 µs | **40× faster** |
| `poisson.pmf(4)` scalar | 1.92 µs | 67.65 µs | **35× faster** |
| `gamma.ppf(0.995)` scalar | 35.46 µs | 82.89 µs | **2.3× faster** |
| `rvs(size=1)` | 2.62 µs | 38.44 µs | **15× faster** |
| `rvs(size=100)` | 27.5 µs | 49.0 µs | **1.8× faster** |
| aggregate loss, 10k trials (claim-by-claim loop) | 81 ms | 807 ms | **10× faster** |
| `import` cost | 17 ms | 931 ms | **54× faster** |
| `rvs(size=1000)` | 249 µs | 84 µs | 3.0× slower |
| `rvs(size=100000)` | 26.8 ms | 3.7 ms | 7.4× slower |
| aggregate loss, 10k trials (fully vectorised NumPy) | 81 ms | 2.2 ms | 37× slower |
| `gamma.fit(5000 obs)` | 1.03 ms | 0.35 ms | 2.9× slower |

The honest summary: **pure Python wins wherever per-call overhead dominates**
— scalar evaluation, small batches, and the claim-by-claim loops that
frequency-severity models are naturally written as — and **loses to vectorised
NumPy on large array sampling**, where the C loop is unbeatable. If your
simulation is one big `rvs(size=1_000_000)`, keep NumPy. If it is a nested
loop over policies, layers and trials, or a service that must start quickly,
this is the faster and far lighter option.

The dependency saving is unconditional: no 100 MB of wheels, no ABI or
compiler concerns, nothing to pin, and an import that costs 17 ms instead of
most of a second — which matters for CLI tools, serverless functions and CI.

## 🔁 Migrating from 0.1.x

Existing scripts keep working. Specifically:

- `actuarial.<name>(...)`, frozen and unfrozen calls, `fit`, and `ks_test`
  behave as before.
- `np_rvs` is kept as an alias of `rvs`. Both are now the same pure-Python
  sampler, so prefer `rvs`.
- `rvs` returns a `Sample` rather than a NumPy array. `.mean()`, `.var()`,
  `.std()`, `.min()` and `.max()` work as they did; call `.to_numpy()` if you
  need an array.

Things that changed deliberately:

- `actstats.scipy_decorators` is gone. It existed to bolt a `fit` onto SciPy's
  discrete distributions and never worked (it read an attribute that was never
  set); `actuarial.poisson.fit` and `actuarial.negative_binomial.fit` are now
  real estimators.
- The internal attributes that exposed SciPy objects (`scipy_dist`,
  `scipy_params`, `np_params`, `to_scipy`, `from_scipy`) no longer exist. Use
  `.params`, `.param_names` and `.dist`.
- `beta.fit` estimates the two shape parameters on `(0, 1)` and rejects data
  outside that range, instead of silently fitting and discarding a scale.

## 🛠️ Development

```bash
python tests/run_tests.py        # no test runner required
python -m pytest tests           # or pytest, if you have it
python examples/examples.py      # worked examples for every distribution
python benchmarks/benchmark.py   # comparison columns appear if SciPy is installed
```

`tests/reference_values.py` holds values generated once from SciPy so the test
suite can check full double precision without it; regenerate with
`python scripts/regenerate_reference.py > tests/reference_values.py` in an
environment that has SciPy.
