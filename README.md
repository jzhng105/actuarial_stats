# actuarial_stats
**Of the Actuary, By the Actuary, For the Actuary**
`actstats` is a Python library unifying Statistical Libraries with Actuarial Conventions

---

## 🔧 Installation

```bash
pip install actstats

---

## 🔢 ActuarialDistribution class
| Distribution             | Actuarial Parameters | SciPy Equivalent             |
| ------------------------ | -------------------- | ---------------------------- |
| `lognormal`              | (μ, σ)               | `lognorm(s=σ, scale=exp(μ))` |
| `gamma`                  | (α, θ)               | `gamma(a=α, scale=θ)`        |
| `weibull`                | (α, β)               | `weibull_min(c=α, scale=β)`  |
| `pareto`                 | (α, θ)               | `pareto(b=α, scale=θ)`       |
| `beta`                   | (α, β)               | `beta(a=α, b=β)`             |
| `poisson`                | (λ)                  | `poisson(mu=λ)`              |
| `negative_binomial`      | (r, p)               | `nbinom(n=r, p=p)`           |
| `normal`                 | (μ, σ)               | `norm(loc=μ, scale=σ)`       |
| `logistic`               | (μ, σ)               | `logistic(loc=μ, scale=σ)`   |
| `exponential`            | (θ)                  | `expon(scale=θ)`             |
| `uniform`                | (a, b)               | `uniform(loc=a, scale=b−a)`  |
| `nonhomogeneous_poisson` | (λ₀, α, ϕ, T)        | custom `NHPPDistribution`    |
