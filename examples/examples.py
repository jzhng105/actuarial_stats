from actstats import actuarial
import numpy as np
######################################
##### All distribution testing########
######################################
# Test the lognormal distribution
lognormal_dist = actuarial.lognormal
mu = 0.5
sigma = 0.2
lognormal_dist = actuarial.lognormal(mu, sigma)
lognormal_dist_sample = lognormal_dist.rvs(size=10000)
lognormal_dist_sample = lognormal_dist.np_rvs(size=10000)
sample_mean = lognormal_dist_sample.mean()
sample_var = lognormal_dist_sample.var()
theoretical_mean = np.exp(mu + sigma**2 / 2)
theoretical_var = (np.exp(sigma**2)-1)*np.exp(2*mu + sigma**2)
print(f"Theoretical mean: {theoretical_mean}, Sample mean: {sample_mean}")
print(f"Theoretical variance: {theoretical_var}, Sample variance: {sample_var}")
actuarial.lognormal.fit(lognormal_dist_sample)

# Test the gamma distribution
gamma_dist = actuarial.gamma    
alpha = 1
beta = 2
gamma_dist = actuarial.gamma(alpha, beta)
gamma_dist_sample = gamma_dist.rvs(size=10000)
gamma_dist_sample = gamma_dist.np_rvs(size=10000)
sample_mean = gamma_dist_sample.mean()
sample_var = gamma_dist_sample.var()
theoretical_mean = alpha * beta
theoretical_var = alpha * beta**2
print(f"Theoretical mean: {theoretical_mean}, Sample mean: {sample_mean}")
print(f"Theoretical variance: {theoretical_var}, Sample variance: {sample_var}")
actuarial.gamma.fit(gamma_dist_sample)

# Test the Weibull distribution
weibull_dist = actuarial.weibull
delta = 1.5
beta = 1
weibull_dist = actuarial.weibull(delta, beta)
weibull_dist_sample = weibull_dist.rvs(size=10000)
weibull_dist_sample = weibull_dist.np_rvs(size=10000)
sample_mean = weibull_dist_sample.mean()
sample_var = weibull_dist_sample.var()
theoretical_mean = beta * np.math.gamma(1 + 1/delta)
theoretical_var = beta**2 * (np.math.gamma(1 + 2/delta) - np.math.gamma(1 + 1/delta)**2)
print(f"Theoretical mean: {theoretical_mean}, Sample mean: {sample_mean}")
print(f"Theoretical variance: {theoretical_var}, Sample variance: {sample_var}")
actuarial.weibull.fit(weibull_dist_sample)

# Test the Pareto distribution
pareto_dist = actuarial.pareto
alpha = 5
beta = 1
pareto_dist = actuarial.pareto(alpha, beta)
pareto_dist_sample = pareto_dist.rvs(size=10000)
pareto_dist_sample = pareto_dist.np_rvs(size=10000)
sample_mean = pareto_dist_sample.mean()
sample_var = pareto_dist_sample.var()
theoretical_mean = beta / (alpha - 1) if alpha > 1 else np.inf
theoretical_var = (beta**2 * alpha) / ((alpha - 1)**2 * (alpha - 2)) if alpha > 2 else np.inf
print(f"Theoretical mean: {theoretical_mean}, Sample mean: {sample_mean}")
print(f"Theoretical variance: {theoretical_var}, Sample variance: {sample_var}")
actuarial.pareto.fit(pareto_dist_sample)

# Test the beta distribution
beta_dist = actuarial.beta
alpha = 1
beta = 2
beta_dist = actuarial.beta(alpha, beta)
beta_dist_sample = beta_dist.rvs(size=10000)
beta_dist_sample = beta_dist.np_rvs(size=10000)
sample_mean = beta_dist.rvs(size=10000).mean()
sample_var = beta_dist_sample.var()
theoretical_mean = alpha / (alpha + beta)
theoretical_var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
print(f"Theoretical mean: {theoretical_mean}, Sample mean: {sample_mean}")
print(f"Theoretical variance: {theoretical_var}, Sample variance: {sample_var}")
actuarial.beta.fit(beta_dist_sample)

# Test the Poisson distribution
poisson_dist = actuarial.poisson
lam = 5
poisson_dist = actuarial.poisson(lam,)
poisson_dist_sample = poisson_dist.rvs(size=10000)
poisson_dist_sample = poisson_dist.np_rvs(size=10000)
sample_mean = poisson_dist_sample.mean()
sample_var = poisson_dist_sample.var()
theoretical_mean = lam
theoretical_var = lam
print(f"Theoretical mean: {theoretical_mean}, Sample mean: {sample_mean}")
print(f"Theoretical variance: {theoretical_var}, Sample variance: {sample_var}")
actuarial.poisson.fit(poisson_dist_sample)

# Test the negative_binomial distribution
negative_binomial_dist = actuarial.negative_binomial
r = 5
p = 0.5
negative_binomial_dist = actuarial.negative_binomial(r, p)
negative_binomial_dist_sample = negative_binomial_dist.rvs(size=10000)
negative_binomial_dist_sample = negative_binomial_dist.np_rvs(size=10000)
sample_mean = negative_binomial_dist_sample.mean()
sample_var = negative_binomial_dist_sample.var()
theoretical_mean = r * (1 - p) / p
theoretical_var = r * (1 - p) / p**2
print(f"Theoretical mean: {theoretical_mean}, Sample mean: {sample_mean}")
print(f"Theoretical variance: {theoretical_var}, Sample variance: {sample_var}")
actuarial.negative_binomial.fit(negative_binomial_dist_sample)

# Test the normal distribution
normal_dist = actuarial.normal
mu = 0
sigma = 1
normal_dist = actuarial.normal(mu, sigma)
normal_dist_sample = normal_dist.rvs(size=10000)
normal_dist_sample = normal_dist.np_rvs(size=10000)
sample_mean = normal_dist_sample.mean()
sample_var = normal_dist_sample.var()
theoretical_mean = mu
theoretical_var = sigma**2
print(f"Theoretical mean: {theoretical_mean}, Sample mean: {sample_mean}")
print(f"Theoretical variance: {theoretical_var}, Sample variance: {sample_var}")
actuarial.normal.fit(normal_dist_sample)

# Test the logistic distribution
logistic_dist = actuarial.logistic
mu = 0
s = 1
logistic_dist = actuarial.logistic(mu, s)
logistic_dist_sample = logistic_dist.rvs(size=10000)
logistic_dist_sample = logistic_dist.np_rvs(size=10000)
sample_mean = logistic_dist_sample.mean()
sample_var = logistic_dist_sample.var()
theoretical_mean = mu
theoretical_var = (sigma**2 * np.pi**2) / 3
print(f"Theoretical mean: {theoretical_mean}, Sample mean: {sample_mean}")
print(f"Theoretical variance: {theoretical_var}, Sample variance: {sample_var}")
actuarial.logistic.fit(logistic_dist_sample)

# Test the exponential distribution
exponential_dist = actuarial.exponential
beta = 2
exponential_dist = actuarial.exponential(beta)
exponential_dist_sample = exponential_dist.rvs(size=10000)  
exponential_dist_sample = exponential_dist.np_rvs(size=10000)
sample_mean = exponential_dist_sample.mean()
sample_var = exponential_dist_sample.var()
theoretical_mean = beta
theoretical_var = beta**2
print(f"Theoretical mean: {theoretical_mean}, Sample mean: {sample_mean}")
print(f"Theoretical variance: {theoretical_var}, Sample variance: {sample_var}")
actuarial.exponential.fit(exponential_dist_sample)

# Test the uniform distribution
uniform_dist = actuarial.uniform
a = 0
b = 1
uniform_dist = actuarial.uniform(a, b)
uniform_dist_sample = uniform_dist.rvs(size=10000)
uniform_dist_sample = uniform_dist.np_rvs(size=10000)
sample_mean = uniform_dist_sample.mean()
sample_var = uniform_dist_sample.var()
theoretical_mean = (a + b) / 2
theoretical_var = ((b - a) ** 2) / 12
print(f"Theoretical mean: {theoretical_mean}, Sample mean: {sample_mean}")
print(f"Theoretical variance: {theoretical_var}, Sample variance: {sample_var}")
actuarial.uniform.fit(uniform_dist_sample)

