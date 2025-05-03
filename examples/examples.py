from actstats import actuarial

######################################
##### All distribution testing########
######################################
# Test the lognormal distribution
lognormal_dist = actuarial.lognormal
lognormal_dist = actuarial.lognormal(0.5, 0.2)
lognormal_dist_sample = lognormal_dist.rvs(size=10000)
lognormal_dist_sample = lognormal_dist.np_rvs(size=10000)
lognormal_dist_sample.mean()
lognormal_dist_sample.std()
actuarial.lognormal.fit(lognormal_dist_sample)

# Test the gamma distribution
gamma_dist = actuarial.gamma
gamma_dist = actuarial.gamma(2, 1)
gamma_dist_sample = gamma_dist.rvs(size=10000)
gamma_dist_sample = gamma_dist.np_rvs(size=10000)
gamma_dist_sample.mean()
gamma_dist_sample.std()
actuarial.gamma.fit(gamma_dist_sample)

# Test the Weibull distribution
weibull_dist = actuarial.weibull
weibull_dist = actuarial.weibull(1.5, 1)
weibull_dist_sample = weibull_dist.rvs(size=10000)
weibull_dist_sample = weibull_dist.np_rvs(size=10000)
weibull_dist_sample.mean()
weibull_dist_sample.std()
actuarial.weibull.fit(weibull_dist_sample)

# Test the Pareto distribution
pareto_dist = actuarial.pareto
pareto_dist = actuarial.pareto(3, 1)
pareto_dist_sample = pareto_dist.rvs(size=10000)
pareto_dist_sample = pareto_dist.np_rvs(size=10000)
pareto_dist_sample.mean()
pareto_dist_sample.std()
actuarial.pareto.fit(pareto_dist_sample)

# Test the beta distribution
beta_dist = actuarial.beta
beta_dist = actuarial.beta(1, 2)
beta_dist_sample = beta_dist.rvs(size=10000)
beta_dist_sample = beta_dist.np_rvs(size=10000)
beta_dist_sample.mean()
beta_dist_sample.std()
actuarial.beta.fit(beta_dist_sample)

# Test the Poisson distribution
poisson_dist = actuarial.poisson
poisson_dist = actuarial.poisson(5,)
poisson_dist_sample = poisson_dist.rvs(size=10000)
poisson_dist_sample = poisson_dist.np_rvs(size=10000)
poisson_dist_sample.mean()
poisson_dist_sample.std()
actuarial.poisson.fit(poisson_dist_sample)

# Test the negative_binomial distribution
negative_binomial_dist = actuarial.negative_binomial
negative_binomial_dist = actuarial.negative_binomial(5, 0.5)
negative_binomial_dist_sample = negative_binomial_dist.rvs(size=10000)
negative_binomial_dist_sample = negative_binomial_dist.np_rvs(size=10000)
negative_binomial_dist_sample.mean()
negative_binomial_dist_sample.std()
actuarial.negative_binomial.fit(negative_binomial_dist_sample)

# Test the normal distribution
normal_dist = actuarial.normal
normal_dist = actuarial.normal(0, 1)
normal_dist_sample = normal_dist.rvs(size=10000)
normal_dist_sample = normal_dist.np_rvs(size=10000)
normal_dist_sample.mean()
normal_dist_sample.std()
actuarial.normal.fit(normal_dist_sample)

# Test the logistic distribution
logistic_dist = actuarial.logistic
logistic_dist = actuarial.logistic(0, 1)
logistic_dist_sample = logistic_dist.rvs(size=10000)
logistic_dist_sample = logistic_dist.np_rvs(size=10000)
logistic_dist_sample.mean()
logistic_dist_sample.std()
actuarial.logistic.fit(logistic_dist_sample)

# Test the exponential distribution
exponential_dist = actuarial.exponential
exponential_dist = actuarial.exponential(2)
exponential_dist_sample = exponential_dist.rvs(size=10000)
exponential_dist_sample = exponential_dist.np_rvs(size=10000)
exponential_dist_sample.mean()
exponential_dist_sample.std()
actuarial.exponential.fit(exponential_dist_sample)

# Test the uniform distribution
uniform_dist = actuarial.uniform
uniform_dist = actuarial.uniform(0, 1)
uniform_dist_sample = uniform_dist.rvs(size=10000)
uniform_dist_sample = uniform_dist.np_rvs(size=10000)
uniform_dist_sample.mean()
uniform_dist_sample.std()
actuarial.uniform.fit(uniform_dist_sample)
