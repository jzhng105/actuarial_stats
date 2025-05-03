from actuarial_stats.core import actuarial

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
poisson_dist = actuarial.poisson(5)
poisson_dist_sample = poisson_dist.rvs(size=10000)
poisson_dist_sample = poisson_dist.np_rvs(size=10000)
poisson_dist_sample.mean()
poisson_dist_sample.std()
actuarial.poisson.fit(poisson_dist_sample)