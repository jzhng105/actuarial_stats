import numpy as np
import scipy.stats as stats
from actstats.scipy_decorators import poisson, nbinom
from actstats.utils.utils import fraction_to_date_full

class NHPPDistribution:
    """
    Custom class to simulate a Nonhomogeneous Poisson Process (NHPP)
    with a seasonal rate function:
      lambda(t) = lambda0 * (1 + alpha * sin(2*pi*t + phase))
    over the interval [0, T].
    
    If n_events is specified in rvs(), the simulation uses the order-statistics
    approach to simulate exactly that many event times, conditional on an event
    occurring. Otherwise, the thinning algorithm is used.
    """
    def __init__(self, lambda0=10, alpha=0.5, phase=0, T=1, num_points=10000):
        self.lambda0 = lambda0
        self.alpha = alpha
        self.phase = phase
        self.T = T
        # Upper bound for thinning
        self.lambda_max = lambda0 * (1 + abs(alpha))
    
        # Precompute cumulative intensity function for efficiency
        self.num_points = num_points
        self.t_grid = np.linspace(0, self.T, self.num_points)
        self.dt = self.T / (self.num_points - 1)
        self.rates = self.rate(self.t_grid)
        self.cum_intensity = np.cumsum(self.rates) * self.dt
        self.F = self.cum_intensity / self.cum_intensity[-1]

    def rate(self, t):
        """Seasonal rate function at time t."""
        return self.lambda0 * (1 + self.alpha * np.sin(2 * np.pi * t + self.phase))
    
    def rvs(self, size=1, n_events=None):
        """
        Simulate event times.
        
        Parameters:
            size: Number of realizations (each realization is an array of event times).
            n_events: If specified (an integer), simulate exactly that many events
                      using the conditional (order statistics) method.
                      Otherwise, simulate using the thinning algorithm until time T.
        
        Returns:
            If size == 1: a single array of event times.
            Otherwise, a list of arrays.
        """
        def simulate_once():
            if n_events is not None:
                u = np.sort(np.random.uniform(0, 1, n_events))
                simulated_t = np.interp(u, self.F, self.t_grid)
                return simulated_t
            else:
                events = []
                t = 0
                while t < self.T:
                    u = np.random.exponential(1 / self.lambda_max)
                    t += u
                    if t >= self.T:
                        break
                    if np.random.uniform() < self.rate(t) / self.lambda_max:
                        events.append(t)
                return np.array(events)

        if size == 1:
            return simulate_once()
        else:
            return [simulate_once() for _ in range(size)]

    def pdf(self, t):
        """
        Returns the instantaneous rate at time t.
        (Note: an NHPP does not have a PDF in the traditional sense.)
        """
        return self.rate(t)

class ActuarialDistribution:
    """Generic class to modify SciPy distributions to use actuarial conventions.
        Supported distributions:
        - lognormal(μ, σ) -> lognorm(s=σ, scale=exp(μ))
        - gamma(α, β) -> gamma(a=α, scale=β)
        - weibull(α, β) -> weibull_min(c=α, scale=β)
        - pareto(α, β) -> lomax(c=α, scale=β)        
        - beta(α, β) -> beta(a=α, b=β)
        - poisson(λ) -> poisson(mu=λ)
        - negative_binomial(r, p) -> nbinom(n=r, p=p)
        - normal(μ, σ) -> norm(loc=μ, scale=σ)
        - logistic(μ, s) -> logistic(loc=μ, scale=s)
        - exponential(β) -> expon(scale=β)
        - uniform(a, b) -> uniform(loc=a, scale=b-a)
    """
    _distributions = {
        "lognormal": (stats.lognorm, 
                      lambda mu=0, sigma=1: (sigma, 0, np.exp(mu)),   
                      lambda params: (np.log(params[2]), params[0]),
                      lambda mu=0, sigma=1: {'mu': mu, 'sigma': sigma},
                      lambda mu, sigma, size: np.random.lognormal(mean=mu, sigma=sigma, size=size)),

        "gamma": (stats.gamma, 
                  lambda alpha=1, theta=1: (alpha, 0, theta),  
                  lambda params: (params[0], params[2]),
                  lambda alpha=1, theta=1: {'alpha': alpha, 'theta': theta},
                  lambda alpha, theta, size: np.random.gamma(shape=alpha, scale=theta, size=size)),

        "weibull": (stats.weibull_min, 
                    lambda delta=1, beta=1: (delta, 0, beta),  # follow bahnemann paper 
                    lambda params: (params[0], params[2]),
                    lambda delta=1, beta=1: {'delta': delta, 'beta': beta},
                    lambda delta, beta, size: beta * np.random.weibull(a=delta, size=size)), # numpy weibull has only 1 parameter

        "pareto": (stats.lomax, 
                   lambda alpha=1, beta=1: (alpha, 0, beta),   # follow bahnemann paper 
                   lambda params: (params[0], params[2]),
                   lambda alpha=1, beta=1: {'alpha': alpha, 'beta': beta},
                   lambda alpha, beta, size: beta * (np.random.pareto(a=alpha, size=size))), # unshifted pareto should be lam * (np.random.pareto(a=alpha, size=size) + 1))

        "beta": (stats.beta, 
                 lambda alpha=1, beta=1: (alpha, beta),   # follow wikipedia convention 
                 lambda params: (params[0], params[1]),
                 lambda alpha=1, beta=1: {'alpha': alpha, 'beta': beta},
                 lambda alpha, beta, size: np.random.beta(a=alpha, b=beta, size=size)),

        "poisson": (stats.poisson, 
                    lambda lambda_=1: (lambda_, 0),   
                    lambda params: (params[0],),
                    lambda lambda_=1: {'mu': lambda_}, 
                    lambda mu, size: np.random.poisson(lam=mu, size = size)),

        "negative_binomial": (stats.nbinom, 
                              lambda r=1, p=0.5: (r, p),   # follow wikipedia convention 
                              lambda params: (params[0], params[1]),
                              lambda r=1, p=0.5: {'n': r, 'p': p},
                              lambda n, p, size: np.random.negative_binomial(n=n, p=p, size=size)),

        "normal": (stats.norm, 
                   lambda mu=0, sigma=1: (mu, sigma),  
                   lambda params: (params[0], params[1]),
                   lambda mu=0, sigma=1: {'mu': mu, 'sigma': sigma},
                   lambda mu, sigma, size: np.random.normal(loc=mu, scale=sigma, size=size)),

        "logistic": (stats.logistic, # follow wikipedia convention
                     lambda mu=0, s=1: (mu, s),  
                     lambda params: (params[0], params[1]),
                     lambda mu=0, s=1: {'mu': mu, 's': s},
                     lambda mu, s, size: np.random.logistic(loc=mu, scale=s, size=size)),

        "exponential": (stats.expon, 
                        lambda beta=1: (0, beta), # follow bahnemann paper 
                        lambda params: (params[1],),
                        lambda beta=1: {'beta': beta},
                        lambda beta, size: np.random.exponential(scale=beta, size=size)),

        "uniform": (stats.uniform, 
                    lambda a=0, b=1: (a, b - a),   
                    lambda params: (params[0], params[0] + params[1]),
                    lambda a=0, b=1: {'a': a, 'b': b},
                    lambda a, b, size: np.random.uniform(low=a, high=b, size=size)),

        # Nonhomogeneous Poisson process simulation
        "nonhomogeneous_poisson": (NHPPDistribution,
                                   lambda lambda0=10, alpha=0.5, phase=0, T=1: (lambda0, alpha, phase, T),
                                   lambda params: (params[0], params[1], params[2], params[3]),
                                   lambda lambda0=10, alpha=0.5, phase=0, T=1: {'lambda0': lambda0, 'alpha': alpha, 'phase': phase, 'T': T},
                                   lambda lambda0, alpha, phase, T, size, n_events: NHPPDistribution(lambda0, alpha, phase, T).rvs(size=size, n_events=n_events)),
    }

    def __init__(self, name, *args, **kwargs):
        """
        - Takes in actuarial parameters and converts them to SciPy's format.
        - Stores the original SciPy distribution with modified parameters.
        """
        if name not in self._distributions:
            raise ValueError(f"Unsupported distribution: {name}")

        self.name = name
        self.scipy_dist, self.to_scipy, self.from_scipy, self.to_numpy, self.np_sampler = self._distributions[name]
         # If no parameters are provided, initialize with defaults
        self.np_params = self.to_numpy(*args) if args else self.to_numpy()
        self.scipy_params = self.to_scipy(*args) if args else self.to_scipy()
        self.used_default_params = not args  # True if default, False if user-supplied
        self.dist = self.scipy_dist(*self.scipy_params, **kwargs)  # Store SciPy instance

    def fit(self, data, *args, **kwargs):
        """Fit distribution and return actuarial parameters."""
        if self.name not in ["uniform", "normal", "logistic", "poisson", "negative_binomial"]:
            # Force loc=0 for consistency, poisson and negative binomial fit functions 
            # are defined in scipy_decorators.py which does not require this
            kwargs["floc"] = 0
        fitted_params = self.scipy_dist.fit(data, *args, **kwargs)
        return self.from_scipy(fitted_params)
    
    def np_rvs(self, size=None, **kwargs):
        """NumPy-based sampling"""
        return self.np_sampler(**self.np_params, size=size, **kwargs)
    
    def ks_test(self, data):
        try:
            cdf = self.dist.cdf
            return stats.kstest(data, cdf).statistic
        except Exception as e:
            raise RuntimeError(f"Error computing KS statistic: {e}")
    def __getattr__(self, name):
        """Ensures that both frozen and unfrozen behavior work."""
        attr = getattr(self.scipy_dist, name, None)
        
        if callable(attr):
            # If calling a method like .ppf() with additional parameters, handle dynamically
            def method(*args, **kwargs):
                if self.used_default_params:  # Dist parameter was not provided when the class was initiated, call the unfrozen SciPy function, and convert actuarial parameters to scipy format
                    # Check if first arg is likely data (array-like or scalar)
                    n_params = self.to_scipy.__code__.co_argcount
                    if name in ("rvs", "stats"):
                        # first n_params are the distribution parameters
                        scipy_params = self.to_scipy(*args[:n_params])
                        func_params = args[n_params:]
                        return attr(*scipy_params, *func_params, **kwargs)
                    else:
                        # last n_params are the distribution parameters
                        scipy_params = self.to_scipy(*args[-n_params:])
                        func_params = args[:-n_params]
                        return attr(*func_params, *scipy_params, **kwargs)
                else:  # Otherwise, call the frozen instance
                    return getattr(self.dist, name)(*args, **kwargs)

            return method
        else:
            return getattr(self.dist, name)

    def __call__(self, *args, **kwargs):
        """Allow callable behavior to create a new instance with parameters."""
        return ActuarialDistribution(self.name, *args, **kwargs)

class ActuarialClass:
    """Dynamically generates actuarial distributions with instance and constructor support."""

    def __getattr__(self, name):
        """Creates an ActuarialDistribution instance dynamically."""
        if name not in ActuarialDistribution._distributions:
            raise AttributeError(f"'ActuarialClass' has no attribute '{name}'")
        
        return ActuarialDistribution(name)  # Directly return the instance


# Create an instance of the ActuarialClass
actuarial = ActuarialClass()

if __name__ == "__main__":
    np.random.seed(42)  # Set seed for NumPy
    dist = stats.poisson(1, 0).rvs(size=10000)
    dist = np.random.poisson(0.5, 1000)
    dist = actuarial.poisson(0.5).rvs(size=1000)
    dist = actuarial.poisson(0.5,).np_rvs(size = 10000)
    dist = actuarial.lognormal
    dist = actuarial.lognormal.rvs(0.5, 0.2,size = 1000)
    actuarial.lognormal(0.5, 0.2).np_rvs(size = 1000).mean()
    actuarial.lognormal.fit(dist)
    actuarial.lognormal(0.5, 0.2).ks_test(actuarial.lognormal.rvs(0.5, 0.2,size = 1000))
    # Test rvs functions
    stats.poisson(10).rvs(1000)
    stats.poisson.rvs(10, 1000)
    actuarial.poisson(10).rvs(1000)
    actuarial.poisson.rvs(10,1000)

    # Test pmf functions
    stats.poisson.ppf(0.5, 10)
    stats.poisson(10).ppf(0.5)
    actuarial.poisson.ppf(0.5, 10)
    actuarial.poisson(10).ppf(0.5)

    # Test logpmf functions
    stats.poisson.logpmf(5, 10)
    stats.poisson(10).logpmf(5)
    actuarial.poisson.logpmf(5, 10)
    actuarial.poisson(10).logpmf(5)

    # Test rvs functions severity
    stats.lognorm(2,1).rvs(1000)
    stats.lognorm.rvs(2, 1, 1000)
    actuarial.lognormal(2,1).rvs(1000)
    actuarial.lognormal.rvs(2, 1, 1000)

    # Test pdf functions severity
    stats.lognorm.ppf(0.5, 1, 0, np.exp(2))
    stats.lognorm(1,0, np.exp(2)).ppf(0.5)
    actuarial.lognormal.ppf(0.5, 2, 1)
    actuarial.lognormal(2, 1).ppf(0.5)


    # Test pdf functions severity
    stats.lognorm.pdf(5, 1, 0, np.exp(2))
    stats.lognorm(1, 0, np.exp(2)).pdf(5)
    actuarial.lognormal.pdf(5, 2, 1)
    actuarial.lognormal(2, 1).pdf(5)

    # Test logpdf functions severity
    stats.lognorm.logpdf(5, 1, 0, np.exp(2))
    stats.lognorm(1, 0, np.exp(2)).logpdf(5)
    actuarial.lognormal.logpdf(5, 2, 1)
    actuarial.lognormal(2, 1).logpdf(5)

    # Create an NHPP instance with lambda0=10, seasonal variation alpha=0.5, phase=0, over one year (T=1)
    nhpp_dist = actuarial.nonhomogeneous_poisson(10, 0.25, 0, 1)

    # Generate a single simulation (one realization of event times)
    simulated_events = nhpp_dist.rvs(size=2,n_events=10)
    simulated_events = nhpp_dist.np_rvs(size=2,n_events=10)


    # Convert each simulated event time to a month and day.
    dates = [fraction_to_date_full(t) for t in simulated_events[1]]

