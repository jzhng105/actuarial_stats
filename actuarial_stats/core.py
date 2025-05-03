import numpy as np
import scipy.stats as stats
import datetime
from actuarial_stats.scipy_decorators import poisson, nbinom

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
        - gamma(α, θ) -> gamma(a=α, scale=θ)
        - weibull(α, β) -> weibull_min(c=α, scale=β)
        - pareto(α, θ) -> pareto(b=α, scale=θ)
        - beta(α, β) -> beta(a=α, b=β)
        - poisson(λ) -> poisson(mu=λ)
        - negative_binomial(r, p) -> nbinom(n=r, p=p)
        - normal(μ, σ) -> norm(loc=μ, scale=σ)
        - logistic(μ, σ) -> logistic(loc=μ, scale=σ)
        - exponential(θ) -> expon(scale=θ)
        - uniform(a, b) -> uniform(loc=a, scale=b-a)
    """
    _distributions = {
        "lognormal": (stats.lognorm, 
                      lambda mu=0, sigma=1: (sigma, 0, np.exp(mu)),   
                      lambda params: (np.log(params[2]), params[0]),
                      lambda mu, sigma: {'mu': mu, 'sigma': sigma},
                      lambda mu, sigma, size: np.random.lognormal(mean=mu, sigma=sigma, size=size)),

        "gamma": (stats.gamma, 
                  lambda alpha=1, theta=1: (alpha, 0, theta),  
                  lambda params: (params[0], params[2]),
                  lambda alpha, theta: {'alpha': alpha, 'theta': theta},
                  lambda alpha, theta, size: np.random.gamma(shape=alpha, scale=theta, size=size)),

        "weibull": (stats.weibull_min, 
                    lambda alpha=1, beta=1: (alpha, 0, beta),   
                    lambda params: (params[0], params[2]),
                    lambda alpha, beta: {'alpha': alpha, 'beta': beta},
                    lambda alpha, beta, size: np.random.weibull(a=alpha, size=size)), # numpy weibull has only 1 parameter

        "pareto": (stats.pareto, 
                   lambda alpha=1, theta=1: (alpha, 0, theta),  
                   lambda params: (params[0], params[2]),
                   lambda alpha, theta: {'alpha': alpha, 'theta': theta},
                   lambda alpha, theta, size: np.random.pareto(a=alpha, size=size)), # numpy pareto has only 1 parameter

        "beta": (stats.beta, 
                 lambda alpha=1, beta=1: (alpha, beta),   
                 lambda params: (params[0], params[1]),
                 lambda alpha, beta: {'alpha': alpha, 'beta': beta},
                 lambda alpha, beta, size: np.random.beta(a=alpha, b=beta, size=size)),

        "poisson": (stats.poisson, 
                    lambda lambda_=1: (lambda_, 0),   
                    lambda params: (params[0],),
                    lambda lambda_=1: {'mu': lambda_}, 
                    lambda mu, size: np.random.poisson(lam=mu, size = size)),

        "negative_binomial": (stats.nbinom, 
                              lambda r=1, p=0.5: (r, 0, p),   
                              lambda params: (params[0], params[2])),

        "normal": (stats.norm, 
                   lambda mu=0, sigma=1: (mu, sigma),  
                   lambda params: (params[0], params[1])),

        "logistic": (stats.logistic, 
                     lambda mu=0, sigma=1: (mu, sigma),  
                     lambda params: (params[0], params[1])),

        "exponential": (stats.expon, 
                        lambda theta=1: (0, theta),   
                        lambda params: (params[1],)),

        "uniform": (stats.uniform, 
                    lambda a=0, b=1: (a, b - a),   
                    lambda params: (params[0], params[0] + params[1])),

        # Nonhomogeneous Poisson process simulation
        "nonhomogeneous_poisson": (NHPPDistribution,
                                   lambda lambda0=10, alpha=0.5, phase=0, T=1: (lambda0, alpha, phase, T),
                                   lambda params: (params[0], params[1], params[2], params[3])),

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
        self.np_params = self.to_numpy(*args) if args else self.to_scipy()
        converted_params = self.to_scipy(*args) if args else self.to_scipy()

        self.dist = self.scipy_dist(*converted_params, **kwargs)  # Store SciPy instance

    def fit(self, data, *args, **kwargs):
        """Fit distribution and return actuarial parameters."""
        if self.name not in ["uniform", "normal", "logistic", "poisson"]:
            kwargs["floc"] = 0  # Force loc=0 for consistency
        fitted_params = self.scipy_dist.fit(data, *args, **kwargs)
        return self.from_scipy(fitted_params)
    
    def np_rvs(self, size=None):
        """NumPy-based sampling"""
        return self.np_sampler(**self.np_params, size=size)
    
    def __getattr__(self, name):
        """Ensures that both frozen and unfrozen behavior work."""
        attr = getattr(self.scipy_dist, name, None)
        
        if callable(attr):
            # If calling a method like .ppf() with additional parameters, handle dynamically
            def method(*args, **kwargs):
                if args:  # If args are provided, call the unfrozen SciPy function
                    return attr(*args, **kwargs)
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

# --- Helper Function to Convert Fraction to Date ---
def fraction_to_date_full(t, year=2024):
    """
    Converts a fraction of a year (0 <= t < 1) into a date (month and day)
    for a given year. Accounts for leap years.
    """
    start_date = datetime.date(year, 1, 1)
    # Determine number of days in the year
    if (year % 400 == 0) or ((year % 4 == 0) and (year % 100 != 0)):
        days_in_year = 366
    else:
        days_in_year = 365
    # Convert fraction to day offset (0-indexed)
    day_offset = int(t * days_in_year)
    event_date = start_date + datetime.timedelta(days=day_offset)
    return event_date

if __name__ == "__main__":
    dist = stats.poisson(1, 0).rvs(size=10000)
    dist = np.random.poisson(0.5, 1000)
    dist = actuarial.poisson(0.5).rvs(size=1000)
    dist = actuarial.poisson(0.5,).np_rvs(size = 10000)
    dist = actuarial.lognormal(0.5, 0.2).rvs(size = 1000)
    actuarial.lognormal(0.5, 0.2).np_rvs(size = 1000).mean()
    actuarial.lognormal.fit(dist)
    actuarial.poisson(10).rvs(1000)
    actuarial.poisson.ppf(0.9, 10)
    stats.poisson.ppf(0.5, 10)

    # Create an NHPP instance with lambda0=10, seasonal variation alpha=0.5, phase=0, over one year (T=1)
    nhpp_dist = actuarial.nonhomogeneous_poisson(10, 0.25, 0, 1)

    # Generate a single simulation (one realization of event times)
    simulated_events = nhpp_dist.rvs(n_events=100)

    # Convert each simulated event time to a month and day.
    dates = [fraction_to_date_full(t) for t in simulated_events]
