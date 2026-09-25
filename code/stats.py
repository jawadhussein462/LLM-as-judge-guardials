"""One-sided Clopper--Pearson bounds on a Bernoulli mean."""
import math

from scipy.stats import beta


def cp_upper(k, n, delta):
    """u such that Pr(true rate > u) <= delta given k events in n trials."""
    if n <= 0 or k >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def cp_lower(k, n, delta):
    """l such that Pr(true rate < l) <= delta given k events in n trials."""
    if n <= 0 or k <= 0:
        return 0.0
    return float(beta.ppf(delta, k, n - k + 1))


def risk_ucb(n_miss, n_mal, delta):
    return cp_upper(n_miss, n_mal, delta)


def util_lcb(n_pass, n_ben, delta):
    return cp_lower(n_pass, n_ben, delta)


def delta_k(k, delta_total):
    """Summable budget: sum_{k>=1} delta_k = delta_total."""
    return delta_total * 6.0 / (math.pi ** 2 * k * k)
