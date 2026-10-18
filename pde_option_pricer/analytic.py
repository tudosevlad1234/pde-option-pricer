import numpy as np
from scipy.stats import norm

from .spec import OptionSpec


def black_scholes(spec: OptionSpec, spot: float | np.ndarray | None = None) -> dict[str, np.ndarray]:
    s = np.asarray(spec.spot if spot is None else spot, dtype=float)
    k = spec.strike
    t = spec.maturity
    r = spec.rate
    q = spec.div_yield
    v = spec.vol

    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t
    discount = np.exp(-r * t)
    carry = np.exp(-q * t)
    pdf = norm.pdf(d1)

    gamma = carry * pdf / (s * v * sqrt_t)
    decay = -s * carry * pdf * v / (2.0 * sqrt_t)

    if spec.is_call:
        price = s * carry * norm.cdf(d1) - k * discount * norm.cdf(d2)
        delta = carry * norm.cdf(d1)
        theta = decay - r * k * discount * norm.cdf(d2) + q * s * carry * norm.cdf(d1)
    else:
        price = k * discount * norm.cdf(-d2) - s * carry * norm.cdf(-d1)
        delta = -carry * norm.cdf(-d1)
        theta = decay + r * k * discount * norm.cdf(-d2) - q * s * carry * norm.cdf(-d1)

    return {"price": price, "delta": delta, "gamma": gamma, "theta": theta}
