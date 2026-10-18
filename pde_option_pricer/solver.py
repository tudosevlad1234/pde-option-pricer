from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import csc_matrix, diags, identity
from scipy.sparse.linalg import splu

from ._checks import require_int
from .grid import build_log_grid
from .spec import OptionSpec

CRANK_NICOLSON = 0.5
IMPLICIT_EULER = 1.0

Stencil = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True, eq=False)
class PricingResult:
    price: float
    delta: float
    gamma: float
    theta: float
    spot_grid: np.ndarray
    price_grid: np.ndarray
    delta_grid: np.ndarray
    gamma_grid: np.ndarray
    theta_grid: np.ndarray


def _stencil_weights(x: np.ndarray) -> tuple[Stencil, Stencil]:
    h_minus = x[1:-1] - x[:-2]
    h_plus = x[2:] - x[1:-1]
    total = h_minus + h_plus
    first = (
        -h_plus / (h_minus * total),
        (h_plus - h_minus) / (h_minus * h_plus),
        h_minus / (h_plus * total),
    )
    second = (
        2.0 / (h_minus * total),
        -2.0 / (h_minus * h_plus),
        2.0 / (h_plus * total),
    )
    return first, second


def _apply_stencil(weights: Stencil, values: np.ndarray) -> np.ndarray:
    return weights[0] * values[:-2] + weights[1] * values[1:-1] + weights[2] * values[2:]


def _generator_coefficients(spec: OptionSpec, first: Stencil, second: Stencil) -> Stencil:
    variance = spec.vol**2
    drift = spec.rate - spec.div_yield - 0.5 * variance
    lower = 0.5 * variance * second[0] + drift * first[0]
    middle = 0.5 * variance * second[1] + drift * first[1] - spec.rate
    upper = 0.5 * variance * second[2] + drift * first[2]
    return lower, middle, upper


def _terminal_payoff(spec: OptionSpec, s: np.ndarray) -> np.ndarray:
    if spec.is_call:
        return np.maximum(s - spec.strike, 0.0)
    return np.maximum(spec.strike - s, 0.0)


def _boundary_values(spec: OptionSpec, s_min: float, s_max: float, tau: float) -> tuple[float, float]:
    discount = np.exp(-spec.rate * tau)
    carry = np.exp(-spec.div_yield * tau)
    if spec.is_call:
        return 0.0, s_max * carry - spec.strike * discount
    return spec.strike * discount - s_min * carry, 0.0


def _time_schedule(maturity: float, n_time: int, rannacher_steps: int) -> list[tuple[float, float]]:
    dt = maturity / n_time
    startup = min(rannacher_steps, n_time)
    half_steps = [(dt / 2.0, IMPLICIT_EULER)] * (2 * startup)
    full_steps = [(dt, CRANK_NICOLSON)] * (n_time - startup)
    return half_steps + full_steps


def _march_backward(payoff, operator: csc_matrix, schedule, boundary_vector) -> np.ndarray:
    eye = identity(operator.shape[0], format="csc")
    stages = {}
    for step_dt, weight in dict.fromkeys(schedule):
        explicit = eye + (1.0 - weight) * step_dt * operator
        implicit = splu((eye - weight * step_dt * operator).tocsc())
        stages[(step_dt, weight)] = (explicit, implicit)

    values = payoff
    tau = 0.0
    boundary_now = boundary_vector(tau)
    for step_dt, weight in schedule:
        tau_next = tau + step_dt
        boundary_next = boundary_vector(tau_next)
        explicit, implicit = stages[(step_dt, weight)]
        rhs = explicit @ values + step_dt * ((1.0 - weight) * boundary_now + weight * boundary_next)
        values = implicit.solve(rhs)
        tau, boundary_now = tau_next, boundary_next
    return values


def price_option(
    spec: OptionSpec,
    n_space: int = 400,
    n_time: int = 200,
    rannacher_steps: int = 2,
    width: float = 5.0,
    concentration: float = 0.15,
) -> PricingResult:
    require_int("n_space", n_space, 8)
    require_int("n_time", n_time, 1)
    require_int("rannacher_steps", rannacher_steps, 0)

    x = build_log_grid(
        spec.spot,
        spec.strike,
        spec.maturity,
        spec.rate,
        spec.div_yield,
        spec.vol,
        n_space,
        width,
        concentration,
    )
    s = np.exp(x)
    x_inner = x[1:-1]
    s_inner = s[1:-1]
    x_spot = np.log(spec.spot)

    if not x_inner[0] <= x_spot <= x_inner[-1]:
        raise ValueError("spot lies outside the interior of the grid; increase n_space or width")

    first, second = _stencil_weights(x)
    lower, middle, upper = _generator_coefficients(spec, first, second)
    operator = diags([lower[1:], middle, upper[:-1]], [-1, 0, 1], format="csc")

    def boundary_vector(tau: float) -> np.ndarray:
        low_value, high_value = _boundary_values(spec, s[0], s[-1], tau)
        vector = np.zeros(len(x_inner))
        vector[0] += lower[0] * low_value
        vector[-1] += upper[-1] * high_value
        return vector

    values = _march_backward(
        _terminal_payoff(spec, s_inner),
        operator,
        _time_schedule(spec.maturity, n_time, rannacher_steps),
        boundary_vector,
    )

    low_final, high_final = _boundary_values(spec, s[0], s[-1], spec.maturity)
    with_boundaries = np.concatenate([[low_final], values, [high_final]])
    v_x = _apply_stencil(first, with_boundaries)
    v_xx = _apply_stencil(second, with_boundaries)

    delta_grid = v_x / s_inner
    gamma_grid = (v_xx - v_x) / s_inner**2
    theta_grid = (
        spec.rate * values
        - (spec.rate - spec.div_yield) * s_inner * delta_grid
        - 0.5 * spec.vol**2 * s_inner**2 * gamma_grid
    )

    def at_spot(curve: np.ndarray) -> float:
        return float(CubicSpline(x_inner, curve)(x_spot))

    return PricingResult(
        price=at_spot(values),
        delta=at_spot(delta_grid),
        gamma=at_spot(gamma_grid),
        theta=at_spot(theta_grid),
        spot_grid=s_inner,
        price_grid=values,
        delta_grid=delta_grid,
        gamma_grid=gamma_grid,
        theta_grid=theta_grid,
    )
