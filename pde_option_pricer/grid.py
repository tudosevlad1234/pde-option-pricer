import numpy as np

from ._checks import require_finite, require_int, require_positive


def build_log_grid(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div_yield: float,
    vol: float,
    n_space: int,
    width: float = 5.0,
    concentration: float = 0.15,
) -> np.ndarray:
    for name, value in (
        ("spot", spot),
        ("strike", strike),
        ("maturity", maturity),
        ("vol", vol),
        ("width", width),
        ("concentration", concentration),
    ):
        require_positive(name, value)
    require_finite("rate", rate)
    require_finite("div_yield", div_yield)
    require_int("n_space", n_space, 4)

    center = np.log(strike)
    spread = width * vol * np.sqrt(maturity) + abs(rate - div_yield) * maturity
    x_min = min(np.log(spot), center) - spread
    x_max = max(np.log(spot), center) + spread

    scale = concentration * (x_max - x_min)
    xi_min = np.arcsinh((x_min - center) / scale)
    xi_max = np.arcsinh((x_max - center) / scale)

    n_left = int(round(n_space * (-xi_min) / (xi_max - xi_min)))
    n_left = min(max(n_left, 2), n_space - 2)
    n_right = n_space - n_left

    xi_left = np.linspace(xi_min, 0.0, n_left + 1)
    xi_right = np.linspace(0.0, xi_max, n_right + 1)[1:]
    xi = np.concatenate([xi_left, xi_right])

    return center + scale * np.sinh(xi)
