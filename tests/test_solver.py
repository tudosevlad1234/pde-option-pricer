import numpy as np
import pytest

from pde_option_pricer import OptionSpec, black_scholes, build_log_grid, price_option

SPECS = [
    OptionSpec(100.0, 100.0, 1.0, 0.05, 0.20),
    OptionSpec(100.0, 100.0, 1.0, 0.05, 0.20, option_type="put"),
    OptionSpec(100.0, 110.0, 0.5, 0.03, 0.30, div_yield=0.02),
    OptionSpec(90.0, 100.0, 2.0, 0.01, 0.15, option_type="put"),
    OptionSpec(100.0, 100.0, 0.1, 0.05, 0.25),
]

CONVERGENCE_SPEC = OptionSpec(100.0, 100.0, 1.0, 0.05, 0.20, div_yield=0.02)


def _observed_orders(values):
    differences = np.abs(np.diff(np.asarray(values)))
    return np.log2(differences[:-1] / differences[1:])


@pytest.mark.parametrize("spec", SPECS)
def test_matches_closed_form(spec):
    result = price_option(spec, n_space=400, n_time=200)
    exact = black_scholes(spec)
    assert result.price == pytest.approx(float(exact["price"]), abs=2e-3)
    assert result.delta == pytest.approx(float(exact["delta"]), abs=1e-3)
    assert result.gamma == pytest.approx(float(exact["gamma"]), abs=1e-3)
    assert result.theta == pytest.approx(float(exact["theta"]), abs=2e-2)


def test_put_call_parity():
    call = OptionSpec(100.0, 105.0, 1.0, 0.04, 0.25, div_yield=0.01, option_type="call")
    put = OptionSpec(100.0, 105.0, 1.0, 0.04, 0.25, div_yield=0.01, option_type="put")
    call_price = price_option(call).price
    put_price = price_option(put).price
    parity = call.spot * np.exp(-call.div_yield * call.maturity) - call.strike * np.exp(
        -call.rate * call.maturity
    )
    assert call_price - put_price == pytest.approx(parity, abs=2e-3)


def test_error_decreases_with_refinement():
    spec = SPECS[0]
    exact = float(black_scholes(spec)["price"])
    coarse = abs(price_option(spec, n_space=100, n_time=50).price - exact)
    fine = abs(price_option(spec, n_space=800, n_time=400).price - exact)
    assert fine < coarse


@pytest.mark.parametrize("greek", ["price", "delta", "gamma", "theta"])
def test_second_order_in_space(greek):
    values = [
        getattr(price_option(CONVERGENCE_SPEC, n_space=n, n_time=2000), greek) for n in (100, 200, 400)
    ]
    assert _observed_orders(values) == pytest.approx([2.0], abs=0.15)


@pytest.mark.parametrize("greek", ["price", "delta", "gamma", "theta"])
def test_second_order_in_time(greek):
    values = [
        getattr(price_option(CONVERGENCE_SPEC, n_space=1600, n_time=n), greek) for n in (50, 100, 200, 400)
    ]
    assert _observed_orders(values) == pytest.approx([2.0, 2.0], abs=0.2)


def test_rannacher_startup_removes_gamma_oscillations():
    spec = OptionSpec(100.0, 100.0, 0.25, 0.05, 0.20)
    plain = price_option(spec, n_space=400, n_time=25, rannacher_steps=0)
    smoothed = price_option(spec, n_space=400, n_time=25, rannacher_steps=2)
    mask = (plain.spot_grid > 85.0) & (plain.spot_grid < 115.0)
    exact = np.asarray(black_scholes(spec, plain.spot_grid[mask])["gamma"])
    assert np.max(np.abs(plain.gamma_grid[mask] - exact)) > 1.0
    assert np.max(np.abs(smoothed.gamma_grid[mask] - exact)) < 1e-3


def test_grid_contains_strike_and_is_increasing():
    x = build_log_grid(100.0, 120.0, 1.0, 0.05, 0.0, 0.2, 300)
    assert len(x) == 301
    assert np.all(np.diff(x) > 0)
    assert np.any(np.isclose(x, np.log(120.0), atol=1e-12))


def test_grid_is_denser_near_strike():
    x = build_log_grid(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 300)
    spacing = np.diff(x)
    center = np.argmin(np.abs(x - np.log(100.0)))
    assert spacing[center] < spacing[0]
    assert spacing[center] < spacing[-1]


@pytest.mark.parametrize("n_space", [8, 9, 50, 401])
def test_grid_is_well_formed_for_any_size(n_space):
    x = build_log_grid(40.0, 100.0, 5.0, 0.15, 0.03, 0.05, n_space)
    assert len(x) == n_space + 1
    assert np.all(np.diff(x) > 0)
    assert np.all(np.isfinite(x))
    assert np.any(np.isclose(x, np.log(100.0), atol=1e-10))


@pytest.mark.parametrize("spec", SPECS)
def test_no_arbitrage_in_central_region(spec):
    result = price_option(spec, n_space=400, n_time=200)
    mask = (result.spot_grid >= 0.5 * spec.strike) & (result.spot_grid <= 1.5 * spec.strike)
    delta_cap = np.exp(-spec.div_yield * spec.maturity)
    slope = np.diff(result.price_grid)

    assert np.all(result.gamma_grid[mask] > 0.0)
    assert np.all(result.price_grid >= -1e-10)
    assert np.all(np.abs(result.delta_grid[mask]) <= delta_cap + 1e-4)
    if spec.option_type == "call":
        assert np.all(result.delta_grid[mask] >= -1e-10)
        assert np.all(slope >= -1e-10)
    else:
        assert np.all(result.delta_grid[mask] <= 1e-10)
        assert np.all(slope <= 1e-10)


@pytest.mark.parametrize("spec", SPECS)
def test_price_above_discounted_intrinsic(spec):
    result = price_option(spec, n_space=400, n_time=200)
    forward = spec.spot * np.exp(-spec.div_yield * spec.maturity)
    discounted_strike = spec.strike * np.exp(-spec.rate * spec.maturity)
    if spec.option_type == "call":
        bound = max(forward - discounted_strike, 0.0)
    else:
        bound = max(discounted_strike - forward, 0.0)
    assert result.price >= bound - 1e-3


def test_results_are_scale_invariant():
    small = OptionSpec(100.0, 100.0, 1.0, 0.05, 0.2)
    large = OptionSpec(100_000.0, 100_000.0, 1.0, 0.05, 0.2)
    a = price_option(small, n_space=400, n_time=200)
    b = price_option(large, n_space=400, n_time=200)
    assert b.price / 1000.0 == pytest.approx(a.price, rel=1e-9)
    assert b.delta == pytest.approx(a.delta, rel=1e-9)


@pytest.mark.parametrize("n_time", [1, 2, 3])
def test_default_rannacher_works_with_few_time_steps(n_time):
    result = price_option(SPECS[0], n_space=200, n_time=n_time)
    assert np.isfinite(result.price)
    assert result.price > 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"option_type": "straddle"},
        {"option_type": "Call"},
        {"spot": -1.0},
        {"strike": 0.0},
        {"vol": 0.0},
        {"maturity": 0.0},
        {"spot": float("nan")},
        {"vol": float("inf")},
        {"rate": float("nan")},
        {"div_yield": float("inf")},
    ],
)
def test_invalid_option_spec_rejected(kwargs):
    params = {"spot": 100.0, "strike": 100.0, "maturity": 1.0, "rate": 0.05, "vol": 0.2}
    params.update(kwargs)
    with pytest.raises(ValueError):
        OptionSpec(**params)


def test_non_numeric_option_spec_rejected():
    with pytest.raises(TypeError):
        OptionSpec("100", 100.0, 1.0, 0.05, 0.2)


@pytest.mark.parametrize(
    "kwargs",
    [{"n_space": 400.0}, {"n_time": 200.0}, {"rannacher_steps": 1.5}, {"n_space": True}],
)
def test_non_integer_sizes_rejected(kwargs):
    with pytest.raises(TypeError):
        price_option(SPECS[0], **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_space": 4},
        {"n_time": 0},
        {"rannacher_steps": -1},
        {"width": 0.0},
        {"width": -1.0},
        {"concentration": 0.0},
    ],
)
def test_invalid_solver_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        price_option(SPECS[0], **kwargs)


def test_numpy_integer_sizes_accepted():
    result = price_option(SPECS[0], n_space=np.int64(400), n_time=np.int32(200))
    assert result.price == pytest.approx(price_option(SPECS[0]).price, abs=0.0)


def test_result_is_hashable():
    assert isinstance(hash(price_option(SPECS[0])), int)
