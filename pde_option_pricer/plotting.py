import matplotlib.pyplot as plt

from .analytic import black_scholes
from .solver import PricingResult, price_option
from .spec import OptionSpec


def plot_solution(spec: OptionSpec, result: PricingResult, spot_range: tuple[float, float] = (0.5, 1.5)):
    low = spot_range[0] * spec.strike
    high = spot_range[1] * spec.strike
    mask = (result.spot_grid >= low) & (result.spot_grid <= high)
    s = result.spot_grid[mask]
    exact = black_scholes(spec, s)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = [("price", "Price"), ("delta", "Delta"), ("gamma", "Gamma"), ("theta", "Theta")]

    for ax, (key, label) in zip(axes.ravel(), panels, strict=True):
        ax.plot(s, getattr(result, f"{key}_grid")[mask], lw=1.8, label="Crank-Nicolson PDE")
        ax.plot(s, exact[key], "--", lw=1.4, label="Black-Scholes closed form")
        ax.axvline(spec.strike, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("Spot")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    axes[0, 0].legend()
    fig.suptitle(
        f"European {spec.option_type}  |  K={spec.strike:g}, T={spec.maturity:g}, "
        f"r={spec.rate:g}, q={spec.div_yield:g}, vol={spec.vol:g}"
    )
    return fig


def plot_convergence(spec: OptionSpec, sizes: tuple[int, ...] = (50, 100, 200, 400, 800, 1600)):
    exact = black_scholes(spec)
    errors = {"price": [], "delta": [], "gamma": []}

    for n in sizes:
        result = price_option(spec, n_space=n, n_time=n)
        for key in errors:
            errors[key].append(max(abs(getattr(result, key) - float(exact[key])), 1e-16))

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for key, values in errors.items():
        ax.loglog(sizes, values, marker="o", label=key.capitalize())

    ax.set_xlabel("Space steps = time steps")
    ax.set_ylabel("Absolute error vs closed form")
    ax.set_title("Convergence")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    return fig
