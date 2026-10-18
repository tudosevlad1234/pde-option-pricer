import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from pde_option_pricer import OptionSpec, PricingResult, black_scholes, price_option
from pde_option_pricer.plotting import plot_convergence, plot_solution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Price a European option with a Crank-Nicolson PDE solver")
    parser.add_argument("--spot", type=float, default=100.0)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--maturity", type=float, default=1.0)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--vol", type=float, default=0.20)
    parser.add_argument("--div-yield", type=float, default=0.0)
    parser.add_argument("--type", choices=["call", "put"], default="call", dest="option_type")
    parser.add_argument("--n-space", type=int, default=400)
    parser.add_argument("--n-time", type=int, default=200)
    parser.add_argument("--rannacher-steps", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def print_report(spec: OptionSpec, result: PricingResult) -> None:
    exact = black_scholes(spec)
    print(f"{'':8}{'PDE':>14}{'Closed form':>16}{'Abs. error':>14}")
    for key in ("price", "delta", "gamma", "theta"):
        pde_value = getattr(result, key)
        exact_value = float(exact[key])
        print(f"{key:8}{pde_value:14.6f}{exact_value:16.6f}{abs(pde_value - exact_value):14.2e}")


def main() -> None:
    args = parse_args()
    try:
        spec = OptionSpec(
            spot=args.spot,
            strike=args.strike,
            maturity=args.maturity,
            rate=args.rate,
            vol=args.vol,
            div_yield=args.div_yield,
            option_type=args.option_type,
        )
        result = price_option(
            spec,
            n_space=args.n_space,
            n_time=args.n_time,
            rannacher_steps=args.rannacher_steps,
        )
    except (TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from None

    print_report(spec, result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_solution(spec, result).savefig(args.output_dir / "price_and_greeks.png", dpi=150)
    plot_convergence(spec).savefig(args.output_dir / "convergence.png", dpi=150)
    print(f"Figures saved to {args.output_dir}/")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
