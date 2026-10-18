import math
import numbers


def require_finite(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def require_positive(name: str, value) -> None:
    require_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def require_int(name: str, value, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
