from dataclasses import dataclass

from ._checks import require_finite, require_positive

OPTION_TYPES = ("call", "put")


@dataclass(frozen=True)
class OptionSpec:
    spot: float
    strike: float
    maturity: float
    rate: float
    vol: float
    div_yield: float = 0.0
    option_type: str = "call"

    def __post_init__(self) -> None:
        if self.option_type not in OPTION_TYPES:
            raise ValueError("option_type must be 'call' or 'put'")
        for name in ("spot", "strike", "maturity", "vol"):
            require_positive(name, getattr(self, name))
        for name in ("rate", "div_yield"):
            require_finite(name, getattr(self, name))

    @property
    def is_call(self) -> bool:
        return self.option_type == "call"
