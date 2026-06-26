"""
Yield estimator — converts a fruit count into an estimated weight.
"""
from __future__ import annotations


class YieldEstimator:
    """
    Converts a fruit count into an estimated harvest weight.

    Parameters
    ----------
    avg_weight_g : float
        Average weight per fruit in grams (default: 180 g for an average apple).
    """

    def __init__(self, avg_weight_g: float = 180.0) -> None:
        self.avg_weight_g = avg_weight_g

    def estimate_kg(self, count: int) -> float:
        """Return estimated total weight in kilograms."""
        return round(count * self.avg_weight_g / 1000, 2)

    def estimate_g(self, count: int) -> float:
        """Return estimated total weight in grams."""
        return round(count * self.avg_weight_g, 1)

    def __repr__(self) -> str:
        return f"YieldEstimator(avg_weight_g={self.avg_weight_g})"
