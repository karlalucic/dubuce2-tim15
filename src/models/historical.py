"""Historical simulation: resample past days (owner: Emir Jusić)."""

import numpy as np

from src.models.base import RiskModel


class HistoricalSimulation(RiskModel):
    """Historical simulation: resample past days."""

    def fit(self, train_returns: np.ndarray) -> "HistoricalSimulation":
        raise NotImplementedError

    def sample(self, n: int) -> np.ndarray:
        raise NotImplementedError
