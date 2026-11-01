"""Multivariate normal with a Ledoit-Wolf shrinkage covariance (owner: Emir Jusić)."""

import numpy as np

from src.models.base import RiskModel


class GaussianLedoitWolf(RiskModel):
    """Multivariate normal with a Ledoit-Wolf shrinkage covariance."""

    def fit(self, train_returns: np.ndarray) -> "GaussianLedoitWolf":
        raise NotImplementedError

    def sample(self, n: int) -> np.ndarray:
        raise NotImplementedError
