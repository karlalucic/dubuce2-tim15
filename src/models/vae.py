"""Variational autoencoder baseline (owner: Borna Matulić)."""

import numpy as np

from src.models.base import RiskModel


class VAEModel(RiskModel):
    """Variational autoencoder baseline."""

    def fit(self, train_returns: np.ndarray) -> "VAEModel":
        raise NotImplementedError

    def sample(self, n: int) -> np.ndarray:
        raise NotImplementedError
