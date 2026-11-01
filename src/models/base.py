"""Shared interface for all risk models (generative and classical)."""

import numpy as np


class RiskModel:
    """A model of the joint distribution of daily returns for 10 industries.

    Every model is fitted on a window of past returns and then sampled to
    produce return scenarios, from which VaR/ES are estimated.
    """

    n_assets: int = 10

    def fit(self, train_returns: np.ndarray) -> "RiskModel":
        """Fit the model on train_returns of shape (n_days, 10)."""
        raise NotImplementedError

    def sample(self, n: int) -> np.ndarray:
        """Return n scenarios of daily returns, shape (n, 10)."""
        raise NotImplementedError
