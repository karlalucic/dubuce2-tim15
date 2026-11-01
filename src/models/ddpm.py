"""Denoising diffusion probabilistic model with an MLP denoiser (owner: Karla Lučić)."""

import numpy as np

from src.models.base import RiskModel


class DDPMModel(RiskModel):
    """Denoising diffusion probabilistic model with an MLP denoiser."""

    def fit(self, train_returns: np.ndarray) -> "DDPMModel":
        raise NotImplementedError

    def sample(self, n: int) -> np.ndarray:
        raise NotImplementedError
