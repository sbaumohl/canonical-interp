from canonical_interp.slt import LLCEstimator
from torch.utils.data import DataLoader
from typing import Tuple, Callable, Literal
from dataclasses import dataclass
from itertools import product
import pandas as pd
import numpy as np
import logging
import torch.nn as nn
import torch as t
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GridSearchConfig:
    epsilon: Tuple[float, float] | float
    gamma: Tuple[float, float] | float
    nbeta: Tuple[float, float] | float
    estimates_per_dim: int
    draws: int = 100
    chains: int = 8
    burnin_steps: int = 100
    steps_bw_draws: int = 1
    grad_accumulation_steps: int = 1


class LLCGridSearch:
    def __init__(self, cfg: GridSearchConfig):
        self.cfg = cfg

        if (
            isinstance(self.cfg.epsilon, float)
            and isinstance(self.cfg.gamma, float)
            and isinstance(self.cfg.nbeta, float)
        ):
            logger.info(
                "No hyperparameter ranges were provided. Running an `LLCEstimator` directly may be more ergonomic."
            )

    @staticmethod
    def _expand_param(val: Tuple[float, float] | float, n: int) -> list[float]:
        if isinstance(val, (int, float)):
            return [float(val)]
        lo, hi = val
        return np.linspace(lo, hi, n).tolist()

    def run_grid_search(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        criterion_fn: Callable[[t.Tensor, t.Tensor], t.Tensor] = F.cross_entropy,
        chain_batch: int | Literal["all"] = "all",
        devices: list[str | t.device] | str | t.device = t.device("cpu"),
        seed=None,
        compile: bool = True,
        unpack_fn: Callable[..., Tuple[t.Tensor, t.Tensor]] | None = None,
        show_progress: bool = True,
        preferred_dtype: t.dtype = t.float32,
    ) -> pd.DataFrame:
        epsilons = self._expand_param(self.cfg.epsilon, self.cfg.estimates_per_dim)
        gammas = self._expand_param(self.cfg.gamma, self.cfg.estimates_per_dim)
        nbetas = self._expand_param(self.cfg.nbeta, self.cfg.estimates_per_dim)

        rows = []
        original_loss = None
        for eps, gam, nb in product(epsilons, gammas, nbetas):
            logger.info(f"Running LLC estimate: epsilon={eps}, gamma={gam}, nbeta={nb}")
            estimator = LLCEstimator(
                draws=self.cfg.draws,
                chains=self.cfg.chains,
                burnin_steps=self.cfg.burnin_steps,
                steps_bw_draws=self.cfg.steps_bw_draws,
                grad_accumulation_steps=self.cfg.grad_accumulation_steps,
                learning_rate=eps,
                localization=gam,
                nbeta=nb,
                dtype=preferred_dtype,
            )
            estimator.estimate_llc(
                model=model,
                train_dataloader=train_dataloader,
                criterion_fn=criterion_fn,
                chain_batch=chain_batch,
                devices=devices,
                seed=seed,
                compile=compile,
                unpack_fn=unpack_fn,
                show_progress=show_progress,
                current_loss=original_loss,
            )

            if original_loss is None and estimator.original_loss is not None:
                original_loss = estimator.original_loss

            metrics = estimator.get_metrics()
            rows.append(
                dict(
                    epsilon=eps,
                    gamma=gam,
                    nbeta=nb,
                    llc_mean=metrics["llc_mean"].item(),
                    llc_std=metrics["llc_std"].item(),
                    loss_mean=metrics["losses_mean"].item(),
                    loss_std=metrics["losses_std"].item(),
                )
            )

        return pd.DataFrame(rows)
