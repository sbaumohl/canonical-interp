from canonical_interp.optim import sgld_step
from torch.func import (
    stack_module_state,
    vmap,
    grad_and_value as functional_grad_and_value,
)
from copy import deepcopy
from torch.utils.data import DataLoader
from typing import Literal, Callable, List
import torch.nn as nn
import torch as t
from itertools import cycle, product
import logging

logger = logging.getLogger(__name__)


class LLCEstimator:
    """Estimate the Local Learning Coefficient (LLC / RLCT) via SGLD sampling.

    The LLC measures effective model complexity at a point in weight space.
    It is estimated by running parallel SGLD chains from the current weights
    and comparing the time-averaged sampled loss against the initial loss:

        LLC = nbeta * (mean_sampled_loss - init_loss)

    All chains are batched into a single vmapped computation over stacked
    model parameters, avoiding a Python loop over chains.

    Args:
        draws: Number of loss samples to collect per chain after burn-in.
        chains: Number of independent SGLD chains to run in parallel.
        burnin_steps: SGLD steps to discard before collecting draws.
        steps_bw_draws: SGLD steps between consecutive draws (thinning).
        grad_accumulation_steps: Micro-batches to accumulate before each
            SGLD update step.
        verbose: If True, print progress information.
        learning_rate: SGLD step size (epsilon).
        localization: Elastic strength (gamma) pulling params toward init.
        nbeta: Inverse temperature scaling for the gradient. Typically
            n / ln(n) where n is the effective sample size.
    """

    def __init__(
        self,
        draws: int,
        chains: int,
        burnin_steps: int,
        steps_bw_draws: int,
        grad_accumulation_steps: int = 1,
        verbose: bool = False,
        learning_rate: float | int = 1.0,  # epsilon
        localization: float | int = 1.0,  # gamma
        nbeta: float | int = 4.0,
        dtype: t.dtype = t.float32
    ):
        self.draws: int = draws
        self.chains: int = chains
        self.burnin_steps: int = burnin_steps
        self.steps_bw_draws: int = steps_bw_draws
        self.grad_accumulation_steps: int = grad_accumulation_steps
        self.verbose: bool = verbose

        self.learning_rate = learning_rate
        self.localization = localization
        self.nbeta = nbeta

        self.preferred_dtype = dtype

        # internal state for holding mid-run llc data

    @staticmethod
    def _accumulate_grad(
        current: dict[str, t.Tensor] | None, new: dict[str, t.Tensor]
    ) -> dict[str, t.Tensor]:
        """Add micro-batch gradients into a running sum. Returns a new dict if current is None."""
        if current is None:
            return {name: g.clone() for name, g in new.items()}
        for name in new.keys():
            current[name].add_(new[name])
        return current
    
    @staticmethod
    def _normalize_devices(devs: List[str | t.device] | str | t.device) -> List[t.device]:
        if isinstance(devs, List[str | t.device]):
            return [t.device(d) if isinstance(d, str) else d for d in devs] 
        elif isinstance(devs, str):
            return [t.device(devs)]
        elif isinstance(devs, t.device):
            return [devs]
        else:
            raise TypeError(f"`devices` is an unexpected type! Got {type(devs)}")


    def _estimate_llc(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        forward_loss: Callable[
            [nn.Module, dict[str, t.Tensor], dict[str, t.Tensor], t.Tensor, t.Tensor],
            t.Tensor], 
        method: Literal["SGLD", "SGMCMC"],
        chain_idxs: list[int],
        device: str | t.device = t.device("cpu"),
        seed=None
    ):
        """
        Runs a batch of chains on a given device.
        """
        pass

    def estimate_llc(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        forward_loss: Callable[
            [nn.Module, dict[str, t.Tensor], dict[str, t.Tensor], t.Tensor, t.Tensor],
            t.Tensor,
        ],
        method: Literal["SGLD", "SGMCMC"],
        chain_batch: int | Literal["all", "auto"] = "all",
        devices: list[str | t.device] | str | t.device = t.device("cpu"),
        seed=None,
    ):
        """Run SGLD sampling and return per-chain LLC estimates.

        Args:
            model: Template module for functional_call (not modified).
            train_dataloader: Training data; cycled during sampling.
            forward_loss: Pure function (model, params, buffers, x, y) -> scalar loss.
                Must use torch.func.functional_call internally so vmap can
                vectorize over stacked chain parameters. Inputs x and y are
                unpacked from the DataLoader batch.
            method: Sampling method (currently only "SGLD" is implemented).
            seed: Optional RNG seed for reproducibility.

        Returns:
            Tensor of shape (chains,) with per-chain LLC estimates.
        """
        if seed is not None:
            t.manual_seed(seed)
        
        devices = LLCEstimator._normalize_devices(devices)

        if self.preferred_dtype != t.float32 and any([not t.amp.autocast_mode.is_autocast_available(d.type) for d in devices]):
            logger.warning(f"Not all devices support AMP. Will use AMP to autocast to {self.preferred_dtype} onl on devices for which AMP is available.")

        models = [deepcopy(model) for _ in range(self.chains)]
        params, buffers = stack_module_state(models)
        array_log_l = t.zeros((self.chains, self.draws))

        # We're abstracting the number of parallel chains away. Pretend we're running a single chain and let pytorch handle the parallel work

        data_iter = cycle(train_dataloader)
        accumulated_grads = None
        all_grad_fn = vmap(
            functional_grad_and_value(forward_loss, argnums=1),
            in_dims=(None, 0, 0, None, None),
        )

        original_loss = t.zeros((self.chains))
        for batch in train_dataloader:
            x, y = batch[0], batch[1]
            _, loss = all_grad_fn(model, params, buffers, x, y)
            original_loss += loss
        original_loss /= len(train_dataloader)

        initial_params = {name: p[0].clone() for name, p in params.items()}

        # burn-in steps
        for _, grad_step_count in product(
            range(self.burnin_steps), range(self.grad_accumulation_steps)
        ):
            batch = next(data_iter)
            x, y = batch[0], batch[1]
            grads, _ = all_grad_fn(model, params, buffers, x, y)
            accumulated_grads = LLCEstimator._accumulate_grad(accumulated_grads, grads)

            is_grad_step = grad_step_count == self.grad_accumulation_steps - 1

            # at the end of the grad_step inner loop, run backwards/optimzer
            # if left at the default, this will return true every time
            if is_grad_step:
                # optimzer step
                sgld_step(
                    params,
                    accumulated_grads,
                    initial_params,
                    grad_accumulation_steps=self.grad_accumulation_steps,
                    learning_rate=self.learning_rate,
                    nbeta=self.nbeta,
                    localization=self.localization,
                )
                accumulated_grads = None

        # draw steps
        cumulative_loss = t.zeros((self.chains))
        for draw_no in range(self.draws):
            for between_draw_count in range(self.steps_bw_draws):
                accumulated_grads = None
                is_draw = between_draw_count == self.steps_bw_draws - 1

                for grad_step_count in range(self.grad_accumulation_steps):
                    batch = next(data_iter)
                    x, y = batch[0], batch[1]
                    grads, losses = all_grad_fn(model, params, buffers, x, y)
                    accumulated_grads = LLCEstimator._accumulate_grad(
                        accumulated_grads, grads
                    )

                    if is_draw:
                        cumulative_loss += losses / self.grad_accumulation_steps

                # optimzer step
                assert accumulated_grads is not None, "No gradients were accumulated!"
                sgld_step(
                    params,
                    accumulated_grads,
                    initial_params,
                    grad_accumulation_steps=self.grad_accumulation_steps,
                    learning_rate=self.learning_rate,
                    nbeta=self.nbeta,
                    localization=self.localization,
                )

            # TODO callback system
            array_log_l[:, draw_no] = cumulative_loss
            cumulative_loss.zero_()

        avg_sampled_loss = t.mean(array_log_l, dim=-1)
        llc = self.nbeta * (avg_sampled_loss - original_loss)

        return llc

