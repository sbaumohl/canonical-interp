from canonical_interp import nbeta_from_effective_size
import tqdm
from concurrent.futures import ThreadPoolExecutor
import math
from canonical_interp.optim import sgld_step
from torch.func import (
    stack_module_state,
    vmap,
    grad_and_value as functional_grad_and_value,
    functional_call,
)
from copy import deepcopy
from torch.utils.data import DataLoader
from typing import Literal, Callable, List, Tuple
import torch.nn as nn
import torch as t
import torch.nn.functional as F
from itertools import cycle, product, batched
from functools import partial
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
        steps_bw_draws: int = 1,
        grad_accumulation_steps: int = 1,
        verbose: bool = False,
        learning_rate: float | int = 0.001,  # epsilon
        localization: float | int = 5.0,  # gamma
        nbeta: float | int = nbeta_from_effective_size(32),
        dtype: t.dtype = t.float32,
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
        self._metrics = dict()

        if self.burnin_steps < self.draws:
            logger.warning(
                "Designated fewer burn-in steps than draws. It is recommended to increase burn-in steps until we reach a loss plateau."
            )

        if not (1.0 < self.localization < 10.0):
            logger.warning(
                f"Localization term is {self.localization}. It is recommended to avoid extreme localization parameters. Try lowering epsilon or increasing draws or burn-in steps."
            )

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
    def _normalize_devices(
        devs: List[str | t.device] | str | t.device,
    ) -> List[t.device]:
        if isinstance(devs, list):
            return [t.device(d) if isinstance(d, str) else d for d in devs]
        elif isinstance(devs, str):
            return [t.device(devs)]
        elif isinstance(devs, t.device):
            return [devs]
        else:
            raise TypeError(f"`devices` is an unexpected type! Got {type(devs)}")

    @staticmethod
    def _construct_forward_pass(
        criterion_fn: Callable[[t.Tensor, t.Tensor], t.Tensor],
    ) -> Callable[
        [nn.Module, dict[str, t.Tensor], dict[str, t.Tensor], t.Tensor, t.Tensor],
        t.Tensor,
    ]:
        """Wrap a user-supplied criterion into the signature required by vmap.

        ``vmap`` needs a pure function with explicit ``params`` and ``buffers``
        arguments so it can vectorize over stacked chain states.  This helper
        lets callers supply a simple ``criterion_fn(logits, y) -> scalar``
        instead of writing the ``functional_call`` boilerplate themselves.

        Args:
            criterion_fn: Loss function taking ``(logits, targets)`` and
                returning a scalar tensor.

        Returns:
            A pure function ``(model, params, buffers, x, y) -> scalar`` that
            runs a forward pass via :func:`~torch.func.functional_call` and
            then applies ``criterion_fn``.
        """

        def wrapped_forward(
            model: nn.Module,
            params: dict[str, t.Tensor],
            buffers: dict[str, t.Tensor],
            x: t.Tensor,
            y: t.Tensor,
        ) -> t.Tensor:
            return criterion_fn(functional_call(model, (params, buffers), (x,)), y)

        return wrapped_forward

    def get_metrics(self) -> dict[str, t.Tensor] | None:
        return self._metrics

    def clear_metrics(self):
        self._metrics = dict()

    def _estimate_llc(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        forward_pass: Callable[
            [nn.Module, dict[str, t.Tensor], dict[str, t.Tensor], t.Tensor, t.Tensor],
            t.Tensor,
        ],
        unpack_fn: Callable[..., Tuple[t.Tensor, t.Tensor]],
        chain_idxs: list[int],
        device: t.device = t.device("cpu"),
        thread_idx: int | None = None,
        show_progress: bool = True,
        seed=None,
        compile: bool = True,
        targeted_params: frozenset[str] | None = None,
    ):
        """Run a batch of SGLD chains on a single device and record loss draws.

        Results are written directly into ``self.array_log_l`` and
        ``self.original_loss`` at positions ``chain_idxs``.

        Args:
            model: Template module used by ``functional_call``; not modified.
            train_dataloader: Data source; cycled throughout sampling.
            forward_pass: Pure ``(model, params, buffers, x, y) -> scalar``
                function, typically produced by :meth:`_construct_forward_pass`.
            unpack_fn: Extracts ``(x, y)`` from a raw dataloader batch.
                Device movement is handled internally; this function should
                only perform extraction and any necessary preprocessing.
            chain_idxs: Indices into the global chain arrays that this call
                is responsible for.
            device: Device on which to run all computation for these chains.
            thread_idx: Index of this thread in a multi-device run, used to
                position tqdm bars so they don't overwrite each other.  ``None``
                for single-batch runs.
            show_progress: If True, display a tqdm progress bar.
            seed: Optional RNG seed for reproducibility on this device.
            compile: If True, JIT-compile the vmapped grad function with
                ``torch.compile``.
            targeted_params: Optional frozenset of validated parameter names
                to restrict SGLD updates to. Passed through to
                :func:`~canonical_interp.optim.sgld_step`.
        """
        if seed is not None:
            t.manual_seed(seed)

        num_chains = len(chain_idxs)

        moved_model = model.to(device)
        models = [deepcopy(moved_model) for _ in range(num_chains)]
        params, buffers = stack_module_state(models)
        del models

        # Autocast must wrap the function *before* vmap/grad see it so the
        # dtype cast is part of the function graph all three transforms operate on.
        if (
            self.preferred_dtype != t.float32
            and t.amp.autocast_mode.is_autocast_available(device.type)
        ):
            _fp = forward_pass

            def forward_pass(model, params, buffers, x, y):
                with t.amp.autocast(
                    device_type=device.type, dtype=self.preferred_dtype
                ):
                    return _fp(model, params, buffers, x, y)

        # We're abstracting the number of parallel chains away. Pretend we're running a single chain and let pytorch handle the parallel work
        data_iter = cycle(train_dataloader)
        accumulated_grads = None
        all_grad_fn = vmap(
            functional_grad_and_value(forward_pass, argnums=1),
            in_dims=(None, 0, 0, None, None),
            randomness="different",
        )

        all_grad_fn = t.compile(all_grad_fn, disable=not compile)

        original_loss = t.zeros((num_chains), device=device)
        for batch in train_dataloader:
            x, y = unpack_fn(batch)
            x, y = x.to(device=device), y.to(device=device)
            _, loss = all_grad_fn(model, params, buffers, x, y)
            original_loss += loss
        original_loss /= len(train_dataloader)
        self.original_loss[chain_idxs] = original_loss.cpu()

        initial_params = {name: p[0].clone() for name, p in params.items()}

        pbar = tqdm.tqdm(
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
            total=(self.burnin_steps + self.draws * self.steps_bw_draws)
            * self.grad_accumulation_steps,
            position=thread_idx if thread_idx is not None else 0,
            desc=f"Thread {thread_idx}" if thread_idx is not None else None,
        )

        # burn-in steps
        for _, grad_step_count in product(
            range(self.burnin_steps), range(self.grad_accumulation_steps)
        ):
            batch = next(data_iter)
            x, y = unpack_fn(batch)
            x, y = x.to(device=device), y.to(device=device)
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
                    targeted_params=targeted_params,
                )
                accumulated_grads = None
            pbar.update(1)

        # draw steps
        cumulative_loss = t.zeros((num_chains), device=device, requires_grad=False)
        for draw_no in range(self.draws):
            for between_draw_count in range(self.steps_bw_draws):
                accumulated_grads = None
                is_draw = between_draw_count == self.steps_bw_draws - 1

                for grad_step_count in range(self.grad_accumulation_steps):
                    batch = next(data_iter)
                    x, y = unpack_fn(batch)
                    x, y = x.to(device=device), y.to(device=device)
                    grads, losses = all_grad_fn(model, params, buffers, x, y)
                    accumulated_grads = LLCEstimator._accumulate_grad(
                        accumulated_grads, grads
                    )

                    if is_draw:
                        cumulative_loss.add_(
                            losses.detach() / self.grad_accumulation_steps
                        )

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
                    targeted_params=targeted_params,
                )

                pbar.update(self.grad_accumulation_steps)

            self.array_log_l[chain_idxs, draw_no] = cumulative_loss.cpu()
            cumulative_loss.zero_()

        pbar.close()

    def estimate_llc(
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
        targeted_params: list[str] | None = None,
    ):
        """Run SGLD sampling and return per-chain LLC estimates.

        Args:
            model: Template module used by ``functional_call``; not modified.
            train_dataloader: Training data source; cycled during sampling.
            criterion_fn: Loss function ``(logits, targets) -> scalar``.
                The library wraps this with ``functional_call`` internally,
                so callers do *not* need to reference ``params`` or ``buffers``.
            chain_batch: Number of chains to run per device call.  ``"all"``
                (default) runs all chains in a single call on the first device.
            devices: Device or list of devices to distribute chain batches
                across.  Extra devices beyond the number of chain batches are
                ignored with a warning.
            seed: Optional integer seed for reproducible sampling.  Derived
                per-call seeds are generated from this master seed.
            compile: If True (default), JIT-compile the vmapped grad function
                with ``torch.compile`` for faster sampling.
            unpack_fn: Optional function ``(batch) -> (x, y)`` to extract
                inputs and targets from a raw dataloader batch.  Defaults to
                ``lambda batch: (batch[0], batch[1])``.  Device movement is
                handled internally; this function should only extract and
                preprocess data.
            show_progress: If True (default), display a tqdm progress bar
                tracking burn-in and draw steps.  In multi-device runs each
                thread gets its own bar at a separate terminal position.
            targeted_params: Optional list of parameter names to restrict
                SGLD sampling to. Only these parameters will be updated
                during sampling; all others are frozen. If ``None``
                (default), all model parameters are updated. Names must
                match those returned by ``model.named_parameters()``.

        Returns:
            Tensor of shape ``(chains,)`` with per-chain LLC estimates.
        """

        device_list = LLCEstimator._normalize_devices(devices)
        if chain_batch == "all":
            chain_batch = self.chains
        num_chain_batches = math.ceil(float(self.chains) / float(chain_batch))

        # master seed -> per-call seed -> reproducible results
        seeds = [None] * num_chain_batches
        if seed is not None:
            rng = t.Generator()
            rng.manual_seed(seed)
            seeds = t.randint(0, 2**31, (num_chain_batches,), generator=rng).tolist()

        if len(device_list) > num_chain_batches:
            logger.warning(
                f"More devices ({len(device_list)}) than chain batches ({chain_batch}). The first {chain_batch} devices will be used."
            )

        if self.preferred_dtype != t.float32 and any(
            [not t.amp.autocast_mode.is_autocast_available(d.type) for d in device_list]
        ):
            logger.warning(
                f"Not all devices support AMP. Will use AMP to autocast to {self.preferred_dtype} only on devices for which AMP is available."
            )

        if not train_dataloader.pin_memory:
            logger.warning(
                "`train_dataloader` has `pin_memory=False`. Enabling pinned memory speeds up CPU-GPU data transfer."
            )

        if train_dataloader.num_workers == 0:
            logger.warning(
                "`train_dataloader` has `num_workers=0`. Increasing worker count reduces downtime between forward passes."
            )

        if train_dataloader.num_workers > 0 and not train_dataloader.persistent_workers:
            logger.warning(
                "`train_dataloader` has `persistent_workers=False`. With `num_workers > 0`, workers respawn between epochs — set `persistent_workers=True` to avoid repeated startup overhead during SGLD cycling."
            )

        self.array_log_l = t.zeros((self.chains, self.draws))
        self.original_loss = t.zeros((self.chains,))

        filtered_param_targets: frozenset[str] | None = None
        if targeted_params is not None and len(targeted_params) == 0:
            logger.error("`targeted_params` has no listed parameters. Exiting...")
            return
        elif targeted_params is not None:
            targets = set(targeted_params)
            model_names = {name for name, _ in model.named_parameters(recurse=True)}
            matched = targets & model_names
            missing = targets - model_names
            if len(matched) == 0:
                logger.error(
                    "`targeted_params` has no parameters that exist in `model`! Exiting..."
                )
                return

            if len(missing) > 0:
                logger.warning(f"Parameters {missing} not found in `model`!")
            filtered_param_targets = frozenset(matched)

        # prep backward/forward pass
        forward_pass = LLCEstimator._construct_forward_pass(criterion_fn)
        if unpack_fn is None:
            unpack_fn = lambda batch: (batch[0], batch[1])

        if num_chain_batches == 1:
            self._estimate_llc(
                model,
                train_dataloader,
                forward_pass,
                unpack_fn,
                list(range(self.chains)),
                device=device_list[0],
                seed=seeds[0],
                compile=compile,
                show_progress=show_progress,
                targeted_params=filtered_param_targets,
            )
        else:
            partial_chain_call = partial(
                self._estimate_llc,
                model,
                train_dataloader,
                forward_pass,
                unpack_fn,
                compile=compile,
                show_progress=show_progress,
                targeted_params=filtered_param_targets,
            )

            chains_device_iter = zip(
                map(list, batched(range(self.chains), chain_batch)), cycle(device_list)
            )  # (chain_idxs, t.device)

            futures = {}
            with ThreadPoolExecutor(max_workers=len(device_list)) as executor:
                for thread_idx, (chain_idxs, device) in enumerate(chains_device_iter):
                    future = executor.submit(
                        lambda idxs, device, t_idx=thread_idx, s=seeds[
                            thread_idx
                        ]: partial_chain_call(
                            idxs, device=device, thread_idx=t_idx, seed=s
                        ),
                        chain_idxs,
                        device,
                    )
                    futures[thread_idx] = future

            first_exc = None
            first_exc_thread = None
            for thread_idx, future in futures.items():
                exc = future.exception()
                if exc is not None:
                    logger.warning(f"Exception in thread {thread_idx}: {exc}")
                    if first_exc is None:
                        first_exc = exc
                        first_exc_thread = thread_idx
            if first_exc is not None:
                raise RuntimeError(
                    f"Exception raised in thread {first_exc_thread}"
                ) from first_exc

        avg_sampled_loss = t.mean(self.array_log_l, dim=-1)
        llcs = self.nbeta * (avg_sampled_loss - self.original_loss)
        mean_llc = llcs.mean()
        self._metrics = dict(
            llcs=llcs,  # [chains] : llc estimate for each chain
            llc_mean=mean_llc,  # avg llc, our lambda-hat true LLC estimate
            llc_std=llcs.std(),  # std of llc estimates
            losses=self.array_log_l.clone(),  # [chains, draws] : full loss trace
            losses_mean=self.array_log_l[:, -1].mean(),
            losses_std=self.array_log_l[:, -1].std(),
        )
        return mean_llc
