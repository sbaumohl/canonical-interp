import torch as t


@t.no_grad()
def sgld_step(
    params: dict[str, t.Tensor],
    grads: dict[str, t.Tensor],
    initial_weights: dict[str, t.Tensor],
    grad_accumulation_steps: int = 1,
    learning_rate: float = 1.0,
    nbeta: float = 1.0,
    localization: float = 0.0,
    weight_decay: float = 0.0,
    noise_level: float = 1.0,
):
    """Perform a single Stochastic Gradient Langevin Dynamics (SGLD) update step in-place.

    Implements the discretized Langevin diffusion update rule:

        w_{t+1} = w_t - (lr/2) * [nbeta * grad/accum + gamma * (w_t - w_0) + wd * w_t]
                  + sqrt(lr) * noise_level * N(0, I)

    where ``gamma`` is the localization strength and ``wd`` is weight decay.

    The noise injection ensures the iterates approximate samples from the
    tempered posterior ``exp(-nbeta * L(w)) * prior(w)`` rather than
    converging to a point estimate.

    Args:
        params: Named parameter tensors to update. Modified in-place via
            :meth:`~torch.Tensor.set_`.  May have a leading batch dimension
            when used with vmapped chains.
        grads: Gradients of the loss with respect to each parameter in
            ``params``, keyed by the same names.
        initial_weights: Snapshot of the parameters at the start of sampling,
            used by the localization (elastic) term to keep the chain near the
            original weights ``w_0``.
        grad_accumulation_steps: Number of micro-batches whose gradients were
            summed into ``grads``. The accumulated gradient is divided by this
            value before scaling by ``nbeta``.
        learning_rate: Step size ``epsilon``. Controls both the deterministic
            gradient step (scaled by ``lr/2``) and the noise magnitude (scaled
            by ``sqrt(lr)``).
        nbeta: Inverse temperature factor applied to the gradient, controlling
            how strongly the loss landscape shapes the posterior.  Typically
            set to ``n / ln(n)`` where *n* is the effective sample size.
        localization: Strength ``gamma`` of the elastic term that pulls
            parameters back toward ``initial_weights``.  Set to 0 to disable.
        weight_decay: Standard L2 weight decay coefficient.  Set to 0 to
            disable.
        noise_level: Standard deviation multiplier for the injected Gaussian
            noise (default 1.0).
    """
    for name, p in params.items():
        delta_weights = (grads[name] / grad_accumulation_steps) * nbeta

        if localization > 0.0:
            delta_weights += localization * (p - initial_weights[name])

        if weight_decay > 0.0:
            delta_weights += weight_decay * p

        noise = t.randn_like(p) * noise_level

        params[name].set_(
            p - 0.5 * learning_rate * delta_weights + (learning_rate**0.5) * noise
        )
