# Canonical Interp: Efficient Developmental Interpretability

[![PyPI version](https://img.shields.io/pypi/v/canonical-interp)](https://pypi.org/project/canonical-interp/)

> **Note:** This package is under active development (0.X.Y). Breaking changes should be expected between minor versions.

A lean, efficient Local Learning Coefficient (LLC / RLCT) estimator. Rewrite of [Timaeus's `devinterp`](https://github.com/timaeus-research/devinterp). MIT licensed.

## What is the LLC?

The Local Learning Coefficient is a measure of effective model complexity at a specific point in weight space, grounded in Singular Learning Theory. For a model trained to a loss minimum $w^{\ast}$, the LLC estimates the real log canonical threshold (RLCT) $\hat\lambda$:

$$\hat\lambda = n\beta \cdot (\bar L_{\text{SGLD}} - L_0)$$

where $L_0$ is the loss at $w^{\ast}$, $\bar L_{\text{SGLD}}$ is the time-averaged loss of SGLD chains run from $w^{\ast}$, and $n\beta$ is an inverse-temperature factor. Higher LLC means the model is using more of its parameter space at that point. Lower LLC signals degeneracy or symmetry.

## Installation

```bash
pip install canonical-interp
# or with uv:
uv add canonical-interp
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.11.

## Quick start

```python
import torch
from torch.utils.data import DataLoader
from canonical_interp.slt import LLCEstimator

n = len(train_dataset)
nbeta = n / math.log(n)  # standard SLT choice

estimator = LLCEstimator(
    draws=200,
    chains=10,
    burnin_steps=100,
    steps_bw_draws=1,
    learning_rate=1e-5,
    localization=100.0,
    nbeta=nbeta,
)

loader = DataLoader(train_dataset, batch_size=512, shuffle=False,
                    pin_memory=True, num_workers=4, persistent_workers=True)

llc = estimator.estimate_llc(model, loader)
print(f"LLC: {llc.mean():.4f}  (per chain: {llc.tolist()})")
```

`criterion_fn` defaults to `F.cross_entropy`. For classification tasks you don't need to pass anything extra.

## Examples

### Custom loss (regression, custom architectures)

Supply any `(logits, targets) -> scalar` function. The library handles the `functional_call` wrapping internally, so you don't write it.

```python
import torch.nn.functional as F

llc = estimator.estimate_llc(model, loader, criterion_fn=F.mse_loss)
```

For more control (e.g. label smoothing, auxiliary losses), pass a lambda or a named function:

```python
def my_loss(logits, targets):
    return F.cross_entropy(logits, targets, label_smoothing=0.1)

llc = estimator.estimate_llc(model, loader, criterion_fn=my_loss)
```

### Custom dataloader format

If your DataLoader yields dicts or tuples with more than two elements, pass an `unpack_fn` to extract `(x, y)`. Device movement is handled internally.

```python
# DataLoader yields {"input_ids": ..., "labels": ...}
llc = estimator.estimate_llc(
    model, loader,
    unpack_fn=lambda batch: (batch["input_ids"], batch["labels"]),
)
```

### Mixed precision (bfloat16 / float16)

Pass `dtype` to the constructor. Autocast is applied to the forward pass before `vmap` and `torch.compile` see it, so the compiler can fuse dtype casts into the rest of the graph.

```python
estimator = LLCEstimator(
    draws=200, chains=10, burnin_steps=100, steps_bw_draws=1,
    learning_rate=1e-5, localization=100.0, nbeta=nbeta,
    dtype=torch.bfloat16,  # or torch.float16
)

llc = estimator.estimate_llc(model, loader, devices="cuda")
```

### Multi-GPU

Pass a list of devices and set `chain_batch` to control how many chains run on each device at once. Chain batches are distributed round-robin and run in parallel via a `ThreadPoolExecutor`.

```python
llc = estimator.estimate_llc(
    model, loader,
    devices=["cuda:0", "cuda:1"],
    chain_batch=5,   # 5 chains per device call; 10 chains total = 2 calls
)
```

### Reproducibility

```python
llc = estimator.estimate_llc(model, loader, seed=42)
```

A master seed is used to derive independent per-device seeds deterministically, so results are reproducible regardless of how chains are batched across devices.

### Accessing per-chain metrics

After calling `estimate_llc`, use `get_metrics()` to retrieve the full loss trace and per-chain LLC values:

```python
llc = estimator.estimate_llc(model, loader)
metrics = estimator.get_metrics()

metrics["llc_mean"]     # scalar: average LLC across chains
metrics["llc_std"]      # scalar: std of per-chain LLCs
metrics["llcs"]         # [chains]: per-chain LLC estimates
metrics["losses"]       # [chains, draws]: full loss trace per chain
metrics["losses_mean"]  # scalar: mean of final-draw losses
metrics["losses_std"]   # scalar: std of final-draw losses
```

### Hyperparameter grid search

When tuning `epsilon`, `gamma`, and `nbeta`, use `LLCGridSearch` to sweep over ranges and compare results in a single DataFrame.

```python
from canonical_interp import LLCGridSearch, GridSearchConfig

cfg = GridSearchConfig(
    epsilon=(1e-6, 1e-4),   # range to sweep
    gamma=(10.0, 500.0),    # range to sweep
    nbeta=nbeta,            # fixed value
    estimates_per_dim=5,    # 5 values per range, so 5x5x1 = 25 runs
    draws=200,
    chains=8,
    burnin_steps=100,
)

gs = LLCGridSearch(cfg)
df = gs.run_grid_search(model, loader, devices="cuda")
print(df)
#   epsilon   gamma   nbeta  llc_mean  llc_std  loss_mean  loss_std
# 0  1e-06    10.0    57.6     2.31     0.12      0.041     0.003
# 1  1e-06   132.5    57.6     2.45     0.09      0.039     0.002
# ...
```

All options supported by `LLCEstimator.estimate_llc` (multi-GPU, compilation, custom loss, `unpack_fn`, seed) pass through to each grid point.

## Performance by default

LLC estimation is compute-intensive: it requires thousands of forward+backward passes across many parallel chains. This library is built around the principle that sensible defaults should leave no performance on the table.

| Feature | What it does |
|---|---|
| **`vmap` over chains** | All chains in a batch run as a single fused kernel with no Python loop and no per-chain overhead |
| **`torch.compile`** | The vmapped grad function is JIT-compiled by default (`compile=True`) |
| **Autocast inside the transform** | Autocast wraps the forward pass *before* `vmap`/`grad` see it, so the compiler can fuse dtype casts rather than treating them as opaque boundaries |
| **DataLoader warnings** | The estimator warns when `pin_memory=False`, `num_workers=0`, or `persistent_workers=False`, all of which cause avoidable stalls when the dataloader is cycled for the full SGLD run |

### Recommended DataLoader settings for GPU runs

```python
loader = DataLoader(
    dataset,
    batch_size=512,
    shuffle=False,       # non-shuffled for deterministic LLC; shuffle for training
    pin_memory=True,     # faster CPU→GPU transfer
    num_workers=4,       # overlap data loading with GPU compute
    persistent_workers=True,  # avoid worker respawn between SGLD epochs
)
```

## Hyperparameter guide

| Parameter | Typical range | Effect |
|---|---|---|
| `nbeta` | `n / log(n)` | Inverse temperature; scales the LLC estimate. Use `nbeta_from_effective_size(n)` or compute directly. |
| `learning_rate` | `1e-6` – `1e-4` | SGLD step size. Too large → divergence; too small → chain doesn't move. |
| `localization` | `1` – `1000` | Elastic pull toward initial weights. Higher values keep chains near $w^{\ast}$, giving tighter LLC estimates. |
| `burnin_steps` | 100 – 500 | Steps discarded before draws. Should cover transient behaviour. |
| `draws` | 100 – 500 | Samples per chain used to estimate $\bar L$. More draws → lower variance. |
| `chains` | 5 – 20 | Independent chains. More chains → lower variance; all run in parallel via vmap. |

## How it works

All chain parameters are stacked into a single batched tensor and a single functional forward+backward is vmapped over them, mapping naturally onto GPU parallelism. The statistical procedure follows the SGLD-based LLC estimator from Singular Learning Theory. Results can be validated against known closed-form RLCTs (see `test_known_rlct.py`).

This library was built on top of ideas from [devinterp](https://github.com/timaeus-research/devinterp).
