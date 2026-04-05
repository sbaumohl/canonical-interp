"""Test LLC estimation against devinterp's reference implementation.

Follows the devinterp MNIST example: train a small MLP on MNIST, then
estimate the LLC with both our LLCEstimator and devinterp's
estimate_learning_coeff_with_summary, and compare.

Usage:
    uv run python test_llc.py
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from canonical_interp.slt import LLCEstimator
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.optim.sgld import SGLD
from devinterp.utils import default_nbeta, evaluate_ce

# -- Model (matches devinterp MNIST example) ----------------------------------


class MNISTNet(nn.Module):
    def __init__(
        self,
        hidden_layer_sizes=[1024, 1024],
        input_dim=28 * 28,
        output_dim=10,
        activation=F.relu,
        with_bias=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.layer_sizes = [input_dim] + hidden_layer_sizes + [output_dim]
        self.activation = activation
        self.with_bias = with_bias
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            dim_in, dim_out = self.layer_sizes[i : i + 2]
            self.layers.append(nn.Linear(dim_in, dim_out, bias=self.with_bias).float())

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


# -- Training loop -------------------------------------------------------------


def train_model(model, train_loader, epochs=10, lr=0.05, device="cpu"):
    model = model.to(device)
    optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  acc={acc:.1f}%")

    return model


# -- Tests ---------------------------------------------------------------------


def test_llc_positive(model, loader, nbeta):
    """LLC should be positive for a trained model (sampled loss > init loss)."""
    estimator = LLCEstimator(
        draws=50,
        chains=2,
        burnin_steps=0,
        steps_bw_draws=1,
        learning_rate=1e-5,
        localization=100.0,
        nbeta=nbeta,
    )
    llc = estimator.estimate_llc(model, loader, seed=42)
    print(f"  Per-chain LLC: {llc}")
    mean_llc = llc.mean().item()
    print(f"  Mean LLC: {mean_llc:.4f}")
    assert mean_llc > 0, f"LLC should be positive, got {mean_llc}"
    print("  PASSED: LLC is positive")
    return mean_llc


def test_llc_deterministic(model, loader, nbeta):
    """Same seed should give same LLC."""
    kwargs = dict(
        draws=20,
        chains=2,
        burnin_steps=0,
        steps_bw_draws=1,
        learning_rate=1e-5,
        localization=100.0,
        nbeta=nbeta,
    )
    est = LLCEstimator(**kwargs)
    llc1 = est.estimate_llc(model, loader, seed=42)
    llc2 = est.estimate_llc(model, loader, seed=42)
    diff = (llc1 - llc2).abs().max().item()
    print(f"  Max abs diff between runs: {diff:.2e}")
    assert diff < 1e-5, f"Same seed should give same LLC, got diff={diff}"
    print("  PASSED: deterministic with same seed")


def test_compare_devinterp(model, loader, nbeta, device="cpu"):
    """Compare our LLC estimate against devinterp's reference implementation.

    Uses the same hyperparameters for both. We allow a generous tolerance
    since the implementations use different sampling mechanics (vmapped
    functional vs. deepcopy per chain) and different init_loss computation.
    """
    num_draws = 100
    num_chains = 2
    lr = 1e-5
    loc = 100.0

    # -- Our estimator --
    est = LLCEstimator(
        draws=num_draws,
        chains=num_chains,
        burnin_steps=0,
        steps_bw_draws=1,
        learning_rate=lr,
        localization=loc,
        nbeta=nbeta,
    )
    our_llc = (
        est.estimate_llc(model, loader, seed=42)
        .mean()
        .item()
    )

    # -- devinterp reference --
    di_results = estimate_learning_coeff_with_summary(
        model,
        loader,
        evaluate=evaluate_ce,
        sampling_method=SGLD,
        optimizer_kwargs=dict(lr=lr, localization=loc, nbeta=nbeta),
        num_draws=num_draws,
        num_chains=num_chains,
        num_burnin_steps=0,
        num_steps_bw_draws=1,
        device=device,
        seed=42,
        verbose=False,
    )
    di_llc = di_results["llc/mean"]

    print(f"  Our LLC:       {our_llc:.4f}")
    print(f"  devinterp LLC: {di_llc:.4f}")

    # Both should be positive and in the same ballpark.
    # Exact match is not expected due to implementation differences:
    # - init_loss computation (full dataset vs multi-batch)
    # - sampling noise (vmap vs per-chain deepcopy)
    # - dataloader iteration order
    assert our_llc > 0, f"Our LLC should be positive, got {our_llc}"
    assert di_llc > 0, f"devinterp LLC should be positive, got {di_llc}"

    # Check they're within an order of magnitude
    ratio = our_llc / di_llc if di_llc != 0 else float("inf")
    print(f"  Ratio (ours/devinterp): {ratio:.2f}")
    assert 0.1 < ratio < 10.0, f"LLCs differ by more than 10x: ratio={ratio:.2f}"
    print("  PASSED: within order of magnitude of devinterp")


def test_loss_trace_is_finite(model, loader, nbeta):
    """Loss trace should contain only finite, positive values.

    Catches numerical instability (NaN/inf) from e.g. too-large learning rate.
    All cross-entropy values are strictly positive, so any non-finite or
    non-positive value indicates a bug or bad hyperparameters.
    """
    est = LLCEstimator(
        draws=50,
        chains=2,
        burnin_steps=0,
        steps_bw_draws=1,
        learning_rate=1e-5,
        localization=100.0,
        nbeta=nbeta,
    )
    llc = est.estimate_llc(model, loader, seed=42)
    assert t.isfinite(
        est.array_log_l
    ).all(), f"Loss trace contains non-finite values: {est.array_log_l}"
    assert (
        est.array_log_l > 0
    ).all(), f"Loss trace should be positive for a trained model: {est.array_log_l}"
    assert t.isfinite(llc).all(), f"LLC contains non-finite values: {llc}"
    assert (llc > 0).all(), f"LLC should be positive for a trained model: {llc}"
    print(f"  LLC values: {llc.tolist()}")
    print("  PASSED: loss trace is finite and positive")


def test_localization_keeps_llc_bounded(model, loader, nbeta):
    """Higher localization should give lower LLC (tighter around init)."""
    llcs = {}
    for loc in [10.0, 1000.0]:
        est = LLCEstimator(
            draws=50,
            chains=2,
            burnin_steps=0,
            steps_bw_draws=1,
            learning_rate=1e-5,
            localization=loc,
            nbeta=nbeta,
        )
        llc = est.estimate_llc(model, loader, seed=42)
        llcs[loc] = llc.mean().item()
        print(f"  localization={loc:>7.1f}  mean_llc={llcs[loc]:.4f}")

    assert llcs[1000.0] < llcs[10.0], (
        f"Higher localization should give lower LLC: "
        f"llc@10={llcs[10.0]:.4f}, llc@1000={llcs[1000.0]:.4f}"
    )
    print("  PASSED: higher localization gives lower LLC")


# -- Main ----------------------------------------------------------------------


def main():
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load MNIST (use a subset for speed on CPU)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Use a small subset for tractable CPU testing
    subset_size = 10_000
    train_subset = Subset(train_dataset, range(subset_size))

    batch_size = 1024
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    nbeta = default_nbeta(train_loader)
    print(f"nbeta (batch_size={batch_size}): {nbeta:.2f}")

    # Train model
    print("\nTraining MNISTNet...")
    model = MNISTNet(hidden_layer_sizes=[256, 256])  # smaller for CPU speed
    model = train_model(model, train_loader, epochs=10, lr=0.05, device=device)
    model = model.to("cpu")  # ensure everything is on CPU for functional transforms
    model.eval()

    # Use a non-shuffled loader for deterministic LLC estimation
    eval_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    print("\n--- Test: LLC should be positive ---")
    test_llc_positive(model, eval_loader, nbeta)

    print("\n--- Test: deterministic with same seed ---")
    test_llc_deterministic(model, eval_loader, nbeta)

    print("\n--- Test: compare against devinterp ---")
    test_compare_devinterp(model, eval_loader, nbeta, device="cpu")

    print("\n--- Test: loss trace is finite and positive ---")
    test_loss_trace_is_finite(model, eval_loader, nbeta)

    print("\n--- Test: higher localization gives lower LLC ---")
    test_localization_keeps_llc_bounded(model, eval_loader, nbeta)

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
