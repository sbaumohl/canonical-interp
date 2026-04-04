"""Test LLC estimation against models with known closed-form RLCTs.

1. Normal crossing (monomial): L(w) = w^{2k}, true RLCT = 1/(2k).
2. Reduced rank regression: H x K matrix of true rank r,
   true RLCT = r(H + K - r) / 2.

These are gold-standard tests because the ground truth is exact, not
estimated by another sampling implementation.

Usage:
    uv run python test_known_rlct.py
"""

import torch as t
import torch.nn as nn
from torch.func import functional_call
from torch.utils.data import DataLoader, TensorDataset

from canonical_interp.slt import LLCEstimator
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.optim.sgld import SGLD


# =============================================================================
# Test 1: Normal crossing / monomial  L(w) = w^{2k}, RLCT = 1/(2k)
# =============================================================================


class MonomialModel(nn.Module):
    """Single-parameter model whose output is w^k * x.

    With MSE loss against y=0 targets this gives L(w) = (w^k)^2 = w^{2k},
    which has RLCT = 1/(2k) at w=0.
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(t.zeros(1))

    def forward(self, x):
        return self.w**self.k * x


def monomial_forward_loss(model, params, buffers, x, y):
    pred = functional_call(model, (params, buffers), (x,))
    return (pred**2).mean()


def make_monomial_loader(n_samples: int = 1024, batch_size: int = 1024):
    """Loader of ones — loss reduces to w^{2k}."""
    x = t.ones(n_samples, 1)
    y = t.zeros(n_samples, 1)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_monomial_rlct():
    """Estimate RLCT for w^{2k} and compare against 1/(2k)."""
    loader = make_monomial_loader()
    n = len(loader.dataset)
    nbeta = n / t.tensor(float(n)).log().item()

    results = {}
    for k in [1, 2, 3]:
        model = MonomialModel(k=k)
        # w is initialized to 0, which is the singular point

        est = LLCEstimator(
            draws=3000,
            chains=10,
            burnin_steps=500,
            steps_bw_draws=1,
            learning_rate=1e-4,
            localization=2.0,
            nbeta=nbeta,
        )
        llc = est.estimate_llc(model, loader, monomial_forward_loss, method="SGLD", seed=42)
        mean_llc = llc.mean().item()
        true_rlct = 1.0 / (2 * k)
        results[k] = (mean_llc, true_rlct)
        print(f"  k={k}:  estimated={mean_llc:.4f}  true={true_rlct:.4f}  ratio={mean_llc/true_rlct:.2f}")

    print()
    for k, (est_val, true_val) in results.items():
        ratio = est_val / true_val
        assert 0.3 < ratio < 3.0, (
            f"k={k}: estimated RLCT {est_val:.4f} too far from true {true_val:.4f} (ratio={ratio:.2f})"
        )
    print("  PASSED: monomial RLCT estimates within 3x of true values")

    # Check ordering: higher k should give lower RLCT
    ests = [results[k][0] for k in [1, 2, 3]]
    assert ests[0] > ests[1] > ests[2], (
        f"RLCT ordering violated: k=1:{ests[0]:.4f}, k=2:{ests[1]:.4f}, k=3:{ests[2]:.4f}"
    )
    print("  PASSED: RLCT ordering k=1 > k=2 > k=3")


def evaluate_monomial_mse(model, data):
    x, y = data
    pred = model(x)
    return (pred**2).mean()


def test_monomial_rlct_devinterp():
    """Estimate monomial RLCT with devinterp for comparison."""
    loader = make_monomial_loader()
    n = len(loader.dataset)
    nbeta = n / t.tensor(float(n)).log().item()

    results = {}
    for k in [1, 2, 3]:
        model = MonomialModel(k=k)

        di_results = estimate_learning_coeff_with_summary(
            model,
            loader,
            evaluate=evaluate_monomial_mse,
            sampling_method=SGLD,
            optimizer_kwargs=dict(lr=1e-4, localization=2.0, nbeta=nbeta),
            num_draws=3000,
            num_chains=10,
            num_burnin_steps=500,
            num_steps_bw_draws=1,
            device="cpu",
            seed=42,
            verbose=False,
        )
        mean_llc = di_results["llc/mean"]
        true_rlct = 1.0 / (2 * k)
        results[k] = (mean_llc, true_rlct)
        print(f"  k={k}:  estimated={mean_llc:.4f}  true={true_rlct:.4f}  ratio={mean_llc/true_rlct:.2f}")

    print()
    for k, (est_val, true_val) in results.items():
        ratio = est_val / true_val
        assert 0.3 < ratio < 3.0, (
            f"k={k}: devinterp RLCT {est_val:.4f} too far from true {true_val:.4f} (ratio={ratio:.2f})"
        )
    print("  PASSED: devinterp monomial RLCT estimates within 3x of true values")

    ests = [results[k][0] for k in [1, 2, 3]]
    assert ests[0] > ests[1] > ests[2], (
        f"RLCT ordering violated: k=1:{ests[0]:.4f}, k=2:{ests[1]:.4f}, k=3:{ests[2]:.4f}"
    )
    print("  PASSED: devinterp RLCT ordering k=1 > k=2 > k=3")
    return results


# =============================================================================
# Test 2: Reduced rank regression  RLCT = r(H + K - r) / 2
# =============================================================================


class FactoredLinearMap(nn.Module):
    """Factored linear map W = A @ B, from R^K -> R^H.

    The factorization W = A B with A in R^{H x r_max}, B in R^{r_max x K}
    creates a singular parameterization: many (A, B) pairs give the same W.
    At a true parameter of rank r < r_max, the RLCT is r(H + K - r) / 2,
    which is strictly less than d/2 = r_max(H + K) / 2.
    """

    def __init__(self, input_dim: int, output_dim: int, max_rank: int):
        super().__init__()
        self.A = nn.Parameter(t.zeros(output_dim, max_rank))
        self.B = nn.Parameter(t.zeros(max_rank, input_dim))

    def forward(self, x):
        # x: (batch, K) -> (batch, H)
        W = self.A @ self.B
        return x @ W.T


def rrr_forward_loss(model, params, buffers, x, y):
    pred = functional_call(model, (params, buffers), (x,))
    return ((pred - y) ** 2).mean()


def make_rrr_data(
    H: int, K: int, true_rank: int, n_samples: int = 2048, seed: int = 0
):
    """Generate noiseless teacher data from a rank-r linear map."""
    rng = t.Generator().manual_seed(seed)
    A = t.randn(H, true_rank, generator=rng) * 0.5
    B = t.randn(true_rank, K, generator=rng) * 0.5
    W_true = A @ B  # H x K, rank r

    x = t.randn(n_samples, K, generator=rng)
    y = x @ W_true.T  # noiseless

    return x, y, W_true, A, B


def test_reduced_rank_regression():
    """Estimate RLCT for reduced rank regression, compare to r(H+K-r)/2.

    Uses a factored parameterization W = A @ B where A is H x r_max and
    B is r_max x K. The student has max_rank = r_max > true_rank, creating
    a genuinely singular point: many (A, B) pairs map to the same rank-r W.
    """
    H, K = 5, 5
    r_max = 5  # student capacity — full rank of the space
    n_samples = 4096
    batch_size = 1024

    results = {}
    for true_rank in [1, 2, 3]:
        x_data, y_data, W_true, A_true, B_true = make_rrr_data(
            H, K, true_rank, n_samples=n_samples
        )
        loader = DataLoader(
            TensorDataset(x_data, y_data), batch_size=batch_size, shuffle=False
        )
        nbeta = n_samples / t.tensor(float(n_samples)).log().item()

        # Initialize student at a valid factorization of W_true.
        # Pad A_true (H x r) and B_true (r x K) to (H x r_max) and (r_max x K)
        # with zeros so A @ B = W_true exactly.
        model = FactoredLinearMap(K, H, r_max)
        with t.no_grad():
            model.A.zero_()
            model.B.zero_()
            model.A[:, :true_rank] = A_true
            model.B[:true_rank, :] = B_true

        est = LLCEstimator(
            draws=3000,
            chains=10,
            burnin_steps=500,
            steps_bw_draws=1,
            learning_rate=1e-5,
            localization=10.0,
            nbeta=nbeta,
        )
        llc = est.estimate_llc(model, loader, rrr_forward_loss, method="SGLD", seed=42)
        mean_llc = llc.mean().item()
        true_rlct = true_rank * (H + K - true_rank) / 2.0
        results[true_rank] = (mean_llc, true_rlct)
        print(f"  rank={true_rank}:  estimated={mean_llc:.4f}  true={true_rlct:.4f}  ratio={mean_llc/true_rlct:.2f}")

    print()
    for r, (est_val, true_val) in results.items():
        ratio = est_val / true_val
        assert 0.3 < ratio < 3.0, (
            f"rank={r}: estimated RLCT {est_val:.4f} too far from true {true_val:.4f} (ratio={ratio:.2f})"
        )
    print("  PASSED: reduced rank RLCT estimates within 3x of true values")

    # Check ordering: higher true rank -> higher RLCT (more effective params)
    ests = [results[r][0] for r in [1, 2, 3]]
    assert ests[0] < ests[1] < ests[2], (
        f"RLCT ordering violated: r=1:{ests[0]:.4f}, r=2:{ests[1]:.4f}, r=3:{ests[2]:.4f}"
    )
    print("  PASSED: RLCT ordering rank=1 < rank=2 < rank=3")


def evaluate_rrr_mse(model, data):
    x, y = data
    pred = model(x)
    return ((pred - y) ** 2).mean()


def test_reduced_rank_regression_devinterp():
    """Estimate RRR RLCT with devinterp for comparison."""
    H, K = 5, 5
    r_max = 5
    n_samples = 4096
    batch_size = 1024

    results = {}
    for true_rank in [1, 2, 3]:
        x_data, y_data, W_true, A_true, B_true = make_rrr_data(
            H, K, true_rank, n_samples=n_samples
        )
        loader = DataLoader(
            TensorDataset(x_data, y_data), batch_size=batch_size, shuffle=False
        )
        nbeta = n_samples / t.tensor(float(n_samples)).log().item()

        model = FactoredLinearMap(K, H, r_max)
        with t.no_grad():
            model.A.zero_()
            model.B.zero_()
            model.A[:, :true_rank] = A_true
            model.B[:true_rank, :] = B_true

        di_results = estimate_learning_coeff_with_summary(
            model,
            loader,
            evaluate=evaluate_rrr_mse,
            sampling_method=SGLD,
            optimizer_kwargs=dict(lr=1e-5, localization=10.0, nbeta=nbeta),
            num_draws=3000,
            num_chains=10,
            num_burnin_steps=500,
            num_steps_bw_draws=1,
            device="cpu",
            seed=42,
            verbose=False,
        )
        mean_llc = di_results["llc/mean"]
        true_rlct = true_rank * (H + K - true_rank) / 2.0
        results[true_rank] = (mean_llc, true_rlct)
        print(f"  rank={true_rank}:  estimated={mean_llc:.4f}  true={true_rlct:.4f}  ratio={mean_llc/true_rlct:.2f}")

    print()
    for r, (est_val, true_val) in results.items():
        ratio = est_val / true_val
        assert 0.3 < ratio < 3.0, (
            f"rank={r}: devinterp RLCT {est_val:.4f} too far from true {true_val:.4f} (ratio={ratio:.2f})"
        )
    print("  PASSED: devinterp RRR RLCT estimates within 3x of true values")

    ests = [results[r][0] for r in [1, 2, 3]]
    assert ests[0] < ests[1] < ests[2], (
        f"RLCT ordering violated: r=1:{ests[0]:.4f}, r=2:{ests[1]:.4f}, r=3:{ests[2]:.4f}"
    )
    print("  PASSED: devinterp RLCT ordering rank=1 < rank=2 < rank=3")
    return results


def test_compare_accuracy():
    """Run both implementations on both problems and compare accuracy to ground truth."""
    print("  Running ours + devinterp on monomial...")
    loader = make_monomial_loader()
    n = len(loader.dataset)
    nbeta = n / t.tensor(float(n)).log().item()

    print("\n  --- Monomial ---")
    print(f"  {'k':<4} {'true':>8} {'ours':>8} {'devinterp':>10} {'ours_err%':>10} {'di_err%':>10} {'winner':>10}")
    for k in [1, 2, 3]:
        true_rlct = 1.0 / (2 * k)
        model = MonomialModel(k=k)

        # Ours
        est = LLCEstimator(
            draws=3000, chains=10, burnin_steps=500, steps_bw_draws=1,
            learning_rate=1e-4, localization=2.0, nbeta=nbeta,
        )
        our_llc = est.estimate_llc(model, loader, monomial_forward_loss, method="SGLD", seed=42).mean().item()

        # devinterp
        model_di = MonomialModel(k=k)  # fresh copy at w=0
        di_results = estimate_learning_coeff_with_summary(
            model_di, loader, evaluate=evaluate_monomial_mse, sampling_method=SGLD,
            optimizer_kwargs=dict(lr=1e-4, localization=2.0, nbeta=nbeta),
            num_draws=3000, num_chains=10, num_burnin_steps=500, num_steps_bw_draws=1,
            device="cpu", seed=42, verbose=False,
        )
        di_llc = di_results["llc/mean"]

        our_err = abs(our_llc - true_rlct) / true_rlct * 100
        di_err = abs(di_llc - true_rlct) / true_rlct * 100
        winner = "ours" if our_err < di_err else "devinterp" if di_err < our_err else "tie"
        print(f"  {k:<4} {true_rlct:>8.4f} {our_llc:>8.4f} {di_llc:>10.4f} {our_err:>9.1f}% {di_err:>9.1f}% {winner:>10}")

    print("\n  --- Reduced Rank Regression ---")
    H, K, r_max = 5, 5, 5
    n_samples, batch_size = 4096, 1024
    print(f"  {'rank':<4} {'true':>8} {'ours':>8} {'devinterp':>10} {'ours_err%':>10} {'di_err%':>10} {'winner':>10}")
    for true_rank in [1, 2, 3]:
        true_rlct = true_rank * (H + K - true_rank) / 2.0
        x_data, y_data, W_true, A_true, B_true = make_rrr_data(H, K, true_rank, n_samples=n_samples)
        rrr_loader = DataLoader(TensorDataset(x_data, y_data), batch_size=batch_size, shuffle=False)
        rrr_nbeta = n_samples / t.tensor(float(n_samples)).log().item()

        # Ours
        model = FactoredLinearMap(K, H, r_max)
        with t.no_grad():
            model.A.zero_(); model.B.zero_()
            model.A[:, :true_rank] = A_true; model.B[:true_rank, :] = B_true
        est = LLCEstimator(
            draws=3000, chains=10, burnin_steps=500, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=rrr_nbeta,
        )
        our_llc = est.estimate_llc(model, rrr_loader, rrr_forward_loss, method="SGLD", seed=42).mean().item()

        # devinterp
        model_di = FactoredLinearMap(K, H, r_max)
        with t.no_grad():
            model_di.A.zero_(); model_di.B.zero_()
            model_di.A[:, :true_rank] = A_true; model_di.B[:true_rank, :] = B_true
        di_results = estimate_learning_coeff_with_summary(
            model_di, rrr_loader, evaluate=evaluate_rrr_mse, sampling_method=SGLD,
            optimizer_kwargs=dict(lr=1e-5, localization=10.0, nbeta=rrr_nbeta),
            num_draws=3000, num_chains=10, num_burnin_steps=500, num_steps_bw_draws=1,
            device="cpu", seed=42, verbose=False,
        )
        di_llc = di_results["llc/mean"]

        our_err = abs(our_llc - true_rlct) / true_rlct * 100
        di_err = abs(di_llc - true_rlct) / true_rlct * 100
        winner = "ours" if our_err < di_err else "devinterp" if di_err < our_err else "tie"
        print(f"  {true_rank:<4} {true_rlct:>8.4f} {our_llc:>8.4f} {di_llc:>10.4f} {our_err:>9.1f}% {di_err:>9.1f}% {winner:>10}")

    print("\n  DONE: see table above for accuracy comparison")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=== Test: Monomial RLCT — ours ===")
    test_monomial_rlct()

    print("\n=== Test: Monomial RLCT — devinterp ===")
    test_monomial_rlct_devinterp()

    print("\n=== Test: Reduced Rank Regression — ours ===")
    test_reduced_rank_regression()

    print("\n=== Test: Reduced Rank Regression — devinterp ===")
    test_reduced_rank_regression_devinterp()

    print("\n=== Accuracy comparison (ours vs devinterp vs ground truth) ===")
    test_compare_accuracy()

    print("\n=== ALL KNOWN-RLCT TESTS PASSED ===")


if __name__ == "__main__":
    main()
