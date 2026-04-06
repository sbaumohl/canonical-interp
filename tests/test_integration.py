"""Integration tests for the full estimate_llc pipeline.

All tests use synthetic data and run on CPU to be fast and CI-friendly.
"""

import pytest
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from canonical_interp.slt import LLCEstimator


class _LinearModel(nn.Module):
    def __init__(self, in_dim=4, out_dim=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# Output shape and type
# ---------------------------------------------------------------------------


class TestOutputShape:
    @pytest.mark.parametrize("chains", [1, 2, 5])
    def test_output_shape_matches_chains(self, trained_linear_model, synthetic_loader, chains):
        est = LLCEstimator(
            draws=5, chains=chains, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=1.0,
        )
        llc = est.estimate_llc(trained_linear_model, synthetic_loader,
                               seed=0, compile=False, show_progress=False)
        assert llc.shape == (chains,)
        assert llc.dtype == t.float32

    def test_array_log_l_shape(self, trained_linear_model, synthetic_loader):
        draws, chains = 10, 3
        est = LLCEstimator(
            draws=draws, chains=chains, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=1.0,
        )
        est.estimate_llc(trained_linear_model, synthetic_loader,
                         seed=0, compile=False, show_progress=False)
        assert est.array_log_l.shape == (chains, draws)

    def test_original_loss_shape(self, trained_linear_model, synthetic_loader):
        chains = 4
        est = LLCEstimator(
            draws=5, chains=chains, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=1.0,
        )
        est.estimate_llc(trained_linear_model, synthetic_loader,
                         seed=0, compile=False, show_progress=False)
        assert est.original_loss.shape == (chains,)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_result(self, trained_linear_model, synthetic_loader):
        kwargs = dict(
            draws=10, chains=2, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=1.0,
        )
        est = LLCEstimator(**kwargs)
        llc1 = est.estimate_llc(trained_linear_model, synthetic_loader,
                                seed=42, compile=False, show_progress=False)
        llc2 = est.estimate_llc(trained_linear_model, synthetic_loader,
                                seed=42, compile=False, show_progress=False)
        t.testing.assert_close(llc1, llc2)

    def test_different_seeds_different_result(self, trained_linear_model, synthetic_loader):
        kwargs = dict(
            draws=10, chains=2, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=1.0,
        )
        est = LLCEstimator(**kwargs)
        llc1 = est.estimate_llc(trained_linear_model, synthetic_loader,
                                seed=42, compile=False, show_progress=False)
        llc2 = est.estimate_llc(trained_linear_model, synthetic_loader,
                                seed=99, compile=False, show_progress=False)
        assert not t.allclose(llc1, llc2)


# ---------------------------------------------------------------------------
# LLC correctness properties
# ---------------------------------------------------------------------------


class TestLLCProperties:
    def test_llc_positive_for_trained_model(self, trained_linear_model, synthetic_loader):
        est = LLCEstimator(
            draws=30, chains=2, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=100.0, nbeta=4.0,
        )
        llc = est.estimate_llc(trained_linear_model, synthetic_loader,
                               seed=42, compile=False, show_progress=False)
        assert (llc > 0).all(), f"LLC should be positive, got {llc}"

    def test_loss_trace_is_finite(self, trained_linear_model, synthetic_loader):
        est = LLCEstimator(
            draws=20, chains=2, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=100.0, nbeta=4.0,
        )
        llc = est.estimate_llc(trained_linear_model, synthetic_loader,
                               seed=42, compile=False, show_progress=False)
        assert t.isfinite(est.array_log_l).all(), "Loss trace has non-finite values"
        assert t.isfinite(llc).all(), "LLC has non-finite values"

    def test_original_loss_is_consistent_across_chains(self, trained_linear_model, synthetic_loader):
        """All chains start from the same weights, so original_loss should be identical."""
        est = LLCEstimator(
            draws=5, chains=4, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=1.0,
        )
        est.estimate_llc(trained_linear_model, synthetic_loader,
                         seed=42, compile=False, show_progress=False)
        # All chains should have the same original loss
        t.testing.assert_close(
            est.original_loss, est.original_loss[0].expand_as(est.original_loss),
        )

    def test_higher_localization_lower_llc(self, trained_linear_model, synthetic_loader):
        """Stronger localization keeps chains closer to init -> lower LLC."""
        llcs = {}
        for loc in [1.0, 1000.0]:
            est = LLCEstimator(
                draws=30, chains=2, burnin_steps=0, steps_bw_draws=1,
                learning_rate=1e-5, localization=loc, nbeta=4.0,
            )
            llc = est.estimate_llc(trained_linear_model, synthetic_loader,
                                   seed=42, compile=False, show_progress=False)
            llcs[loc] = llc.mean().item()
        assert llcs[1000.0] < llcs[1.0], (
            f"Higher localization should give lower LLC: "
            f"loc=1 -> {llcs[1.0]:.4f}, loc=1000 -> {llcs[1000.0]:.4f}"
        )

    def test_more_draws_reduces_variance(self, trained_linear_model, synthetic_loader):
        """More draws should yield more stable per-chain estimates."""
        variances = {}
        for draws in [5, 50]:
            est = LLCEstimator(
                draws=draws, chains=4, burnin_steps=0, steps_bw_draws=1,
                learning_rate=1e-5, localization=100.0, nbeta=4.0,
            )
            llc = est.estimate_llc(trained_linear_model, synthetic_loader,
                                   seed=42, compile=False, show_progress=False)
            variances[draws] = llc.var().item()
        assert variances[50] < variances[5], (
            f"More draws should reduce variance: "
            f"5 draws var={variances[5]:.6f}, 50 draws var={variances[50]:.6f}"
        )


# ---------------------------------------------------------------------------
# Burnin
# ---------------------------------------------------------------------------


class TestBurnin:
    def test_burnin_steps_executed(self, trained_linear_model, synthetic_loader):
        """With burnin, the sampled loss should differ from zero-burnin."""
        kwargs = dict(
            draws=10, chains=2, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=4.0,
        )
        est0 = LLCEstimator(burnin_steps=0, **kwargs)
        llc0 = est0.estimate_llc(trained_linear_model, synthetic_loader,
                                 seed=42, compile=False, show_progress=False)

        est10 = LLCEstimator(burnin_steps=10, **kwargs)
        llc10 = est10.estimate_llc(trained_linear_model, synthetic_loader,
                                   seed=42, compile=False, show_progress=False)

        # They should differ because burnin changes the chain state before drawing
        assert not t.allclose(llc0, llc10, atol=1e-6)


# ---------------------------------------------------------------------------
# Custom criterion and unpack
# ---------------------------------------------------------------------------


class TestCustomCallbacks:
    def test_custom_criterion_fn(self, regression_loader):
        """MSE criterion should work for a regression task."""
        t.manual_seed(42)
        model = _LinearModel(in_dim=4, out_dim=1)
        # Quick train
        opt = t.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(10):
            for x, y in regression_loader:
                opt.zero_grad()
                F.mse_loss(model(x), y).backward()
                opt.step()
        model.eval()

        def mse(pred, target):
            return ((pred - target) ** 2).mean()

        est = LLCEstimator(
            draws=10, chains=2, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-6, localization=100.0, nbeta=4.0,
        )
        llc = est.estimate_llc(model, regression_loader, criterion_fn=mse,
                               seed=42, compile=False, show_progress=False)
        assert t.isfinite(llc).all()
        assert llc.shape == (2,)

    def test_custom_unpack_fn(self, synthetic_loader):
        """Custom unpack function should be called to extract (x, y)."""
        t.manual_seed(42)
        model = _LinearModel()

        # Wrap data so default unpack won't work
        x_all = t.randn(32, 4)
        y_all = t.randint(0, 2, (32,))
        ds = TensorDataset(x_all, y_all, t.zeros(32))  # extra column
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        def unpack(batch):
            return batch[0], batch[1]  # ignore the third element

        est = LLCEstimator(
            draws=5, chains=1, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=1.0,
        )
        llc = est.estimate_llc(model, loader, unpack_fn=unpack,
                               seed=42, compile=False, show_progress=False)
        assert llc.shape == (1,)
        assert t.isfinite(llc).all()


# ---------------------------------------------------------------------------
# Compile flag
# ---------------------------------------------------------------------------


class TestCompileFlag:
    def test_compile_false_runs(self, trained_linear_model, synthetic_loader):
        est = LLCEstimator(
            draws=5, chains=1, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=1.0,
        )
        llc = est.estimate_llc(trained_linear_model, synthetic_loader,
                               seed=42, compile=False, show_progress=False)
        assert t.isfinite(llc).all()

    def test_compile_true_runs(self, trained_linear_model, synthetic_loader):
        est = LLCEstimator(
            draws=5, chains=1, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=1.0,
        )
        llc = est.estimate_llc(trained_linear_model, synthetic_loader,
                               seed=42, compile=True, show_progress=False)
        assert t.isfinite(llc).all()


# ---------------------------------------------------------------------------
# Chain batching (multi-batch on single device)
# ---------------------------------------------------------------------------


class TestChainBatching:
    def test_chain_batch_splits_work(self, trained_linear_model, synthetic_loader):
        """chain_batch < chains should split chains across multiple calls."""
        est = LLCEstimator(
            draws=5, chains=4, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=1.0,
        )
        llc = est.estimate_llc(
            trained_linear_model, synthetic_loader,
            chain_batch=2, devices=["cpu", "cpu"],
            seed=42, compile=False, show_progress=False,
        )
        assert llc.shape == (4,)
        assert t.isfinite(llc).all()

    def test_chain_batch_all_same_as_explicit(self, trained_linear_model, synthetic_loader):
        """chain_batch='all' and chain_batch=chains should give same result."""
        kwargs = dict(
            draws=10, chains=2, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=1.0,
        )
        est1 = LLCEstimator(**kwargs)
        llc1 = est1.estimate_llc(trained_linear_model, synthetic_loader,
                                 seed=42, compile=False, show_progress=False,
                                 chain_batch="all")

        est2 = LLCEstimator(**kwargs)
        llc2 = est2.estimate_llc(trained_linear_model, synthetic_loader,
                                 seed=42, compile=False, show_progress=False,
                                 chain_batch=2)

        t.testing.assert_close(llc1, llc2)


# ---------------------------------------------------------------------------
# Grad accumulation
# ---------------------------------------------------------------------------


class TestGradAccumulation:
    def test_grad_accumulation_runs(self, trained_linear_model, synthetic_loader):
        est = LLCEstimator(
            draws=5, chains=1, burnin_steps=0, steps_bw_draws=1,
            grad_accumulation_steps=2,
            learning_rate=1e-5, localization=10.0, nbeta=1.0,
        )
        llc = est.estimate_llc(trained_linear_model, synthetic_loader,
                               seed=42, compile=False, show_progress=False)
        assert t.isfinite(llc).all()
        assert llc.shape == (1,)


# ---------------------------------------------------------------------------
# Steps between draws (thinning)
# ---------------------------------------------------------------------------


class TestStepsBetweenDraws:
    def test_thinning_changes_result(self, trained_linear_model, synthetic_loader):
        """More steps between draws should change the LLC estimate."""
        llcs = {}
        for sbw in [1, 3]:
            est = LLCEstimator(
                draws=10, chains=2, burnin_steps=0, steps_bw_draws=sbw,
                learning_rate=1e-5, localization=10.0, nbeta=4.0,
            )
            llc = est.estimate_llc(trained_linear_model, synthetic_loader,
                                   seed=42, compile=False, show_progress=False)
            llcs[sbw] = llc
        assert not t.allclose(llcs[1], llcs[3], atol=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_chain(self, trained_linear_model, synthetic_loader):
        est = LLCEstimator(
            draws=5, chains=1, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=1.0,
        )
        llc = est.estimate_llc(trained_linear_model, synthetic_loader,
                               seed=0, compile=False, show_progress=False)
        assert llc.shape == (1,)
        assert t.isfinite(llc).all()

    def test_single_draw(self, trained_linear_model, synthetic_loader):
        est = LLCEstimator(
            draws=1, chains=2, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=1.0,
        )
        llc = est.estimate_llc(trained_linear_model, synthetic_loader,
                               seed=0, compile=False, show_progress=False)
        assert llc.shape == (2,)
        assert t.isfinite(llc).all()

    def test_no_seed_runs(self, trained_linear_model, synthetic_loader):
        """seed=None should still work (non-deterministic)."""
        est = LLCEstimator(
            draws=5, chains=1, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=1.0,
        )
        llc = est.estimate_llc(trained_linear_model, synthetic_loader,
                               seed=None, compile=False, show_progress=False)
        assert t.isfinite(llc).all()

    def test_reuse_estimator(self, trained_linear_model, synthetic_loader):
        """Calling estimate_llc twice on the same estimator should work."""
        est = LLCEstimator(
            draws=5, chains=2, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=10.0, nbeta=1.0,
        )
        llc1 = est.estimate_llc(trained_linear_model, synthetic_loader,
                                seed=42, compile=False, show_progress=False)
        llc2 = est.estimate_llc(trained_linear_model, synthetic_loader,
                                seed=99, compile=False, show_progress=False)
        # Both should be valid
        assert t.isfinite(llc1).all()
        assert t.isfinite(llc2).all()
        # Internal state should reflect the second run
        assert est.array_log_l.shape == (2, 5)


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------


class TestDifferentModels:
    def test_mlp_model(self, trained_mlp, synthetic_loader):
        est = LLCEstimator(
            draws=10, chains=2, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=100.0, nbeta=4.0,
        )
        llc = est.estimate_llc(trained_mlp, synthetic_loader,
                               seed=42, compile=False, show_progress=False)
        assert llc.shape == (2,)
        assert t.isfinite(llc).all()
        assert (llc > 0).all()

    def test_untrained_model(self, synthetic_loader):
        """An untrained model should still produce finite LLC (may not be positive)."""
        model = _LinearModel()
        model.eval()
        est = LLCEstimator(
            draws=10, chains=2, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-6, localization=100.0, nbeta=4.0,
        )
        llc = est.estimate_llc(model, synthetic_loader,
                               seed=42, compile=False, show_progress=False)
        assert t.isfinite(llc).all()


# ---------------------------------------------------------------------------
# Known RLCT (lightweight versions of the gold-standard tests)
# ---------------------------------------------------------------------------


class TestKnownRLCT:
    """Lightweight versions of the closed-form RLCT tests (CPU, fewer samples)."""

    def test_monomial_ordering(self):
        """Higher k in w^{2k} should give lower RLCT (1/(2k))."""
        class MonomialModel(nn.Module):
            def __init__(self, k):
                super().__init__()
                self.k = k
                self.w = nn.Parameter(t.zeros(1))
            def forward(self, x):
                return self.w ** self.k * x

        def monomial_loss(pred, y):
            return (pred ** 2).mean()

        x = t.ones(256, 1)
        y = t.zeros(256, 1)
        loader = DataLoader(TensorDataset(x, y), batch_size=256, shuffle=False)
        n = 256
        nbeta = n / t.tensor(float(n)).log().item()

        rlcts = []
        for k in [1, 2, 3]:
            model = MonomialModel(k)
            est = LLCEstimator(
                draws=500, chains=4, burnin_steps=100, steps_bw_draws=1,
                learning_rate=1e-4, localization=2.0, nbeta=nbeta,
            )
            llc = est.estimate_llc(model, loader, criterion_fn=monomial_loss,
                                   seed=42, compile=False, show_progress=False)
            rlcts.append(llc.mean().item())

        # RLCT should decrease: 1/2 > 1/4 > 1/6
        assert rlcts[0] > rlcts[1] > rlcts[2], (
            f"RLCT ordering violated: k=1:{rlcts[0]:.4f}, k=2:{rlcts[1]:.4f}, k=3:{rlcts[2]:.4f}"
        )

    def test_monomial_k1_close_to_half(self):
        """For k=1, L(w)=w^2, true RLCT = 0.5. Check we're in the ballpark."""
        class MonomialModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(t.zeros(1))
            def forward(self, x):
                return self.w * x

        def monomial_loss(pred, y):
            return (pred ** 2).mean()

        x = t.ones(512, 1)
        y = t.zeros(512, 1)
        loader = DataLoader(TensorDataset(x, y), batch_size=512, shuffle=False)
        n = 512
        nbeta = n / t.tensor(float(n)).log().item()

        model = MonomialModel()
        est = LLCEstimator(
            draws=1000, chains=6, burnin_steps=200, steps_bw_draws=1,
            learning_rate=1e-4, localization=2.0, nbeta=nbeta,
        )
        llc = est.estimate_llc(model, loader, criterion_fn=monomial_loss,
                               seed=42, compile=False, show_progress=False)
        mean_llc = llc.mean().item()
        # Should be within 3x of 0.5
        assert 0.15 < mean_llc < 1.5, f"Expected ~0.5, got {mean_llc:.4f}"
