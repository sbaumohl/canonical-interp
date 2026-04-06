"""Unit tests for LLCEstimator internals."""

import pytest
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging

from canonical_interp.slt import LLCEstimator


class _LinearModel(nn.Module):
    def __init__(self, in_dim=4, out_dim=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_values(self):
        est = LLCEstimator(draws=10, chains=2, burnin_steps=5, steps_bw_draws=1)
        assert est.draws == 10
        assert est.chains == 2
        assert est.burnin_steps == 5
        assert est.steps_bw_draws == 1
        assert est.grad_accumulation_steps == 1
        assert est.verbose is False
        assert est.learning_rate == 1.0
        assert est.localization == 1.0
        assert est.nbeta == 4.0
        assert est.preferred_dtype == t.float32

    def test_custom_values(self):
        est = LLCEstimator(
            draws=100, chains=8, burnin_steps=50, steps_bw_draws=3,
            grad_accumulation_steps=4, verbose=True, learning_rate=0.01,
            localization=5.0, nbeta=10.0, dtype=t.bfloat16,
        )
        assert est.draws == 100
        assert est.chains == 8
        assert est.burnin_steps == 50
        assert est.steps_bw_draws == 3
        assert est.grad_accumulation_steps == 4
        assert est.verbose is True
        assert est.learning_rate == 0.01
        assert est.localization == 5.0
        assert est.nbeta == 10.0
        assert est.preferred_dtype == t.bfloat16


# ---------------------------------------------------------------------------
# _normalize_devices
# ---------------------------------------------------------------------------


class TestNormalizeDevices:
    def test_string_input(self):
        result = LLCEstimator._normalize_devices("cpu")
        assert result == [t.device("cpu")]

    def test_device_input(self):
        result = LLCEstimator._normalize_devices(t.device("cpu"))
        assert result == [t.device("cpu")]

    def test_list_of_strings(self):
        result = LLCEstimator._normalize_devices(["cpu", "cpu"])
        assert result == [t.device("cpu"), t.device("cpu")]

    def test_list_of_devices(self):
        devs = [t.device("cpu"), t.device("cpu")]
        result = LLCEstimator._normalize_devices(devs)
        assert result == devs

    def test_mixed_list(self):
        result = LLCEstimator._normalize_devices(["cpu", t.device("cpu")])
        assert all(isinstance(d, t.device) for d in result)
        assert len(result) == 2

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="unexpected type"):
            LLCEstimator._normalize_devices(42)

    def test_empty_list(self):
        result = LLCEstimator._normalize_devices([])
        assert result == []


# ---------------------------------------------------------------------------
# _accumulate_grad
# ---------------------------------------------------------------------------


class TestAccumulateGrad:
    def test_first_call_clones(self):
        new = {"w": t.tensor([1.0, 2.0])}
        result = LLCEstimator._accumulate_grad(None, new)
        assert "w" in result
        t.testing.assert_close(result["w"], new["w"])
        # Should be a clone, not the same object
        assert result["w"] is not new["w"]

    def test_accumulates_correctly(self):
        g1 = {"w": t.tensor([1.0, 2.0])}
        g2 = {"w": t.tensor([3.0, 4.0])}
        acc = LLCEstimator._accumulate_grad(None, g1)
        acc = LLCEstimator._accumulate_grad(acc, g2)
        t.testing.assert_close(acc["w"], t.tensor([4.0, 6.0]))

    def test_three_way_accumulation(self):
        grads = [
            {"a": t.tensor([1.0]), "b": t.tensor([10.0])},
            {"a": t.tensor([2.0]), "b": t.tensor([20.0])},
            {"a": t.tensor([3.0]), "b": t.tensor([30.0])},
        ]
        acc = None
        for g in grads:
            acc = LLCEstimator._accumulate_grad(acc, g)
        t.testing.assert_close(acc["a"], t.tensor([6.0]))
        t.testing.assert_close(acc["b"], t.tensor([60.0]))

    def test_accumulation_modifies_current_in_place(self):
        current = {"w": t.tensor([1.0])}
        new = {"w": t.tensor([2.0])}
        result = LLCEstimator._accumulate_grad(current, new)
        assert result is current  # same object
        t.testing.assert_close(result["w"], t.tensor([3.0]))


# ---------------------------------------------------------------------------
# _construct_forward_pass
# ---------------------------------------------------------------------------


class TestConstructForwardPass:
    def test_wraps_criterion(self):
        model = _LinearModel(in_dim=2, out_dim=2)
        x = t.randn(4, 2)
        y = t.randint(0, 2, (4,))

        forward_pass = LLCEstimator._construct_forward_pass(F.cross_entropy)
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        loss = forward_pass(model, params, buffers, x, y)

        assert loss.ndim == 0  # scalar
        assert loss.item() > 0  # cross-entropy is positive

    def test_custom_criterion(self):
        model = _LinearModel(in_dim=2, out_dim=1)
        x = t.randn(4, 2)
        y = t.randn(4, 1)

        def mse(pred, target):
            return ((pred - target) ** 2).mean()

        forward_pass = LLCEstimator._construct_forward_pass(mse)
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        loss = forward_pass(model, params, buffers, x, y)

        assert loss.ndim == 0
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# Dataloader warnings
# ---------------------------------------------------------------------------


class TestDataloaderWarnings:
    def _make_loader(self, pin_memory=False, num_workers=0, persistent_workers=False):
        x = t.randn(32, 4)
        y = t.randint(0, 2, (32,))
        kwargs = dict(batch_size=32, shuffle=False, pin_memory=pin_memory,
                      num_workers=num_workers)
        if num_workers > 0:
            kwargs["persistent_workers"] = persistent_workers
        return DataLoader(TensorDataset(x, y), **kwargs)

    def _make_estimator(self):
        return LLCEstimator(
            draws=2, chains=1, burnin_steps=0, steps_bw_draws=1,
            learning_rate=1e-5, localization=1.0, nbeta=1.0,
        )

    def _make_model(self):
        return _LinearModel()

    def test_warns_pin_memory_false(self, caplog):
        est = self._make_estimator()
        loader = self._make_loader(pin_memory=False)
        model = self._make_model()
        with caplog.at_level(logging.WARNING, logger="canonical_interp.slt"):
            est.estimate_llc(model, loader, compile=False, show_progress=False)
        assert any("pin_memory" in r.message for r in caplog.records)

    def test_warns_num_workers_zero(self, caplog):
        est = self._make_estimator()
        loader = self._make_loader(num_workers=0)
        model = self._make_model()
        with caplog.at_level(logging.WARNING, logger="canonical_interp.slt"):
            est.estimate_llc(model, loader, compile=False, show_progress=False)
        assert any("num_workers" in r.message for r in caplog.records)

    def test_warns_persistent_workers_false(self, caplog):
        est = self._make_estimator()
        loader = self._make_loader(num_workers=2, persistent_workers=False)
        model = self._make_model()
        with caplog.at_level(logging.WARNING, logger="canonical_interp.slt"):
            est.estimate_llc(model, loader, compile=False, show_progress=False)
        assert any("persistent_workers" in r.message for r in caplog.records)

    def test_no_pin_memory_warning_when_true(self, caplog):
        """pin_memory=True should not trigger the pin_memory warning."""
        est = self._make_estimator()
        loader = self._make_loader(pin_memory=True, num_workers=0)
        model = self._make_model()
        with caplog.at_level(logging.WARNING, logger="canonical_interp.slt"):
            est.estimate_llc(model, loader, compile=False, show_progress=False)
        assert not any("pin_memory" in r.message for r in caplog.records)
