"""Unit tests for canonical_interp.optim.sgld_step."""

import pytest
import torch as t

from canonical_interp.optim import sgld_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(*shapes, seed=0):
    """Create named param dicts (params, grads, initial_weights)."""
    rng = t.Generator().manual_seed(seed)
    params = {f"p{i}": t.randn(s, generator=rng) for i, s in enumerate(shapes)}
    grads = {k: t.randn_like(v) for k, v in params.items()}
    initial_weights = {k: v.clone() for k, v in params.items()}
    return params, grads, initial_weights


# ---------------------------------------------------------------------------
# Basic update mechanics
# ---------------------------------------------------------------------------


class TestBasicUpdate:
    def test_params_are_modified_in_place(self):
        params, grads, init_w = _make_params((4,), (3, 2))
        originals = {k: v.clone() for k, v in params.items()}
        sgld_step(params, grads, init_w, noise_level=0.0)
        for k in params:
            assert not t.equal(params[k], originals[k]), f"param {k} was not updated"

    def test_pure_gradient_descent_without_noise(self):
        """With noise_level=0, localization=0: w' = w - (lr/2) * nbeta * grad."""
        params = {"w": t.tensor([1.0, 2.0])}
        grads = {"w": t.tensor([0.5, -0.5])}
        init_w = {"w": t.tensor([1.0, 2.0])}
        lr, nbeta = 0.1, 2.0

        expected = t.tensor([1.0, 2.0]) - 0.5 * lr * nbeta * t.tensor([0.5, -0.5])

        sgld_step(params, grads, init_w, learning_rate=lr, nbeta=nbeta,
                  localization=0.0, noise_level=0.0)

        t.testing.assert_close(params["w"], expected, atol=1e-6, rtol=1e-6)

    def test_zero_learning_rate_no_change(self):
        params, grads, init_w = _make_params((5,))
        originals = {k: v.clone() for k, v in params.items()}
        sgld_step(params, grads, init_w, learning_rate=0.0, noise_level=0.0)
        for k in params:
            t.testing.assert_close(params[k], originals[k])

    def test_zero_gradient_only_noise(self):
        """With zero grads and no localization, update is purely noise."""
        t.manual_seed(99)
        params = {"w": t.tensor([0.0, 0.0])}
        grads = {"w": t.tensor([0.0, 0.0])}
        init_w = {"w": t.tensor([0.0, 0.0])}
        sgld_step(params, grads, init_w, learning_rate=1.0, localization=0.0,
                  noise_level=1.0)
        # params should be non-zero (noise was injected)
        assert params["w"].abs().sum() > 0


# ---------------------------------------------------------------------------
# Localization
# ---------------------------------------------------------------------------


class TestLocalization:
    def test_localization_pulls_toward_init(self):
        """Elastic term pushes params back toward initial_weights."""
        params = {"w": t.tensor([5.0])}
        grads = {"w": t.tensor([0.0])}
        init_w = {"w": t.tensor([0.0])}

        sgld_step(params, grads, init_w, learning_rate=0.1, localization=10.0,
                  noise_level=0.0)

        # param should have moved toward 0 (init)
        assert params["w"].item() < 5.0, "localization should pull param toward init"

    def test_localization_zero_has_no_effect(self):
        """localization=0 should give same result regardless of init position."""
        p1 = {"w": t.tensor([3.0])}
        g1 = {"w": t.tensor([1.0])}
        i1 = {"w": t.tensor([0.0])}

        p2 = {"w": t.tensor([3.0])}
        g2 = {"w": t.tensor([1.0])}
        i2 = {"w": t.tensor([100.0])}  # different init — shouldn't matter

        sgld_step(p1, g1, i1, localization=0.0, noise_level=0.0)
        sgld_step(p2, g2, i2, localization=0.0, noise_level=0.0)

        t.testing.assert_close(p1["w"], p2["w"])

    def test_stronger_localization_means_closer_to_init(self):
        """Higher gamma should pull params closer to initial_weights per step."""
        results = {}
        for gamma in [1.0, 10.0]:
            params = {"w": t.tensor([1.0])}
            grads = {"w": t.tensor([0.0])}
            init_w = {"w": t.tensor([0.0])}
            sgld_step(params, grads, init_w, learning_rate=0.01,
                      localization=gamma, noise_level=0.0)
            results[gamma] = params["w"].item()

        # Both should move toward 0; stronger gamma -> closer to 0
        assert 0 < results[10.0] < results[1.0] < 1.0


# ---------------------------------------------------------------------------
# Scaling factors
# ---------------------------------------------------------------------------


class TestScaling:
    def test_nbeta_scales_gradient(self):
        """Doubling nbeta should double the gradient contribution."""
        results = {}
        for nb in [1.0, 2.0]:
            params = {"w": t.tensor([0.0])}
            grads = {"w": t.tensor([1.0])}
            init_w = {"w": t.tensor([0.0])}
            sgld_step(params, grads, init_w, learning_rate=0.1, nbeta=nb,
                      localization=0.0, noise_level=0.0)
            results[nb] = params["w"].item()

        # w' = w - 0.5*lr*nb*grad = -0.5*0.1*nb*1.0
        # ratio of displacements should be 2
        ratio = results[2.0] / results[1.0]
        assert abs(ratio - 2.0) < 1e-5

    def test_grad_accumulation_divides_gradient(self):
        """grad_accumulation_steps should divide the gradient."""
        # Single step with grad=2
        params1 = {"w": t.tensor([0.0])}
        grads1 = {"w": t.tensor([2.0])}
        init_w1 = {"w": t.tensor([0.0])}
        sgld_step(params1, grads1, init_w1, grad_accumulation_steps=1,
                  learning_rate=0.1, nbeta=1.0, localization=0.0, noise_level=0.0)

        # Two accumulated steps with grad=4 (should be equivalent: 4/2 = 2)
        params2 = {"w": t.tensor([0.0])}
        grads2 = {"w": t.tensor([4.0])}
        init_w2 = {"w": t.tensor([0.0])}
        sgld_step(params2, grads2, init_w2, grad_accumulation_steps=2,
                  learning_rate=0.1, nbeta=1.0, localization=0.0, noise_level=0.0)

        t.testing.assert_close(params1["w"], params2["w"], atol=1e-6, rtol=1e-6)

    def test_noise_scales_with_sqrt_lr(self):
        """Noise magnitude should scale with sqrt(learning_rate)."""
        deltas = {}
        for lr in [0.01, 1.0]:
            t.manual_seed(0)
            params = {"w": t.zeros(1000)}
            grads = {"w": t.zeros(1000)}
            init_w = {"w": t.zeros(1000)}
            sgld_step(params, grads, init_w, learning_rate=lr, localization=0.0,
                      noise_level=1.0)
            deltas[lr] = params["w"].std().item()

        # std(delta) = sqrt(lr) * noise_level * std(N(0,1)) ≈ sqrt(lr)
        ratio = deltas[1.0] / deltas[0.01]
        expected_ratio = (1.0 / 0.01) ** 0.5  # = 10
        assert abs(ratio - expected_ratio) / expected_ratio < 0.15


# ---------------------------------------------------------------------------
# Multiple parameters
# ---------------------------------------------------------------------------


class TestMultipleParams:
    def test_all_params_updated(self):
        params, grads, init_w = _make_params((3,), (2, 4), (5,))
        originals = {k: v.clone() for k, v in params.items()}
        sgld_step(params, grads, init_w, noise_level=0.0)
        for k in params:
            assert not t.equal(params[k], originals[k])

    def test_independent_params_get_independent_noise(self):
        """Each parameter tensor should get its own noise draw."""
        t.manual_seed(0)
        params = {"a": t.zeros(100), "b": t.zeros(100)}
        grads = {"a": t.zeros(100), "b": t.zeros(100)}
        init_w = {"a": t.zeros(100), "b": t.zeros(100)}
        sgld_step(params, grads, init_w, learning_rate=1.0, localization=0.0)
        # a and b should have different noise (extremely unlikely to be equal)
        assert not t.allclose(params["a"], params["b"], atol=1e-3)


# ---------------------------------------------------------------------------
# Batched params (vmapped chains dimension)
# ---------------------------------------------------------------------------


class TestBatchedParams:
    def test_batched_update_shape_preserved(self):
        """Params with a leading chain dimension should keep their shape."""
        num_chains = 3
        params = {"w": t.randn(num_chains, 4)}
        grads = {"w": t.randn(num_chains, 4)}
        init_w = {"w": params["w"][0].clone()}  # single initial weights
        sgld_step(params, grads, init_w, noise_level=0.0, localization=0.0)
        assert params["w"].shape == (num_chains, 4)

    def test_batched_chains_get_different_noise(self):
        """Different chains (batch dim) should get different noise."""
        num_chains = 5
        params = {"w": t.zeros(num_chains, 10)}
        grads = {"w": t.zeros(num_chains, 10)}
        init_w = {"w": t.zeros(10)}
        sgld_step(params, grads, init_w, learning_rate=1.0, localization=0.0)
        # Each chain should have different noise
        for i in range(num_chains - 1):
            assert not t.allclose(params["w"][i], params["w"][i + 1], atol=1e-3)
