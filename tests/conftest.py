"""Shared fixtures for canonical-interp tests."""

import pytest
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Simple models
# ---------------------------------------------------------------------------


class LinearModel(nn.Module):
    """Minimal linear model for fast testing."""

    def __init__(self, in_dim=4, out_dim=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class TwoLayerMLP(nn.Module):
    """Small MLP for integration tests."""

    def __init__(self, in_dim=4, hidden=8, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class SingleParamModel(nn.Module):
    """Model with a single scalar parameter, useful for testing SGLD dynamics."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(t.zeros(1))

    def forward(self, x):
        return self.w * x


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_loader():
    """Small synthetic classification dataset (deterministic)."""
    rng = t.Generator().manual_seed(0)
    x = t.randn(256, 4, generator=rng)
    y = t.randint(0, 2, (256,), generator=rng)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=64, shuffle=False)


@pytest.fixture
def tiny_loader():
    """Minimal loader for unit tests that need speed over realism."""
    x = t.randn(32, 4)
    y = t.randint(0, 2, (32,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=32, shuffle=False)


@pytest.fixture
def regression_loader():
    """Small synthetic regression dataset."""
    rng = t.Generator().manual_seed(0)
    x = t.randn(128, 4, generator=rng)
    w_true = t.randn(4, 1, generator=rng)
    y = x @ w_true + 0.01 * t.randn(128, 1, generator=rng)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=64, shuffle=False)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_model():
    """Fresh LinearModel on CPU."""
    t.manual_seed(42)
    return LinearModel()


@pytest.fixture
def trained_linear_model(synthetic_loader):
    """LinearModel trained for a few epochs on synthetic data."""
    t.manual_seed(42)
    model = LinearModel()
    opt = t.optim.SGD(model.parameters(), lr=0.1)
    model.train()
    for _ in range(20):
        for x, y in synthetic_loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    model.eval()
    return model


@pytest.fixture
def trained_mlp(synthetic_loader):
    """TwoLayerMLP trained for a few epochs on synthetic data."""
    t.manual_seed(42)
    model = TwoLayerMLP()
    opt = t.optim.SGD(model.parameters(), lr=0.1)
    model.train()
    for _ in range(20):
        for x, y in synthetic_loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def has_cuda():
    """Skip test if CUDA is not available."""
    if not t.cuda.is_available():
        pytest.skip("CUDA not available")
