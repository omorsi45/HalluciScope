"""Multinomial logistic regression (Newton) and temperature scaling.

Expected values come from hand-derived literals, finite differences, or a known data-generating
model, never from the code under test. Classes are always (supported, contradicted, insufficient).
"""

import math

import numpy as np
import pytest

from backend.calibration.logreg import fit_logreg, objective, predict_logits
from backend.calibration.metrics import softmax
from backend.calibration.temperature import fit_temperature

SEED = 20261026
N_CLASSES = 3


def _one_hot(y: np.ndarray) -> np.ndarray:
    return np.eye(N_CLASSES)[y]


def _sample_labels(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Draw one class index per row from the row's probabilities (inverse-CDF sampling)."""
    cumulative = np.cumsum(probs, axis=1)
    u = rng.random(len(probs))[:, None]
    return np.minimum((u > cumulative).sum(axis=1), probs.shape[1] - 1)


def _small_problem():
    """Random 25 x 3 problem: standardized-style features plus the ones column, and a theta."""
    rng = np.random.default_rng(SEED)
    n, d = 25, 3
    Xs = np.hstack([rng.normal(size=(n, d)), np.ones((n, 1))])
    Y = _one_hot(rng.integers(0, N_CLASSES, size=n))
    theta = rng.normal(scale=0.5, size=N_CLASSES * (d + 1))
    return Xs, Y, theta


def test_gradient_matches_finite_differences():
    # Breaks if the gradient or Hessian drops the penalty, penalizes the bias column, or mixes
    # up the (class, feature) layout of the flattened theta.
    Xs, Y, theta = _small_problem()
    l2, h = 0.7, 1e-6
    _, grad, hess = objective(theta, Xs, Y, l2)
    assert grad.shape == theta.shape
    assert hess.shape == (theta.size, theta.size)

    numeric_grad = np.zeros_like(theta)
    for j in range(theta.size):
        step = np.zeros_like(theta)
        step[j] = h
        plus = objective(theta + step, Xs, Y, l2)[0]
        minus = objective(theta - step, Xs, Y, l2)[0]
        numeric_grad[j] = (plus - minus) / (2 * h)
    assert np.max(np.abs(grad - numeric_grad)) < 1e-5

    # Coordinates are (class * 4 + feature): 0 and 8 are weights, 5 is a weight of class 1,
    # 3 and 11 are biases (feature 3 is the ones column).
    for j in (0, 3, 5, 8, 11):
        step = np.zeros_like(theta)
        step[j] = h
        plus = objective(theta + step, Xs, Y, l2)[1]
        minus = objective(theta - step, Xs, Y, l2)[1]
        numeric_column = (plus - minus) / (2 * h)
        assert np.max(np.abs(hess[:, j] - numeric_column)) < 1e-5


def test_objective_value_and_gradient_literal():
    # Features are all zero, so the logits equal the biases. With theta = W of ones and b = 0 the
    # probabilities are 1/3 each: value = 4 ln 3 + 0.5 * 2 * ||W||^2 = 4 ln 3 + 3 (only the
    # three weights are penalized). Labels [0, 0, 1, 2] give class counts [2, 1, 1], so the bias
    # gradient is 4/3 - count and the weight gradient is just l2 * W = 2.
    Xs = np.hstack([np.zeros((4, 1)), np.ones((4, 1))])
    Y = _one_hot(np.array([0, 0, 1, 2]))
    theta = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    value, grad, _ = objective(theta, Xs, Y, 2.0)
    assert value == pytest.approx(4 * math.log(3) + 3.0, abs=1e-12)
    expected = np.array([2.0, 4 / 3 - 2, 2.0, 4 / 3 - 1, 2.0, 4 / 3 - 1])
    assert grad == pytest.approx(expected, abs=1e-12)


def test_recovers_known_logit_differences():
    # Breaks if the Newton step, the standardization round trip, or the bias handling is wrong:
    # the fit would not reproduce the generating model on points it never saw.
    rng = np.random.default_rng(SEED)
    loc, scale = np.array([1.0, -2.0]), np.array([2.0, 0.5])
    W_true = np.array([[0.0, 0.0], [0.5, -1.0], [-0.4, 1.2]])
    # Cancel the mean logit offset W_true @ loc so all three classes are common (about 25%, 45%,
    # 30%). With a rare class the logit differences are poorly determined and the fit is noisy.
    b_true = np.array([0.0, 0.3, -0.2]) - W_true @ loc
    X = rng.normal(loc=loc, scale=scale, size=(20_000, 2))
    y = _sample_labels(rng, softmax(X @ W_true.T + b_true))

    model = fit_logreg(X, y, l2=1e-3)
    assert model.converged

    X_new = rng.normal(loc=loc, scale=scale, size=(1_000, 2))
    true_z = X_new @ W_true.T + b_true
    fitted_z = predict_logits(model, X_new)
    for c in (1, 2):
        true_diff = true_z[:, c] - true_z[:, 0]
        fitted_diff = fitted_z[:, c] - fitted_z[:, 0]
        assert np.mean(np.abs(fitted_diff - true_diff)) < 0.1


def test_separable_data_stays_finite():
    # Without the penalty the weights of separable data diverge; the default l2 keeps them finite.
    rng = np.random.default_rng(SEED)
    centers = np.array([[8.0, 0.0], [-8.0, 0.0], [0.0, 8.0]])
    y = np.tile(np.arange(N_CLASSES), 30)
    X = centers[y] + rng.normal(scale=0.5, size=(90, 2))

    model = fit_logreg(X, y)
    for name in ("mean", "std", "W", "b"):
        assert np.all(np.isfinite(getattr(model, name))), name
    assert model.converged
    assert np.all(np.argmax(predict_logits(model, X), axis=1) == y)


def test_missing_class_ok():
    # Class 1 never occurs: its bias runs toward minus infinity without a floor, but the fit must
    # still return and give that class a tiny probability everywhere.
    rng = np.random.default_rng(SEED)
    y = np.tile(np.array([0, 2]), 100)
    X = rng.normal(size=(200, 2)) + y[:, None]

    model = fit_logreg(X, y)
    probs = softmax(predict_logits(model, X))
    assert np.all(np.isfinite(probs))
    assert np.all(probs[:, 1] < 0.05)


def test_constant_feature_no_nan():
    # A zero-variance column has std 0; dividing by it would produce NaN. It is replaced by 1,
    # and the column then carries no weight, so changing it at predict time changes nothing.
    rng = np.random.default_rng(SEED)
    n = 120
    y = rng.integers(0, N_CLASSES, size=n)
    X = np.column_stack([rng.normal(size=n) + y, np.full(n, 3.0), rng.normal(size=n) - y])

    model = fit_logreg(X, y)
    assert model.std[1] == 1.0
    for name in ("mean", "std", "W", "b"):
        assert np.all(np.isfinite(getattr(model, name))), name
    assert np.all(model.W[:, 1] == 0.0)

    shifted = X.copy()
    shifted[:, 1] = 100.0
    logits = predict_logits(model, X)
    assert np.all(np.isfinite(logits))
    assert predict_logits(model, shifted) == pytest.approx(logits, abs=1e-12)


def test_temperature_recovers_scale():
    # Labels follow softmax(z) but the model reports 2 * z (over-confident by a factor of 2), so
    # the NLL-optimal temperature is 2.
    rng = np.random.default_rng(SEED)
    z = rng.normal(scale=1.5, size=(20_000, N_CLASSES))
    y = _sample_labels(rng, softmax(z))

    temperature = fit_temperature(2.0 * z, y)
    assert 1.9 <= temperature <= 2.1


def test_temperature_bounds():
    # Every logit row ranks its true class first, so the NLL keeps falling as T shrinks and the
    # search ends at the lower end of the log T range, exp(-3).
    rng = np.random.default_rng(SEED)
    y = rng.integers(0, N_CLASSES, size=300)
    logits = _one_hot(y).astype(float)

    temperature = fit_temperature(logits, y)
    assert temperature <= math.exp(-3) + 1e-3
