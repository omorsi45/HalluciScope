"""Multinomial logistic regression fit by Newton's method, in plain numpy.

Classes are always (supported, contradicted, insufficient). Features are standardized inside
`fit_logreg`, a column of ones is appended for the bias, and the parameters are one flat vector
`theta` holding a `(n_classes, d + 1)` matrix in row-major order: element `(c, k)` sits at index
`c * (d + 1) + k`, and the last column of that matrix is the bias. The L2 penalty covers the
weights only. The fit is deterministic: zero start, no randomness.
"""

from dataclasses import dataclass

import numpy as np

RIDGE = 1e-6  # added to the Hessian diagonal so the Newton system is always solvable
ARMIJO_C = 1e-4
MAX_HALVINGS = 30


@dataclass(frozen=True)
class LogRegModel:
    mean: np.ndarray  # (d,) feature means used for standardization
    std: np.ndarray  # (d,) feature standard deviations, 0 replaced by 1
    W: np.ndarray  # (n_classes, d)
    b: np.ndarray  # (n_classes,), mean-centered across classes
    n_iter: int  # Newton steps taken
    converged: bool  # True when max |grad| fell below tol * n


def objective(
    theta: np.ndarray, Xs: np.ndarray, Y: np.ndarray, l2: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Penalized summed NLL with its gradient and full Hessian.

    `Xs` is `(n, d + 1)`: standardized features with a trailing column of ones. `Y` is the
    `(n, n_classes)` one-hot labels. Returns the value `sum_i -log p_i[y_i] + 0.5 * l2 * ||W||^2`
    (bias column unpenalized), the gradient of shape `(n_classes * (d + 1),)`, and the Hessian
    `sum_i kron(diag(p_i) - p_i p_i^T, x_i x_i^T)` plus `l2` on the weight diagonal entries.
    """
    n_classes = Y.shape[1]
    width = Xs.shape[1]
    Theta = theta.reshape(n_classes, width)
    weights = Theta[:, :-1]

    logits = Xs @ Theta.T
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probs = np.exp(log_probs)

    value = -float(np.sum(Y * log_probs)) + 0.5 * l2 * float(np.sum(weights**2))

    grad = (probs - Y).T @ Xs
    grad[:, :-1] += l2 * weights

    # Per-row class covariance diag(p) - p p^T, shape (n, K, K), and per-row x x^T, (n, w, w).
    class_cov = probs[:, :, None] * np.eye(n_classes) - probs[:, :, None] * probs[:, None, :]
    feature_outer = Xs[:, :, None] * Xs[:, None, :]
    hess = np.einsum("nab,nkl->akbl", class_cov, feature_outer, optimize=True)
    hess = hess.reshape(n_classes * width, n_classes * width)
    penalty = np.full((n_classes, width), l2)
    penalty[:, -1] = 0.0
    hess[np.diag_indices_from(hess)] += penalty.ravel()

    return value, grad.ravel(), hess


def fit_logreg(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int = 3,
    l2: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> LogRegModel:
    """Fit by Newton's method with Armijo backtracking.

    Each step solves `(H + RIDGE * I) step = grad` and moves to `theta - t * step`, halving `t`
    from 1 (at most `MAX_HALVINGS` times) until the Armijo condition holds. Stops when
    `max |grad| < tol * n`, after `max_iter` steps, or if no step size gives sufficient decrease.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n, d = X.shape

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    Xs = np.hstack([(X - mean) / std, np.ones((n, 1))])
    Y = np.eye(n_classes)[y]

    theta = np.zeros(n_classes * (d + 1))
    value, grad, hess = objective(theta, Xs, Y, l2)
    n_iter = 0
    converged = False
    while True:
        if np.max(np.abs(grad)) < tol * n:
            converged = True
            break
        if n_iter == max_iter:
            break

        step = np.linalg.solve(hess + RIDGE * np.eye(theta.size), grad)
        slope = float(grad @ step)
        t = 1.0
        for _ in range(MAX_HALVINGS + 1):
            candidate = theta - t * step
            cand_value, cand_grad, cand_hess = objective(candidate, Xs, Y, l2)
            if cand_value <= value - ARMIJO_C * t * slope:
                break
            t *= 0.5
        else:
            break  # no acceptable step: cannot improve further
        theta, value, grad, hess = candidate, cand_value, cand_grad, cand_hess
        n_iter += 1

    Theta = theta.reshape(n_classes, d + 1)
    b = Theta[:, -1]
    return LogRegModel(
        mean=mean,
        std=std,
        W=Theta[:, :-1].copy(),
        b=b - b.mean(),
        n_iter=n_iter,
        converged=converged,
    )


def predict_logits(model: LogRegModel, X: np.ndarray) -> np.ndarray:
    """Class logits `(n, n_classes)` for raw (unstandardized) features."""
    Xs = (np.asarray(X, dtype=float) - model.mean) / model.std
    return Xs @ model.W.T + model.b
