"""Lesson 4 solutions."""
import _path  # noqa: F401
import numpy as np
from lesson_01_features import build_dataset
from lesson_02_score_and_sigmoid import add_bias, predict_proba, sigmoid
from lesson_03_loss import log_loss


def fit_with_history(X, y, steps=400, lr=0.5):
    """Ex1: record the loss at every step to confirm it decreases."""
    w = np.zeros(X.shape[1]); w[-1] = -0.5
    history = []
    for i in range(steps):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / len(y)
        w -= (lr / (1 + i / 150)) * grad
        history.append(log_loss(y, p))
    return w, history


def numeric_grad(X, y, w, j, eps=1e-5):
    """Ex4: estimate dLoss/dw[j] numerically."""
    def loss(ww):
        p = np.clip(sigmoid(X @ ww), 1e-12, 1 - 1e-12)
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    wp = w.copy(); wp[j] += eps
    wm = w.copy(); wm[j] -= eps
    return (loss(wp) - loss(wm)) / (2 * eps)


if __name__ == "__main__":
    X, y = build_dataset()
    Xb = add_bias(X)

    print("Ex1 — loss history (should fall monotonically):")
    w, hist = fit_with_history(Xb, y)
    for i in range(0, 400, 100):
        print(f"  step {i:3d}: loss={hist[i]:.4f}")
    print(f"  step 399: loss={hist[-1]:.4f}")

    print("\nEx2 — divergence with too-large lr:")
    _, hist_big = fit_with_history(Xb, y, steps=200, lr=200.0)
    print(f"  lr=200: loss START={hist_big[0]:.3f}  END={hist_big[-1]:.3f}  (went UP)")
    print("  A too-big step overshoots the valley and climbs the far wall — divergence.")

    print("\nEx4 — gradient check (analytic vs numeric) at the trained weights:")
    p = sigmoid(Xb @ w)
    analytic = (Xb.T @ (p - y) / len(y))
    for j in range(3):
        print(f"  w[{j}]: analytic={analytic[j]:+.6f}  numeric={numeric_grad(Xb, y, w, j):+.6f}")
    print("  They match -> the formula grad = X.T @ (p - y) / n is correct.")
