"""Lesson 9 solutions."""
import _path  # noqa: F401
import numpy as np
from lesson_08_neural_networks import accuracy, forward, make_xor, relu
from lesson_09_backprop import fit_mlp


def fit_zero_init(X, y, hidden=8, epochs=200, lr=0.1):
    """Ex2: initialize ALL weights to zero and watch it fail (symmetry)."""
    n, d = X.shape
    w1 = np.zeros((hidden, d)); b1 = np.zeros(hidden)
    w2 = np.zeros((1, hidden)); b2 = np.array([0.0])
    for _ in range(epochs):
        z1 = X @ w1.T + b1; h1 = relu(z1)
        p = 1 / (1 + np.exp(-(h1 @ w2.T + b2).ravel()))
        err = ((p - y) / n)[:, None]
        w2 -= lr * (err.T @ h1); b2 -= lr * err.sum(0)
        dz1 = (err @ w2) * (z1 > 0)
        w1 -= lr * (dz1.T @ X); b1 -= lr * dz1.sum(0)
    return accuracy(y, forward(X, w1, b1, w2, b2))


if __name__ == "__main__":
    X, y = make_xor()

    print("Ex1 — lr tuning: reach >95% in the fewest epochs:")
    for lr in (0.005, 0.1, 0.5, 2.0):
        w1, b1, w2, b2 = fit_mlp(X, y, hidden=8, epochs=200, lr=lr, verbose=False)
        print(f"  lr={lr:<5} final acc={accuracy(y, forward(X, w1, b1, w2, b2)):.1%}")
    print("  Too small crawls; too big is unstable; ~0.1-0.5 is the sweet spot here.")

    print("\nEx2 — zero init FAILS (all hidden units stay identical = 'symmetry'):")
    print(f"  zero-init accuracy = {fit_zero_init(X, y):.1%}  (stuck near chance)")
    print("  With identical weights every unit gets the same gradient forever, so they")
    print("  never specialize. Random init breaks the tie — that's why neural.py uses it.")

    print("\nEx3 — gradient-check the HIDDEN layer weight w1[0,0] too:")
    rng = np.random.default_rng(0)
    hidden, d = 4, X.shape[1]
    w1 = rng.normal(0, .5, (hidden, d)); b1 = np.zeros(hidden)
    w2 = rng.normal(0, .5, (1, hidden)); b2 = np.array([0.0])

    def loss():
        p = np.clip(forward(X, w1, b1, w2, b2), 1e-9, 1 - 1e-9)
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

    z1 = X @ w1.T + b1; h1 = relu(z1)
    p = 1 / (1 + np.exp(-(h1 @ w2.T + b2).ravel()))
    err = ((p - y) / len(y))[:, None]
    dz1 = (err @ w2) * (z1 > 0)
    analytic = (dz1.T @ X)[0, 0]
    eps = 1e-6
    w1[0, 0] += eps; lp = loss(); w1[0, 0] -= 2 * eps; lm = loss(); w1[0, 0] += eps
    print(f"  analytic dL/dw1[0,0]={analytic:+.6f}  numeric={(lp - lm) / (2 * eps):+.6f}")
    print("  Match -> the backprop-through-ReLU-into-W1 path is correct.")
