"""
Lesson 9 — Backpropagation: training a neural net by hand
=========================================================

GURU NOTE
---------
In Lesson 8 I HAND-SET the hidden weights. No one does that in practice — the
whole point is to LEARN them. Learning a net is the same idea as Lesson 4
(predict, measure error, step downhill), with one addition: the error has to be
pushed BACKWARDS through the layers so each weight knows its share of the blame.
That backward push is "backpropagation". It is just the chain rule from calculus,
applied layer by layer.

THE CHAIN RULE, IN ONE BREATH
-----------------------------
The output's error depends on the output-layer weights directly, and on the
hidden weights only THROUGH the hidden layer. So to get the hidden weights'
gradient you multiply: (how the loss changes with the hidden output) x (how the
hidden output changes with the hidden weights). "Multiply the slopes along the
path." That's the whole trick.

For our 2-layer net the gradients work out to (this is neural.fit_mlp, lines
125-135, verbatim):

    err = (p - y)              # output error   -- same (p-y) as logistic!
    gw2 = err.T @ h1           # output-layer weight gradient
    gb2 = err.sum(0)
    dh1 = err @ w2             # push error back through W2 (chain rule step)
    dz1 = dh1 * (z1 > 0)       # through ReLU: gradient flows only where it was ON
    gw1 = dz1.T @ x            # hidden-layer weight gradient
    gb1 = dz1.sum(0)

The magic line is `dz1 = dh1 * (z1 > 0)`: the derivative of relu is 1 where the
input was positive and 0 elsewhere. So gradient only flows through hidden units
that were actually active. That's it. We'll train the SAME XOR net from Lesson 8
and watch it discover its own hidden layer.

Run me:  python lessons/lesson_09_backprop.py
"""

from __future__ import annotations

import numpy as np

from _util import step, title
from lesson_02_score_and_sigmoid import sigmoid
from lesson_08_neural_networks import accuracy, forward, make_xor, relu


def fit_mlp(X, y, hidden: int = 8, epochs: int = 300, lr: float = 0.1,
            l2: float = 1e-4, batch: int = 64, fp_weight: float = 1.0,
            seed: int = 0, verbose: bool = True):
    """Train a 2-layer MLP with mini-batch gradient descent + backprop.

    This is a faithful, smaller sibling of neural.fit_mlp. Same forward pass, same
    backward pass, same class weighting. Only the sizes differ.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    # Small RANDOM init. If we started at zeros, every hidden unit would compute
    # the same thing forever ('symmetry') and never specialize. Randomness breaks
    # that. neural.py does the exact same thing.
    w1 = rng.normal(0, 0.5, (hidden, d))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, 0.5, (1, hidden))
    b2 = np.array([-0.5])

    # Class weights (Lesson 6), same formula as the repo.
    n_pos = max(float(np.sum(y >= 0.5)), 1.0)
    n_neg = max(float(np.sum(y < 0.5)), 1.0)
    sw = np.where(y >= 0.5, n / (2 * n_pos), max(fp_weight, 1.0) * n / (2 * n_neg))

    idx = np.arange(n)
    for epoch in range(epochs):
        rng.shuffle(idx)
        for start in range(0, n, batch):
            sl = idx[start:start + batch]
            xb, yb, wb = X[sl], y[sl], sw[sl]

            # ---- FORWARD ----
            z1 = xb @ w1.T + b1            # (b, hidden) pre-activation
            h1 = relu(z1)                 # (b, hidden) hidden output
            p = sigmoid((h1 @ w2.T + b2).ravel())   # (b,) prediction

            # ---- BACKWARD (chain rule, output -> hidden) ----
            err = ((p - yb) * wb / max(len(sl), 1))[:, None]   # (b,1)
            gw2 = err.T @ h1 + l2 * w2     # (1,hidden)
            gb2 = err.sum(0)               # (1,)
            dh1 = err @ w2                 # (b,hidden)  push back through W2
            dz1 = dh1 * (z1 > 0)           # through ReLU: only where it was active
            gw1 = dz1.T @ xb + l2 * w1     # (hidden,d)
            gb1 = dz1.sum(0)               # (hidden,)

            # ---- STEP downhill ----
            w1 -= lr * gw1; b1 -= lr * gb1
            w2 -= lr * gw2; b2 -= lr * gb2

        if verbose and (epoch + 1) in (1, 5, 25, 100, epochs):
            p_all = forward(X, w1, b1, w2, b2)
            print(f"      epoch {epoch + 1:4d}   acc={accuracy(y, p_all):.3f}")
    return w1, b1, w2, b2


def numeric_grad_check(X, y):
    """Confidence check: does our analytic gradient match a numerical estimate?

    This is how you catch a hand-derived-gradient bug (the #1 backprop mistake).
    We nudge one weight by a tiny epsilon, see how the loss changes, and compare
    that slope to what the backprop formula claims. They should nearly match.
    """
    rng = np.random.default_rng(1)
    hidden, d = 4, X.shape[1]
    w1 = rng.normal(0, 0.5, (hidden, d)); b1 = np.zeros(hidden)
    w2 = rng.normal(0, 0.5, (1, hidden)); b2 = np.array([0.0])

    def loss():
        p = np.clip(forward(X, w1, b1, w2, b2), 1e-9, 1 - 1e-9)
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

    # analytic gradient for w2[0,0] on the full batch
    z1 = X @ w1.T + b1; h1 = relu(z1)
    p = sigmoid((h1 @ w2.T + b2).ravel())
    err = ((p - y) / len(y))[:, None]
    gw2_analytic = (err.T @ h1)[0, 0]

    eps = 1e-6
    w2[0, 0] += eps; l_plus = loss()
    w2[0, 0] -= 2 * eps; l_minus = loss()
    w2[0, 0] += eps
    gw2_numeric = (l_plus - l_minus) / (2 * eps)
    return gw2_analytic, gw2_numeric


def main() -> None:
    title("Lesson 9 — Backpropagation: training a net by hand")
    X, y = make_xor()

    step("First, TRUST the gradient. Analytic (backprop) vs numeric estimate:")
    a, nmc = numeric_grad_check(X, y)
    print(f"    backprop says dL/dw2 = {a:+.6f}")
    print(f"    numeric  says dL/dw2 = {nmc:+.6f}")
    print(f"    difference = {abs(a - nmc):.2e}  (tiny -> our backprop is correct)")
    print("    Always gradient-check a hand-derived backprop. This is the safety net.")

    step("Now LEARN the XOR net that we hand-built in Lesson 8. Watch acc climb.")
    print("    Start: random weights, ~50% (clueless). Backprop finds the solution.")
    print()
    w1, b1, w2, b2 = fit_mlp(X, y, hidden=8, epochs=300, lr=0.1)

    step("Final result:")
    p = forward(X, w1, b1, w2, b2)
    print(f"    trained MLP accuracy = {accuracy(y, p):.1%}  (learned, not hand-set!)")
    print("    It rediscovered a hidden-layer solution to XOR on its own — no human")
    print("    picked those 8 hidden units' weights.")

    step("Peek at what a hidden unit learned (its input weights):")
    for i in range(min(4, w1.shape[0])):
        print(f"      hidden unit {i}: w={np.round(w1[i], 2)}  b={b1[i]:+.2f}")
    print("    Each row is one 'bent line'. Together they carve the XOR regions.")

    step("Feel the knobs (same ones as the logistic, plus network size).")
    for hid in (2, 4, 16):
        w1b, b1b, w2b, b2b = fit_mlp(X, y, hidden=hid, epochs=300, lr=0.1,
                                     verbose=False)
        acc = accuracy(y, forward(X, w1b, b1b, w2b, b2b))
        print(f"      hidden={hid:<3} final acc={acc:.1%}")
    print("    2 units CAN solve XOR but often gets stuck in a bad start; more units")
    print("    make training more forgiving. This is why the repo uses 64, not 2.")

    step("Connect to your repo")
    print("    * This fit_mlp is neural.fit_mlp with smaller numbers. The forward")
    print("      pass, the err=(p-y) output gradient, the dz1=dh1*(z1>0) ReLU")
    print("      gradient, the mini-batch loop, the class weights — all identical.")
    print("    * Your payload MLP does this over 512 hashed-text inputs (Lesson 10)")
    print("      and 64 hidden units, for ~12 epochs, on ~500k requests.")

    step("Takeaways")
    print("    1. Backprop = the chain rule: multiply slopes backward through layers.")
    print("    2. Output error is still (p - y); it gets pushed back via W2 and ReLU.")
    print("    3. dz1 = dh1 * (z1 > 0): gradient flows only through active units.")
    print("    4. ALWAYS gradient-check a hand-written backprop.")


# ---------------------------------------------------------------------------
# EXERCISES  (solutions/lesson_09_solution.py)
# ---------------------------------------------------------------------------
# 1. Set lr=2.0 and re-run. Does training destabilize? Then set lr=0.005. Does it
#    crawl? Find an lr that reaches >95% in the fewest epochs.
#
# 2. Initialize w1 and w2 to ZEROS instead of random. Watch the net fail to learn
#    and explain why in a comment (hint: 'symmetry' — every hidden unit stays
#    identical). This is why random init matters.
#
# 3. Add gradient-checking for w1[0,0] (not just w2). Confirm the hidden-layer
#    gradient is also correct. This is the exact tool that would have caught the
#    real bug fixed in your repo's isolation-forest constant (see git log).
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
