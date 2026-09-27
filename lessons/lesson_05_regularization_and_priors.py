"""
Lesson 5 — Regularization, and the "don't erase JNDI" prior trick
=================================================================

GURU NOTE
---------
Two problems appear the moment you train on real, limited data:

  1. OVERFITTING. With few examples, the model can grow huge weights to memorize
     quirks of THIS data that won't generalize. Fix: "regularization" — a penalty
     on big weights that keeps the model humble.

  2. RARE-BUT-CRITICAL SIGNALS GET ERASED. Standard regularization pulls weights
     toward ZERO. But some features (like `jndi` / Log4Shell) are rare in any
     corpus yet must stay damning. Pulling their weight to zero is exactly wrong.

Your repo solves #2 with a beautiful trick (train.py lines 33-34): regularize
toward a hand-written PRIOR (the shipped basic weights) instead of toward zero.
"Trust the expert's JNDI weight unless the data strongly argues otherwise."

This lesson demonstrates both, with a JNDI-like rare feature so you SEE the trick
save it.

Run me:  python lessons/lesson_05_regularization_and_priors.py
"""

from __future__ import annotations

import numpy as np

from _util import ascii_bars, step, title
from lesson_02_score_and_sigmoid import sigmoid
from lesson_03_loss import log_loss

# A synthetic feature-space dataset (like generate.py), 4 features:
#   [sqli, special, jndi, bias]
# 'sqli' and 'special' are COMMON. 'jndi' is RARE (a handful of attacks) but a
# PERFECT tell when present — exactly Log4Shell's statistical profile.
FEATURES = ["sqli", "special", "jndi", "bias"]


def make_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    # 200 benign: low everything
    for _ in range(200):
        rows.append([rng.uniform(0, 0.2), rng.uniform(0, 0.3), 0.0, 1.0])
        labels.append(0)
    # 200 common attacks: high sqli/special, no jndi
    for _ in range(200):
        rows.append([rng.uniform(0.6, 1.0), rng.uniform(0.4, 0.9), 0.0, 1.0])
        labels.append(1)
    # only 6 JNDI attacks: everything else looks almost benign, jndi=1 is the tell
    for _ in range(6):
        rows.append([rng.uniform(0, 0.2), rng.uniform(0, 0.3), 1.0, 1.0])
        labels.append(1)
    X = np.array(rows)
    y = np.array(labels, float)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def fit(X, y, l2: float = 0.0, prior=None, init=None, steps: int = 800, lr: float = 0.4):
    """Logistic regression with L2 regularization toward a PRIOR.

    This is train.fit_logistic's exact structure (minus class weights, Lesson 6):
        pen = l2 * (w - prior)      # pull toward prior, not necessarily zero
        w  -= step_lr * (grad + pen)
    prior=None  -> pull toward zero (standard L2).
    prior=vec   -> pull toward that vector (the repo's warm-start trick).
    """
    n, d = X.shape
    w = np.zeros(d) if init is None else np.array(init, float)
    prior = np.zeros(d) if prior is None else np.array(prior, float)
    for step_i in range(steps):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / n
        pen = l2 * (w - prior)
        step_lr = lr / (1.0 + step_i / 250.0)
        w -= step_lr * (grad + pen)
    return w


def main() -> None:
    title("Lesson 5 — Regularization and the prior trick")
    X, y = make_data()
    print(f"    dataset: {int((y==0).sum())} benign, {int(y.sum())} attacks "
          f"(of which only 6 are JNDI-style rare).")

    step("Part A — Overfitting and how L2 tames it.")
    print("    With NO regularization the model is free to grow large weights.")
    w_none = fit(X, y, l2=0.0)
    w_l2 = fit(X, y, l2=0.5)
    print(f"    ||w|| with l2=0.0 : {np.linalg.norm(w_none):.2f}  (bigger, more eager")
    print(f"    ||w|| with l2=0.5 : {np.linalg.norm(w_l2):.2f}  to memorize)")
    print("    Regularization = deliberately keep weights small so the model")
    print("    generalizes instead of memorizing. (We'll MEASURE generalization")
    print("    properly with a train/val split in Lesson 7.)")

    step("Part B — the problem: standard L2 (toward ZERO) erases the rare JNDI tell.")
    w_zero_prior = fit(X, y, l2=0.5, prior=None)
    jndi_i = FEATURES.index("jndi")
    print(f"    Learned weights, L2 toward zero:")
    ascii_bars(FEATURES, w_zero_prior)
    print(f"    -> jndi weight = {w_zero_prior[jndi_i]:+.3f}. Only 6 examples push it")
    print("       up, but the penalty pulls it toward 0 on every step. It's weak.")

    step("Part C — the fix: regularize toward a PRIOR that already trusts JNDI.")
    # An 'expert' hand-written prior, like the shipped basic weights:
    prior = np.array([1.5, 0.5, 3.0, -1.5])   # note jndi prior = 3.0
    w_prior = fit(X, y, l2=0.5, prior=prior, init=prior)
    print("    Prior (expert knowledge, jndi=3.0):")
    ascii_bars(FEATURES, prior)
    print("    Weights after training with L2 pulling TOWARD that prior:")
    ascii_bars(FEATURES, w_prior)
    print(f"    -> jndi weight = {w_prior[jndi_i]:+.3f}. Preserved! The data never")
    print("       argued against JNDI, so the prior kept it strong.")

    step("Prove it matters: score a fresh JNDI attack with each model.")
    jndi_attack = np.array([0.1, 0.2, 1.0, 1.0])   # looks benign except jndi=1
    p_zero = float(sigmoid(jndi_attack @ w_zero_prior))
    p_prior = float(sigmoid(jndi_attack @ w_prior))
    print(f"    L2-to-zero model : p(malicious) = {p_zero:.3f}")
    print(f"    prior-anchored   : p(malicious) = {p_prior:.3f}   <- catches it")
    print("    Same architecture, same data. The prior is the difference between")
    print("    catching Log4Shell and missing it.")

    step("Connect to your repo")
    print("    * train.py: `prior = w.copy()` then `pen = l2 * (w - prior)`. The")
    print("      warm-start `init` is the vendored basic weights; L2 pulls back to")
    print("      them. The comment literally says 'L2-to-zero would erase JNDI'.")
    print("    * You can verify the payoff: in models/supervised.json the jndi weight")
    print("      stayed 3.8 (data agreed with the prior) while `entropy` moved from")
    print("      +0.35 to -0.50 (data DID argue, so it changed). The prior bends.")

    step("Takeaways")
    print("    1. Regularization = penalty on big weights -> less overfitting.")
    print("    2. Standard L2 pulls toward zero and can erase rare-but-vital signals.")
    print("    3. Pulling toward an expert PRIOR keeps that knowledge unless the data")
    print("       overrules it. That's a warm start + prior-anchored L2.")


# ---------------------------------------------------------------------------
# EXERCISES  (solutions/lesson_05_solution.py)
# ---------------------------------------------------------------------------
# 1. Increase the number of JNDI examples from 6 to 100 and retrain the
#    L2-to-zero model. Does jndi's weight recover? Explain why more data reduces
#    the need for a prior.
#
# 2. Set the prior's jndi entry to a WRONG value (say -3.0) and retrain with lots
#    of JNDI attacks in the data. Does the data eventually overrule the bad prior?
#    This shows the prior is a bias, not a straitjacket.
#
# 3. Sweep l2 in {0.0, 0.1, 1.0, 10.0} with prior=None and print ||w|| for each.
#    Describe the trade-off between fitting the data and staying humble.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
