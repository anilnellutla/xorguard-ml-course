"""
Lesson 6 — Class imbalance and cost-sensitive training
======================================================

GURU NOTE
---------
Real traffic is ~99% benign. Train naively on that and the model discovers a lazy
shortcut: "always say benign" scores 99% accuracy while catching ZERO attacks.
Accuracy is a liar on imbalanced data. This lesson shows the trap and the two-line
fix your repo uses.

TWO IDEAS, both living in one line of train.py (line 41):

  1. CLASS BALANCING — up-weight the rare class so its examples count as much as
     the common one. Now the model can't ignore attacks.

  2. FALSE-POSITIVE COST (`fp_weight`) — for a firewall, blocking a real customer
     is worse than logging one attack. So we make benign mistakes EXTRA expensive
     (fp_weight=6 in the repo: a false alarm costs 6x a miss). This trades a little
     recall for a lot of precision — exactly what a WAF wants.

The one line in the repo:
    sample_w = np.where(y >= 0.5, n/(2*n_pos), fp * n/(2*n_neg))
We build it and feel each half.

Run me:  python lessons/lesson_06_class_imbalance.py
"""

from __future__ import annotations

import numpy as np

from _util import step, title
from lesson_02_score_and_sigmoid import sigmoid


def make_imbalanced(seed: int = 0):
    """1900 benign, 100 attacks. Features [sqli, special, bias]."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(1900):
        rows.append([rng.uniform(0, 0.35), rng.uniform(0, 0.4), 1.0])
        labels.append(0)
    for _ in range(100):
        # attacks overlap benign a bit, so the model must WANT to catch them
        rows.append([rng.uniform(0.3, 1.0), rng.uniform(0.3, 1.0), 1.0])
        labels.append(1)
    X = np.array(rows)
    y = np.array(labels, float)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def fit(X, y, fp_weight: float = 1.0, balance: bool = True,
        l2: float = 0.05, steps: int = 800, lr: float = 0.4):
    """Logistic regression with the repo's per-sample class weights.

    balance=False -> every example counts equally (the naive trap).
    balance=True  -> rare class up-weighted; fp_weight makes benign mistakes cost
                     more. This is train.fit_logistic's sample_w, exactly.
    """
    n, d = X.shape
    w = np.zeros(d)
    w[-1] = -0.5
    n_pos = max(float(np.sum(y >= 0.5)), 1.0)
    n_neg = max(float(np.sum(y < 0.5)), 1.0)
    if balance:
        sample_w = np.where(y >= 0.5,
                            n / (2.0 * n_pos),
                            max(fp_weight, 1.0) * n / (2.0 * n_neg))
    else:
        sample_w = np.ones(n)
    for step_i in range(steps):
        p = sigmoid(X @ w)
        err = (p - y) * sample_w          # <-- the only change from Lesson 4
        grad = X.T @ err / n
        step_lr = lr / (1.0 + step_i / 250.0)
        w -= step_lr * (grad + l2 * w)
    return w


def report(name, X, y, w):
    p = sigmoid(X @ w)
    pred = p >= 0.5
    truth = y >= 0.5
    tp = int(np.sum(pred & truth)); fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth)); tn = int(np.sum(~pred & ~truth))
    acc = (tp + tn) / len(y)
    recall = tp / (tp + fn) if (tp + fn) else 0.0     # of attacks, % caught
    prec = tp / (tp + fp) if (tp + fp) else 0.0       # of alarms, % real
    tnr = tn / (tn + fp) if (tn + fp) else 0.0        # of benign, % left alone
    print(f"    {name}")
    print(f"      accuracy={acc:.1%}  recall(caught attacks)={recall:.1%}  "
          f"precision={prec:.1%}  TNR(benign spared)={tnr:.1%}")
    print(f"      confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    return recall, prec, tnr


def main() -> None:
    title("Lesson 6 — Class imbalance and cost-sensitive training")
    X, y = make_imbalanced()
    print(f"    dataset: {int((y==0).sum())} benign vs {int(y.sum())} attacks "
          f"({y.mean():.0%} positive). This is a MILD version of real traffic.")

    step("The trap: train with every example weighted equally.")
    w_naive = fit(X, y, balance=False)
    report("naive (no balancing)", X, y, w_naive)
    print("    Look at accuracy vs recall. High accuracy can hide a model that")
    print("    misses most attacks, because benign examples dominate the loss.")

    step("Fix #1: balance the classes so attacks count as much as benign.")
    w_bal = fit(X, y, balance=True, fp_weight=1.0)
    report("balanced (fp_weight=1)", X, y, w_bal)
    print("    Recall jumps: the rare class now pulls the weights toward catching it.")

    step("Fix #2: make FALSE ALARMS expensive (fp_weight), like a real WAF.")
    print("    A firewall must not block real customers. Raising fp_weight trades")
    print("    a little recall for higher precision and TNR (fewer false blocks).")
    for fp_w in (1.0, 3.0, 6.0, 12.0):
        w = fit(X, y, balance=True, fp_weight=fp_w)
        r, pr, tnr = report(f"fp_weight={fp_w}", X, y, w)
    print("    Watch TNR/precision climb and recall ease off as fp_weight rises.")
    print("    The repo ships fp_weight=6.0: 'a false prevent costs 6x a miss'.")

    step("Why this is the RIGHT trade for your product specifically")
    print("    * A missed attack -> the engine's OTHER layers (rules, critical-family")
    print("      floor) can still catch it, and Detect still logs it.")
    print("    * A false block -> a paying customer gets a 403 on a legit request.")
    print("      That's the expensive error, so we bias the model against it.")

    step("Connect to your repo")
    print("    * train.py line 41 IS the sample_w you just wrote. Same formula.")
    print("    * neural.py fit_mlp uses the identical trick (line 113) for the MLP.")
    print("    * --fp-weight is a CLI flag (default 6.0) in both train and train-neural.")

    step("Takeaways")
    print("    1. On imbalanced data, accuracy lies. Track recall/precision/TNR.")
    print("    2. Balance classes so the rare one isn't ignored.")
    print("    3. fp_weight encodes 'which mistake is worse'. For a WAF, false")
    print("       blocks are worse, so benign errors cost more.")


# ---------------------------------------------------------------------------
# EXERCISES  (solutions/lesson_06_solution.py)
# ---------------------------------------------------------------------------
# 1. Make the data 99% benign (9900 vs 100). Re-run the naive model. What is its
#    accuracy, and its recall? Explain in one sentence why accuracy is useless
#    here.
#
# 2. Find the fp_weight (to the nearest integer) where TNR first reaches 99% on
#    this data. What did recall drop to in exchange? This is the precision/recall
#    trade-off you'll formalize with thresholds in Lesson 7.
#
# 3. Set fp_weight very high (say 100). What happens to recall, and why would a
#    security team NOT want to ship that model even though it never false-alarms?
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
