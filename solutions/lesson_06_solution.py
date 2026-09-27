"""Lesson 6 solutions."""
import _path  # noqa: F401
import numpy as np
from lesson_06_class_imbalance import fit, report
from lesson_02_score_and_sigmoid import sigmoid


def make_severe(seed=0):
    """Ex1: 99% benign (9900 benign vs 100 attacks)."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(9900):
        rows.append([rng.uniform(0, .35), rng.uniform(0, .4), 1.0]); labels.append(0)
    for _ in range(100):
        rows.append([rng.uniform(.3, 1), rng.uniform(.3, 1), 1.0]); labels.append(1)
    X = np.array(rows); y = np.array(labels, float)
    p = rng.permutation(len(y)); return X[p], y[p]


if __name__ == "__main__":
    X, y = make_severe()
    print("Ex1 — 99% benign, NAIVE (no balancing):")
    r, _, _ = report("naive @ 99% benign", X, y, fit(X, y, balance=False))
    print("  Accuracy ~99% while recall is near 0: 'always benign' wins on accuracy")
    print("  alone. Accuracy is useless when one class dominates this hard.")

    print("\nEx2 — smallest integer fp_weight that reaches TNR>=99%:")
    from lesson_06_class_imbalance import make_imbalanced
    Xi, yi = make_imbalanced()
    for fpw in range(1, 13):
        w = fit(Xi, yi, balance=True, fp_weight=float(fpw))
        p = sigmoid(Xi @ w); pred = p >= 0.5; truth = yi >= 0.5
        tn = np.sum(~pred & ~truth); fp = np.sum(pred & ~truth)
        tp = np.sum(pred & truth); fn = np.sum(~pred & truth)
        tnr = tn / (tn + fp); recall = tp / (tp + fn)
        if tnr >= 0.99:
            print(f"  fp_weight={fpw}: TNR={tnr:.1%}, and recall fell to {recall:.1%}")
            break

    print("\nEx3 — fp_weight=100 collapses recall (never blocks, catches almost nothing):")
    w = fit(Xi, yi, balance=True, fp_weight=100.0)
    report("fp_weight=100", Xi, yi, w)
    print("  A WAF that never false-alarms but misses every attack is worthless —")
    print("  the point is BALANCE, tuned to the real cost of each mistake.")
