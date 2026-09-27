"""Lesson 7 solutions."""
import _path  # noqa: F401
import numpy as np
from lesson_02_score_and_sigmoid import sigmoid
from lesson_06_class_imbalance import fit
from lesson_07_evaluation_and_thresholds import (
    at_threshold, calibrate_for_tnr, make_noisy, roc_auc, split,
)


def precision_at_recall(y, p, target_recall):
    """Ex2: highest-precision threshold that still catches >= target_recall."""
    best = (0.0, None)
    for t in np.linspace(0.01, 0.99, 197):
        s = at_threshold(y, p, float(t))
        if s["recall"] >= target_recall and s["precision"] > best[0]:
            best = (s["precision"], float(t))
    return best


if __name__ == "__main__":
    X, y = make_noisy()
    Xtr, ytr, Xva, yva = split(X, y)

    print("Ex1 — does fp_weight change AUC? (AUC is about RANKING, so barely):")
    for fpw in (1.0, 6.0):
        w = fit(Xtr, ytr, balance=True, fp_weight=fpw)
        p = sigmoid(Xva @ w)
        print(f"  fp_weight={fpw}: AUC={roc_auc(yva, p):.4f}")
    print("  fp_weight moves the DECISION boundary (precision/recall), not how well")
    print("  the model ranks attacks above benign. Different jobs, different metrics.")

    print("\nEx2 — precision at a target recall (how clean can we be at 90% recall?):")
    w = fit(Xtr, ytr, balance=True, fp_weight=1.0)
    p = sigmoid(Xva @ w)
    prec, t = precision_at_recall(yva, p, 0.90)
    print(f"  best precision while catching >=90% of attacks: {prec:.1%} at threshold {t:.3f}")

    print("\nEx3 — lower the TNR target -> lower prevent cutoff, higher recall:")
    for target in (0.995, 0.95, 0.90):
        pt, ps, _ = calibrate_for_tnr(yva, p, target_tnr=target)
        print(f"  target TNR={target}: cutoff={pt:.3f}  actual TNR={ps['tnr']:.1%}  recall={ps['recall']:.1%}")
    print("  Relaxing the false-alarm budget lets you block more attacks — but blocks")
    print("  more customers too. That dial is a security/product decision.")
