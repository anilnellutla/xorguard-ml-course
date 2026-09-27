"""
Lesson 7 — Evaluation and thresholds: grading a model honestly
==============================================================

GURU NOTE
---------
A trained model gives a probability. A firewall must give a DECISION. This lesson
covers both halves your repo's eval.py handles:

  A. HONEST GRADING — never grade on data the model trained on (it may have
     memorized it). Hold out a validation set. Then measure:
       precision  = of what we flagged, how much was real?
       recall/TPR = of real attacks, how many did we catch?
       TNR        = of benign traffic, how much did we correctly leave alone?
       AUC        = probability a random attack outranks a random benign request.

  B. CHOOSING THRESHOLDS — the model outputs p in [0,1]; we pick a cutoff. Your
     repo picks TWO: a high `prevent` cutoff (block — demands TNR >= 0.995) and a
     lower `detect` cutoff (just log). We reproduce that calibration here.

Every function here has a twin in eval.py. After this lesson, open eval.py and
you'll recognize all of it.

Run me:  python lessons/lesson_07_evaluation_and_thresholds.py
"""

from __future__ import annotations

import numpy as np

from _util import step, title
from lesson_02_score_and_sigmoid import sigmoid
from lesson_06_class_imbalance import fit


def make_noisy(seed: int = 1):
    """Overlapping traffic: 3 informative features + bias, classes that BLEND.

    Unlike Lesson 6's cleanly separable blobs, here the two clouds overlap, so no
    cutoff is perfect — which is the whole point of learning to grade and to
    choose thresholds. 1500 benign, 500 attacks.
    """
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(1500):
        a = np.clip(rng.normal(0.30, 0.16), 0, 1)
        b = np.clip(rng.normal(0.30, 0.18), 0, 1)
        c = np.clip(rng.normal(0.25, 0.16), 0, 1)
        rows.append([a, b, c, 1.0]); labels.append(0)
    for _ in range(500):
        a = np.clip(rng.normal(0.70, 0.18), 0, 1)
        b = np.clip(rng.normal(0.66, 0.20), 0, 1)
        c = np.clip(rng.normal(0.62, 0.20), 0, 1)
        rows.append([a, b, c, 1.0]); labels.append(1)
    X = np.array(rows)
    y = np.array(labels, float)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def split(X, y, val_frac: float = 0.2, seed: int = 11):
    """Shuffle, then hold out the last val_frac as validation. Same as train.split.

    WARNING you'll unpack in Lesson 12: this splits by ROW. If near-duplicate rows
    exist, copies leak across the split and inflate your scores. Fine here (no
    dupes); a real pitfall in the repo.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * (1.0 - val_frac))
    tr, va = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[va], y[va]


def at_threshold(y, p, t):
    """Confusion-matrix stats at cutoff t. Mirrors eval.at_threshold."""
    pred = p >= t
    truth = y >= 0.5
    tp = float(np.sum(pred & truth)); fp = float(np.sum(pred & ~truth))
    fn = float(np.sum(~pred & truth)); tn = float(np.sum(~pred & ~truth))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {"t": t, "precision": prec, "recall": rec, "tnr": tnr, "fpr": 1 - tnr}


def roc_auc(y, p):
    """AUC = P(a random attack scores above a random benign request).

    This is the Wilcoxon-Mann-Whitney formulation — count winning pairs, ties
    half. Identical idea to eval.roc_auc. 1.0 = perfect ranking, 0.5 = coin flip.
    """
    pos = p[y >= 0.5]
    neg = p[y < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for pi in pos:
        wins += float(np.sum(pi > neg)) + 0.5 * float(np.sum(pi == neg))
    return wins / (len(pos) * len(neg))


def calibrate_for_tnr(y, p, target_tnr: float = 0.995):
    """Lowest threshold whose TNR meets the target. Simplified eval.calibrate_threshold.

    Lower threshold = higher recall, so 'lowest passing' catches the most attacks
    while staying inside the false-alarm budget. If the target is unreachable, we
    relax it step by step (as the repo does).
    """
    grid = np.unique(np.clip(np.concatenate([
        np.linspace(0.5, 0.999, 200),
        np.quantile(p[y < 0.5], np.linspace(0.9, 1.0, 20)) if np.any(y < 0.5) else [0.9],
    ]), 0.4, 0.999))
    for want in [target_tnr, 0.99, 0.98, 0.95]:
        for t in grid:
            s = at_threshold(y, p, float(t))
            if s["tnr"] + 1e-12 >= want:
                return float(t), s, want
    return 0.92, at_threshold(y, p, 0.92), 0.0


def main() -> None:
    title("Lesson 7 — Evaluation and thresholds")
    X, y = make_noisy()

    step("Rule 1: split. Train on 80%, GRADE on the held-out 20%.")
    Xtr, ytr, Xva, yva = split(X, y)
    print(f"    train rows = {len(ytr)},  validation rows = {len(yva)}")
    print("    The model never sees validation rows during training. That's the")
    print("    only way to estimate how it does on traffic it hasn't met.")

    w = fit(Xtr, ytr, balance=True, fp_weight=1.0)
    p_va = sigmoid(Xva @ w)

    step("Rule 2: report the metrics that matter (at the default 0.5 cutoff).")
    s = at_threshold(yva, p_va, 0.5)
    print(f"    precision = {s['precision']:.1%}  (of alarms, how many were real)")
    print(f"    recall    = {s['recall']:.1%}  (of attacks, how many we caught)")
    print(f"    TNR       = {s['tnr']:.1%}  (of benign, how many we spared)")
    print(f"    AUC       = {roc_auc(yva, p_va):.4f}  (ranking quality, cutoff-free)")
    print("    AUC is special: it grades the model's RANKING, independent of where")
    print("    you put the cutoff. 1.0 = every attack scores above every benign req.")

    step("Notice: the model's probabilities are COMPRESSED, not spread over 0..1.")
    print(f"    p ranges from {p_va.min():.2f} to {p_va.max():.2f} — attacks don't reach")
    print("    0.99, benign don't reach 0.01. This is normal, and it's exactly WHY we")
    print("    don't hard-code 0.5: we calibrate the cutoff to wherever the scores land.")

    step("The threshold is a DIAL, not a fixed 0.5. Sweep it and watch the trade.")
    print("      cutoff   recall   TNR      precision")
    for t in (0.40, 0.48, 0.52, 0.56, 0.60, 0.65):
        s = at_threshold(yva, p_va, t)
        print(f"      {t:<7}  {s['recall']:.1%}   {s['tnr']:.1%}   {s['precision']:.1%}")
    print("    Raise the cutoff -> fewer false alarms (higher TNR) but you catch")
    print("    fewer attacks (lower recall). Lower it -> the opposite. YOU choose")
    print("    where to sit based on the cost of each mistake.")

    step("Your repo calibrates TWO cutoffs automatically. Reproduce them.")
    prevent_t, prevent_s, got = calibrate_for_tnr(yva, p_va, target_tnr=0.995)
    print(f"    PREVENT (block): lowest cutoff with TNR>=99.5%  ->  t={prevent_t:.3f}")
    print(f"       at that cutoff: TNR={prevent_s['tnr']:.1%}  recall={prevent_s['recall']:.1%}")
    detect_t, detect_s, _ = calibrate_for_tnr(yva, p_va, target_tnr=0.95)
    detect_t = max(detect_t, 0.55)
    if detect_t >= prevent_t:               # detect must fire BEFORE prevent
        detect_t = min(prevent_t - 0.05, 0.80)
    ds = at_threshold(yva, p_va, detect_t)
    print(f"    DETECT (log):    lower cutoff, TNR>=95%          ->  t={detect_t:.3f}")
    print(f"       at that cutoff: TNR={ds['tnr']:.1%}  recall={ds['recall']:.1%}")
    print("    Two cutoffs = two behaviors from ONE model: log early, block only")
    print("    when very sure. That's exactly train.py's threshold logic.")

    step("Connect to your repo")
    print("    * split, at_threshold, roc_auc, calibrate_threshold here all mirror")
    print("      eval.py. The 'metrics'/'thresholds' blocks in your model JSONs are")
    print("      produced by exactly this calibration on the validation split.")
    print("    * The 0.995 target TNR is why the models rarely false-block: the")
    print("      prevent cutoff is chosen to spare 99.5% of benign traffic.")

    step("Takeaways")
    print("    1. Always grade on held-out data, never the training set.")
    print("    2. precision/recall/TNR/AUC each answer a different question.")
    print("    3. The threshold is a dial trading recall against false alarms.")
    print("    4. Calibrate cutoffs to a TNR budget; use two for detect vs prevent.")


# ---------------------------------------------------------------------------
# EXERCISES  (solutions/lesson_07_solution.py)
# ---------------------------------------------------------------------------
# 1. Retrain with fp_weight=1 and fp_weight=6, and for each print the AUC on the
#    validation set. Does fp_weight change the AUC much? (Surprise: AUC is about
#    RANKING, which the cutoff/cost doesn't change nearly as much as you'd think.)
#
# 2. Write precision_at_recall(y, p, target_recall): find the highest-precision
#    threshold that still catches at least target_recall of attacks. This is how
#    a team asks "how clean can we be while still catching 90% of attacks?"
#
# 3. Lower the target_tnr to 0.90 in calibrate_for_tnr and observe the prevent
#    cutoff drop and recall rise. Explain the safety trade-off in one sentence.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
