"""
Lesson 3 — Loss: measuring wrongness so a machine can fix it
============================================================

GURU NOTE
---------
In Lesson 2 we GUESSED the weights. To let a machine find good weights, it needs a
single number telling it how wrong it currently is — the "loss". Then training
(Lesson 4) is just: change the weights to make that number smaller.

You might think "just use accuracy — % correct". Here is the deep reason that
fails, and why every classifier including yours uses LOG-LOSS instead:

  * Accuracy only cares whether p is above or below 0.5. Predicting 0.51 vs 0.99
    for a true attack counts the same. So accuracy is FLAT — nudging a weight
    usually doesn't change it at all, and when it does it JUMPS. You can't walk
    downhill on a staircase.
  * Log-loss cares HOW confident you were. It rewards confident-and-right and
    savagely punishes confident-and-wrong, smoothly. Smooth = you can always tell
    which way is downhill. That's what makes learning possible.

This is the objective `fit_logistic` minimizes in your repo.

Run me:  python lessons/lesson_03_loss.py
"""

from __future__ import annotations

import numpy as np

from _util import ascii_curve, step, title
from lesson_01_features import build_dataset
from lesson_02_score_and_sigmoid import add_bias, predict_proba


def log_loss(y, p, eps: float = 1e-12):
    """Binary cross-entropy: the average 'surprise' of the true labels.

    For each example:
        if the truth is 1 (attack):  cost = -log(p)      -> 0 if p=1, huge if p=0
        if the truth is 0 (benign):  cost = -log(1 - p)  -> 0 if p=0, huge if p=1

    We clip p away from exactly 0/1 so log() never blows up to infinity.
    Lower loss = better. Perfect confident predictions -> loss 0.
    """
    p = np.clip(p, eps, 1 - eps)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def main() -> None:
    title("Lesson 3 — Loss: measuring wrongness")

    step("The shape of the penalty. Suppose the TRUTH is 1 (a real attack).")
    print("    Cost = -log(p). Watch how the penalty explodes as p -> 0")
    print("    (i.e. as the model confidently says 'benign' about a real attack):")
    ascii_curve(lambda p: -np.log(np.clip(p, 1e-3, 1)), 0.01, 0.99,
                label="cost when truth=1:  x-axis = model's p,  y-axis = penalty")
    for p in (0.99, 0.9, 0.5, 0.1, 0.01):
        print(f"      truth=1, model said p={p:>4}: cost = {-np.log(p):.3f}")
    print("    Being confidently WRONG (p=0.01 on a real attack) costs ~4.6.")
    print("    Being confidently RIGHT (p=0.99) costs ~0.01. That asymmetry is the")
    print("    whole point — it PULLS the model toward confident, correct answers.")

    step("Why not accuracy? Because it's flat and can't guide learning.")
    print("    Imagine slowly increasing one weight. Accuracy stays 80%... 80%...")
    print("    then suddenly jumps to 85% when one point crosses 0.5. No gradient.")
    print("    Log-loss, by contrast, changes smoothly with EVERY weight nudge.")

    step("Compute the loss of our hand-tuned model from Lesson 2.")
    X, y = build_dataset()
    Xb = add_bias(X)
    w_good = np.array([2.0, 4.0, -1.5])
    p_good = predict_proba(Xb, w_good)
    print(f"    log-loss of the hand-tuned weights = {log_loss(y, p_good):.4f}")

    step("Now compare three sets of weights. Lower loss = better model.")
    w_zero = np.array([0.0, 0.0, 0.0])        # predicts 0.5 for everything
    w_bad = np.array([-3.0, -3.0, 2.0])       # deliberately backwards
    for name, w in [("all-zero (p=0.5 always)", w_zero),
                    ("backwards (bad)", w_bad),
                    ("hand-tuned (good)", w_good)]:
        p = predict_proba(Xb, w)
        acc = float(np.mean((p >= 0.5) == (y >= 0.5)))
        print(f"      {name:28} loss={log_loss(y, p):.4f}   acc={acc:.0%}")
    print("    Notice loss RANKS them correctly and finely, while accuracy is coarse.")

    step("The big idea for Lesson 4")
    print("    Training = find the weights with the SMALLEST loss. We won't guess —")
    print("    we'll compute which direction reduces the loss and step that way,")
    print("    over and over. That direction is called the GRADIENT. Next lesson.")

    step("Connect to your repo")
    print("    * fit_logistic in train.py minimizes exactly this log-loss")
    print("      (with two twists we'll add: regularization in L5, class weights in L6).")
    print("    * 'metrics' in your model JSONs report accuracy/precision/recall for")
    print("      HUMANS to read, but the TRAINING objective is log-loss, because it's")
    print("      the one with a usable gradient.")

    step("Takeaways")
    print("    1. Loss = one number for 'how wrong', lower is better.")
    print("    2. Log-loss punishes confident mistakes smoothly -> learnable.")
    print("    3. Accuracy is for reporting, not for training.")


# ---------------------------------------------------------------------------
# EXERCISES  (solutions/lesson_03_solution.py)
# ---------------------------------------------------------------------------
# 1. What is the log-loss of a model that predicts EXACTLY p=0.5 for every
#    example? Derive it by hand (hint: -log(0.5)) and confirm with w_zero above.
#
# 2. Construct weights whose ACCURACY is high (say 90%) but whose LOSS is worse
#    than the hand-tuned model. (Hint: be correct but barely — predictions near
#    0.5 — versus correct AND confident.) This shows why the two disagree.
#
# 3. Add a single catastrophic example: a real attack (label 1) that the
#    hand-tuned model scores p=0.001. Append it and recompute the loss. Watch one
#    confident mistake dominate the average. This is why outliers matter.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
