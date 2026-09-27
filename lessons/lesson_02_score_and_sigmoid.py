"""
Lesson 2 — The score and the sigmoid
=====================================

GURU NOTE
---------
We have features (Lesson 1). Now Move 2: a formula with knobs that turns those
features into a guess. For logistic regression — your `xorguard-supervised-v1`
model — the formula is astonishingly simple, just two steps:

  1. WEIGHTED SUM:   z = w1*x1 + w2*x2 + ... + wD*xD        (a raw "score")
  2. SQUASH:         p = sigmoid(z) = 1 / (1 + e^-z)        (a probability 0..1)

That's the entire model. `predict_proba` in train.py is literally
`sigmoid(x @ weights)` — one line. This lesson makes both steps concrete, and
introduces the "bias trick" that explains why the LAST feature in your repo is
always 1.0.

Run me:  python lessons/lesson_02_score_and_sigmoid.py
"""

from __future__ import annotations

import numpy as np

from _util import ascii_bars, ascii_curve, step, title
from lesson_01_features import build_dataset, two_features


def sigmoid(z):
    """Squash any real number into (0, 1). Same as train.sigmoid in the repo.

    The np.clip guards against overflow: e^1000 is infinity in floating point,
    but sigmoid(1000) is 1.0 for all practical purposes, so clipping at +-30 is a
    safe no-op. This exact trick is in the real code.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def add_bias(X):
    """The BIAS TRICK.

    A model needs a baseline — a constant it can add regardless of the features
    ("how suspicious am I before seeing any evidence?"). Instead of tracking that
    constant separately, we glue a column of 1.0s onto X. Then its weight behaves
    like any other weight, and z = w . x already includes the baseline.

    This is why your repo's 32nd feature is 'bias' and is always 1.0.
    """
    ones = np.ones((X.shape[0], 1))
    return np.concatenate([X, ones], axis=1)


def predict_proba(X, w):
    """Inference in one line. This is what the Rust engine runs per request."""
    return sigmoid(X @ w)


def main() -> None:
    title("Lesson 2 — The score and the sigmoid")

    step("Meet the sigmoid. It turns a raw score into a probability.")
    print("    Very negative score -> ~0 (surely benign)")
    print("    Score of 0          -> exactly 0.5 (no idea)")
    print("    Very positive score -> ~1 (surely malicious)")
    ascii_curve(sigmoid, -8, 8, label="sigmoid(z):  z on x-axis, probability on y-axis")
    for z in (-6, -2, 0, 2, 6):
        print(f"      sigmoid({z:+d}) = {sigmoid(z):.4f}")

    step("The weighted sum z = w . x. Each weight says how much a feature matters.")
    print("    Positive weight  -> this feature is evidence of an ATTACK.")
    print("    Negative weight  -> this feature is evidence of BENIGN traffic.")
    print("    Big magnitude    -> strong evidence.  Near zero -> the model ignores it.")

    step("Load Lesson 1's data and add the bias column.")
    X, y = build_dataset()
    Xb = add_bias(X)   # now 3 columns: [special_ratio, sqli_hits, 1.0]
    print(f"    X was {X.shape}, with bias it is {Xb.shape} (last column all 1.0)")
    print(f"    first row with bias: {Xb[0]}")

    step("Set weights BY HAND (no training yet) using our human knowledge.")
    print("    We believe: sqli_hits matters a lot, special_ratio a bit, and the")
    print("    baseline should be negative (assume benign until proven guilty).")
    #        [special_ratio, sqli_hits, bias]
    w = np.array([2.0,          4.0,      -1.5])
    ascii_bars(["special_ratio", "sqli_hits", "bias"], w)

    step("Run the model. p = sigmoid(z) for every request.")
    p = predict_proba(Xb, w)
    print("    A few predictions (p = model's estimated P(malicious)):")
    strings = [
        "ada@example.com", "red dress size 10", "id=42",       # benign
        "' OR 1=1--", "<script>alert(1)</script>", "$(id)",     # malicious
    ]
    for s in strings:
        f = add_bias(two_features(s)[None, :])
        pi = float(predict_proba(f, w)[0])
        verdict = "MALICIOUS" if pi >= 0.5 else "benign"
        print(f"      p={pi:.3f}  [{verdict:9}]  {s!r}")

    step("How good is this hand-tuned guess? (threshold at 0.5)")
    pred = p >= 0.5
    acc = float(np.mean(pred == (y >= 0.5)))
    print(f"    accuracy = {acc:.1%} on our 20 examples — not bad for GUESSED weights!")
    print("    But we guessed. Next lessons: measure wrongness (L3) and let the")
    print("    machine FIND the best weights automatically (L4).")

    step("Connect to your repo")
    print("    * predict_proba() here == predict_proba() in train.py == the engine's")
    print("      per-request math: sigmoid(w . x).")
    print("    * The 'bias' feature is the constant-1.0 column; its weight is the")
    print("      baseline. In models/supervised.json the bias weight is about -2.46,")
    print("      i.e. 'assume benign' — sigmoid(-2.46) is only ~0.08.")
    print("    * In that same file 'jndi' has weight 3.8: Log4Shell indicators are")
    print("      the strongest attack evidence the model has. You can READ the model.")

    step("Takeaways")
    print("    1. A logistic model = weighted sum, then sigmoid. Two steps.")
    print("    2. Weights are the knobs; their sign and size are human-readable.")
    print("    3. The bias trick = a constant-1.0 feature so the baseline is 'just")
    print("       another weight'. That's your 32nd feature.")


# ---------------------------------------------------------------------------
# EXERCISES  (solutions/lesson_02_solution.py)
# ---------------------------------------------------------------------------
# 1. Make the model STRICTER (fewer attacks slip through, but more false alarms)
#    by changing ONE number in w. Which one, and which direction? Verify by
#    re-running and watching accuracy + which strings flip.
#
# 2. What weights make the model predict 0.5 for EVERYTHING regardless of input?
#    (Hint: think about what makes z always 0.)
#
# 3. Compute z by hand (no sigmoid) for "' OR 1=1--" with the weights above.
#    Then confirm sigmoid(z) matches what predict_proba prints.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
