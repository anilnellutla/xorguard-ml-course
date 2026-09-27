"""
Lesson 1 — Features: turning requests into numbers
==================================================

GURU NOTE
---------
A model cannot read `GET /search?q=' OR 1=1--`. It reads *numbers*. So the very
first move in all of machine learning is to convert the messy real thing into a
fixed-length list of numbers — a "feature vector".

Your real repo does exactly this. The engine turns every request into 32 numbers
(open `../xorguard-ml/contract/supervised-v1.json` to see their names: sqli, xss,
rce, ..., entropy, path_len, ..., bias). This lesson builds a tiny 2-feature
version so you can SEE it, then scales the idea up to the real 32.

Two rules you will carry all course:
  * A **feature** is one number describing the input. Many features = a vector.
  * A **label** is the known correct answer. In xorguard: 1 = malicious, 0 = benign.

Run me:  python lessons/lesson_01_features.py
"""

from __future__ import annotations

import numpy as np

from _util import ascii_scatter, step, title


# ---------------------------------------------------------------------------
# A hand-written feature extractor for tiny strings.
# In the REAL repo the Rust engine does this (and produces 32 features). Here we
# make just two features so we can plot them on a screen.
# ---------------------------------------------------------------------------
def two_features(text: str) -> np.ndarray:
    """Return [special_char_ratio, sqli_hits] for a string.

    Feature 1 - special_char_ratio: fraction of chars that are 'punctuation-ish'.
                Attacks are full of  ' " < > ; ( ) = -  ; normal text is not.
    Feature 2 - sqli_hits: how many classic SQLi tokens appear (crude on purpose).
    """
    if not text:
        return np.array([0.0, 0.0])
    specials = sum(c in "'\"<>;()=-/*|&$" for c in text)
    special_ratio = specials / len(text)

    lowered = text.lower()
    sqli_tokens = ["'", " or ", " and ", "union", "select", "--", "1=1", "#"]
    sqli_hits = sum(lowered.count(tok) for tok in sqli_tokens)
    # squash the raw count into roughly 0..1 so both features share a scale
    sqli_hits = min(1.0, sqli_hits / 4.0)
    return np.array([special_ratio, sqli_hits])


BENIGN = [
    "ada@example.com",
    "blue-sneakers",
    "page=2&limit=20",
    "red dress size 10",
    "john",
    "latest",
    "hello world",
    "search=winter+coats",
    "id=42",
    "correct-horse-battery",
]

MALICIOUS = [
    "' OR 1=1--",
    "admin' OR '1'='1",
    "1' UNION SELECT password FROM users--",
    "<script>alert(1)</script>",
    "; cat /etc/passwd",
    "' OR SLEEP(5)--",
    "1;DROP TABLE users",
    "\" OR \"\"=\"",
    "%27%20OR%201=1",
    "$(id)",
]


def build_dataset():
    """Featurize both lists, stack into X (features) and y (labels)."""
    rows = []
    labels = []
    for s in BENIGN:
        rows.append(two_features(s))
        labels.append(0)
    for s in MALICIOUS:
        rows.append(two_features(s))
        labels.append(1)
    X = np.stack(rows)           # shape (n_samples, n_features)
    y = np.array(labels, float)  # shape (n_samples,)
    return X, y


def main() -> None:
    title("Lesson 1 — Features: turning requests into numbers")

    step("First, watch ONE request become numbers.")
    example = "' OR 1=1--"
    f = two_features(example)
    print(f"    text     : {example!r}")
    print(f"    features : [special_ratio={f[0]:.3f}, sqli_hits={f[1]:.3f}]")
    print("    A human sees an attack. The model will only ever see those 2 numbers.")

    step("Now featurize a benign one for contrast.")
    example2 = "red dress size 10"
    f2 = two_features(example2)
    print(f"    text     : {example2!r}")
    print(f"    features : [special_ratio={f2[0]:.3f}, sqli_hits={f2[1]:.3f}]")

    step("Build the whole tiny dataset: X = features, y = labels (1=mal, 0=benign).")
    X, y = build_dataset()
    print(f"    X shape = {X.shape}  (rows = requests, columns = features)")
    print(f"    y shape = {y.shape}  (one label per row)")
    print(f"    positives (attacks) = {int(y.sum())}, negatives (benign) = {int((y==0).sum())}")

    step("The key insight — plot the two classes in feature space.")
    print("    If the two clouds separate, a model can draw a boundary between them.")
    print("    THIS is what 'learnable' looks like. Keep this picture in your head.")
    ascii_scatter(
        X[:, 0], X[:, 1], y,
        xlabel="special_char_ratio", ylabel="sqli_hits",
    )

    step("Why this matters for your repo:")
    print("    * The engine produces 32 features instead of 2 (we can't plot 32-D,")
    print("      but the model does the same thing there — separate two clouds).")
    print("    * Good features make the clouds separate. Bad features make them")
    print("      overlap, and NO model can save you. Feature design > model choice.")
    print("    * Look at generate.py in xorguard-ml: benign_row() and malicious_row()")
    print("      set these same kinds of numbers by hand to make a fake dataset.")

    step("Takeaways")
    print("    1. Move 1 of ML = turn the thing into a fixed list of numbers.")
    print("    2. X is the feature matrix (n_samples x n_features).")
    print("    3. y is the labels. Here 1=malicious, 0=benign — same as the repo.")
    print("    4. Learning is possible only if the classes separate in feature space.")


# ---------------------------------------------------------------------------
# EXERCISES  (try these, then check solutions/lesson_01_solution.py)
# ---------------------------------------------------------------------------
# 1. Add a THIRD feature to two_features(): `length` = min(1.0, len(text)/40).
#    Does it help separate the classes? Print it for a few strings.
#
# 2. Add a benign string that FOOLS the current features — something that scores
#    high on special_char_ratio but is actually harmless (hint: a math
#    expression like "(3*4)=12"). This is a "hard negative", and Lesson 6 +
#    hardneg.py in the repo are all about these.
#
# 3. Write featurize_like_engine(text) that returns a length-3 vector
#    [special_ratio, sqli_hits, xss_hits] where xss_hits counts "<script",
#    "onerror", "onload", "<svg". Featurize "<svg onload=alert(1)>" and confirm
#    the xss feature lights up but sqli stays ~0.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
