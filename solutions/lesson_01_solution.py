"""Lesson 1 solutions."""
import _path  # noqa: F401  (puts lessons/ on the path)
import numpy as np
from lesson_01_features import BENIGN, MALICIOUS


# --- Ex 1: add a length feature --------------------------------------------
def three_features(text: str) -> np.ndarray:
    if not text:
        return np.array([0.0, 0.0, 0.0])
    specials = sum(c in "'\"<>;()=-/*|&$" for c in text)
    special_ratio = specials / len(text)
    lowered = text.lower()
    toks = ["'", " or ", " and ", "union", "select", "--", "1=1", "#"]
    sqli_hits = min(1.0, sum(lowered.count(t) for t in toks) / 4.0)
    length = min(1.0, len(text) / 40.0)          # NEW third feature
    return np.array([special_ratio, sqli_hits, length])


# --- Ex 3: an engine-style featurizer with a separate XSS channel ----------
def featurize_like_engine(text: str) -> np.ndarray:
    t = text.lower()
    special_ratio = sum(c in "'\"<>;()=-/*|&$" for c in text) / max(len(text), 1)
    sqli = min(1.0, sum(t.count(k) for k in ["'", " or ", "union", "select", "--"]) / 4)
    xss = min(1.0, sum(t.count(k) for k in ["<script", "onerror", "onload", "<svg"]) / 2)
    return np.array([special_ratio, sqli, xss])


if __name__ == "__main__":
    print("Ex1 — length feature separates long UNION payloads from short benign:")
    for s in ["id=42", "1' UNION SELECT password FROM users--"]:
        print(f"  {s!r:45} -> {np.round(three_features(s), 3)}")

    print("\nEx2 — a HARD NEGATIVE: high special-char ratio but harmless:")
    from lesson_01_features import two_features
    hard = "(3*4)=12&(5-2)=3"
    print(f"  {hard!r} -> {np.round(two_features(hard), 3)}  (looks attacky, isn't)")
    print("  This is exactly what hardneg.py manufactures so the model isn't fooled.")

    print("\nEx3 — engine-style featurizer, XSS channel lights up, SQLi stays low:")
    print(f"  '<svg onload=alert(1)>' -> {np.round(featurize_like_engine('<svg onload=alert(1)>'), 3)}")
    print("  order is [special, sqli, xss] — note xss high, sqli ~0.")
