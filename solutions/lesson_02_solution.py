"""Lesson 2 solutions."""
import _path  # noqa: F401
import numpy as np
from lesson_01_features import build_dataset, two_features
from lesson_02_score_and_sigmoid import add_bias, predict_proba, sigmoid

if __name__ == "__main__":
    X, y = build_dataset()
    Xb = add_bias(X)

    print("Ex1 — make it STRICTER by raising the bias (less negative baseline).")
    print("  Stricter = flags more, so it misses fewer attacks but false-alarms more.")
    for bias in (-1.5, -0.5, +0.5):
        w = np.array([2.0, 4.0, bias])
        p = predict_proba(Xb, w)
        flagged = int((p >= 0.5).sum())
        print(f"  bias={bias:+.1f}: flags {flagged}/20 requests")
    print("  Raising the bias is the ONE number that shifts strictness. (Raising a")
    print("  feature weight also works, but only for requests that have that feature.)")

    print("\nEx2 — weights that predict 0.5 for EVERYTHING: all zeros.")
    w0 = np.zeros(3)
    print(f"  z is always 0, sigmoid(0)=0.5. mean p = {predict_proba(Xb, w0).mean():.3f}")

    print("\nEx3 — hand-compute z for \"' OR 1=1--\":")
    f = two_features("' OR 1=1--")   # [special, sqli]
    w = np.array([2.0, 4.0, -1.5])
    z = 2.0 * f[0] + 4.0 * f[1] + (-1.5) * 1.0
    print(f"  features={np.round(f,3)}  z = 2*{f[0]:.1f} + 4*{f[1]:.1f} - 1.5 = {z:.3f}")
    print(f"  sigmoid(z) = {sigmoid(z):.4f}  (matches predict_proba)")
