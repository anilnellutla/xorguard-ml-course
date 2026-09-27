"""Lesson 3 solutions."""
import _path  # noqa: F401
import numpy as np
from lesson_01_features import build_dataset
from lesson_02_score_and_sigmoid import add_bias, predict_proba
from lesson_03_loss import log_loss

if __name__ == "__main__":
    X, y = build_dataset()
    Xb = add_bias(X)

    print("Ex1 — loss when the model predicts p=0.5 for everything:")
    p_half = np.full_like(y, 0.5)
    print(f"  log_loss = {log_loss(y, p_half):.4f}   (== -log(0.5) = {-np.log(0.5):.4f})")

    print("\nEx2 — high ACCURACY but bad LOSS: be correct, but barely (near 0.5).")
    # weak weights: right SIGN, tiny magnitude -> predictions hug 0.5
    w_weak = np.array([0.2, 0.4, -0.15])
    w_conf = np.array([2.0, 4.0, -1.5])
    for name, w in [("barely-correct", w_weak), ("confident-correct", w_conf)]:
        p = predict_proba(Xb, w)
        acc = float(np.mean((p >= 0.5) == (y >= 0.5)))
        print(f"  {name:18} acc={acc:.0%}  loss={log_loss(y, p):.4f}")
    print("  Same-ish accuracy, very different loss. Loss rewards CONFIDENT-correct.")

    print("\nEx3 — one catastrophic row dominates the average loss:")
    p = predict_proba(Xb, w_conf)
    base = log_loss(y, p)
    y2 = np.append(y, 1.0)                 # a real attack...
    p2 = np.append(p, 0.001)              # ...the model is sure is benign
    print(f"  loss before = {base:.4f}   after one confident miss = {log_loss(y2, p2):.4f}")
    print("  A single confident mistake can swamp hundreds of good predictions.")
