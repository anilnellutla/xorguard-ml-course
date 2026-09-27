"""Lesson 11 solutions.

These reuse the capstone's building blocks. The capstone's decide() closure isn't
importable, so here we rebuild the minimum needed for each exercise.
"""
import _path  # noqa: F401
import numpy as np
from lesson_02_score_and_sigmoid import sigmoid
from lesson_06_class_imbalance import fit as fit_logistic
from lesson_08_neural_networks import forward
from lesson_09_backprop import fit_mlp
from lesson_10_text_hashing_ngrams import featurize
from lesson_11_capstone_mini_waf import (
    generate_corpus, indicator_features, split_items,
)


def train_models(tr):
    Xi = np.stack([indicator_features(t) for t, _, _ in tr])
    yi = np.array([l for _, l, _ in tr], float)
    w_log = fit_logistic(Xi, yi, balance=True, fp_weight=2.0)
    H = np.stack([featurize(t) for t, _, _ in tr])
    w1, b1, w2, b2 = fit_mlp(H, yi, hidden=16, epochs=120, lr=0.2, verbose=False)
    return w_log, (w1, b1, w2, b2)


if __name__ == "__main__":
    items = generate_corpus()
    tr, va = split_items(items)
    w_log, (w1, b1, w2, b2) = train_models(tr)

    def scores(text):
        f = indicator_features(text)
        p_log = float(sigmoid(f @ w_log))
        p_mlp = float(forward(featurize(text)[None, :], w1, b1, w2, b2)[0])
        return p_log, p_mlp, 0.5 * p_log + 0.5 * p_mlp

    print("Ex2 — single-model ablation: which model catches what the other misses?")
    fam = {}
    for t, lab, f in va:
        if lab == 1:
            pl, pm, _ = scores(t)
            fam.setdefault(f, [0, 0, 0])
            fam[f][0] += 1
            fam[f][1] += pl >= 0.5
            fam[f][2] += pm >= 0.5
    print(f"  {'family':7} {'n':>3} {'logistic':>9} {'neural':>7}  (attacks caught @0.5)")
    for f, (n, cl, cm) in sorted(fam.items()):
        print(f"  {f:7} {n:>3} {cl:>9} {cm:>7}")
    print("  On THIS toy corpus the neural net (raw text) is the stronger detector —")
    print("  it catches SQLi the indicator-logistic misses. So why keep the logistic?")
    print("  Three real reasons from your repo: it's interpretable (you can read its")
    print("  weights), it feeds the critical-family PREVENT floor, and two independent")
    print("  models fail independently — defense in depth. Coverage isn't the only goal.")

    print("\nEx3 — generalization to an UNSEEN family (traversal was never trained):")
    for t in ["q=../../../../etc/passwd", "file=..%2f..%2fetc%2fshadow"]:
        pl, pm, fu = scores(t)
        print(f"  {t:34} logistic={pl:.2f} neural={pm:.2f} fused={fu:.2f}")
    print("  The text MLP often flags novel payloads via shared n-grams (/etc/, ../),")
    print("  even with no traversal template in training. That's Lesson 10 paying off.")
