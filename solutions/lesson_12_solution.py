"""Lesson 12 solutions."""
import _path  # noqa: F401
from urllib.parse import quote

import numpy as np
from lesson_08_neural_networks import forward
from lesson_09_backprop import fit_mlp
from lesson_10_text_hashing_ngrams import featurize
from lesson_11_capstone_mini_waf import generate_corpus


def url_encode_mut(s: str) -> str:
    """Ex1: a new mutation — percent-encode the payload."""
    return quote(s, safe="=&")


if __name__ == "__main__":
    items = generate_corpus()
    texts = [t for t, _, _ in items]
    y = np.array([l for _, l, _ in items], float)

    print("Ex1 — does URL-encoding evade a model trained on decoded text?")
    X = np.stack([featurize(t) for t in texts])
    w1, b1, w2, b2 = fit_mlp(X, y, hidden=16, epochs=120, lr=0.2, verbose=False)

    def score(t):
        return float(forward(featurize(t)[None, :], w1, b1, w2, b2)[0])

    for atk in ["q=' OR 1=1--", "q=<script>alert(1)</script>"]:
        print(f"  {atk:32} original={score(atk):.2f}  url-encoded={score(url_encode_mut(atk)):.2f}")
    print("  Encoding hides the tell-tale n-grams -> evasion. The real fix is to")
    print("  DECODE before featurizing (the engine does), so encoding can't help.")

    print("\nEx2 — more duplication widens the leakage gap. Repeats 15 vs 60:")
    for reps in (15, 60):
        rng = np.random.default_rng(0)
        bases = [(f"q=session_{''.join(rng.choice(list('0123456789abcdef'),8))}", i % 2)
                 for i in range(80)]
        rows = [(f"{t}&n={k}", lab, bid)
                for bid, (t, lab) in enumerate(bases) for k in range(reps)]
        Xb = np.stack([featurize(t) for t, _, _ in rows])
        yb = np.array([l for _, l, _ in rows], float)
        bid = np.array([b for _, _, b in rows])
        n = len(rows); r = np.random.default_rng(3); perm = r.permutation(n); cut = int(n * .8)

        def nn(tr, va):
            c = 0
            for i in va:
                c += int(yb[tr[int(np.argmax(Xb[tr] @ Xb[i]))]] == yb[i])
            return c / len(va)
        row = nn(perm[:cut], perm[cut:])
        ub = np.unique(bid); r.shuffle(ub); trb = set(ub[:int(len(ub) * .8)].tolist())
        gtr = np.array([i for i in range(n) if bid[i] in trb])
        gva = np.array([i for i in range(n) if bid[i] not in trb])
        grp = nn(gtr, gva)
        print(f"  reps={reps:>2}: row-split={row:.1%}  group-split={grp:.1%}  gap={row-grp:+.1%}")
    print("  More copies of each base = more chances a val row's twin sits in train,")
    print("  so the leaky row-split looks even better while the truth is unchanged.")
