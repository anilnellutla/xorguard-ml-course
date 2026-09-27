"""
Lesson 12 — Adversarial evasion, and how metrics lie
====================================================

GURU NOTE
---------
You can train a model. Now learn to DISTRUST it the way a security engineer must.
Two hard truths that separate a practitioner from a tutorial-follower:

  A. ATTACKERS ADAPT. A payload can be rewritten to mean the same thing but LOOK
     different, dodging a model that only saw the original form. Your repo has a
     whole module for this (mutate.py: case-swap, /**/ comments, $IFS, ca$1t...)
     mirroring the WAF-A-MoLE evasion research. We'll evade a model, then note the
     fix: train on the mutations too.

  B. YOUR METRICS PROBABLY LIE. If near-duplicate rows land in BOTH the training
     and validation split, the model is graded partly on things it memorized, and
     your AUC looks amazing while real-world performance is worse. This is the #1
     criticism of the xorguard-ml eval today: generate.py/hardneg.py manufacture
     correlated rows, and train.split() shuffles by ROW, so copies leak across the
     split. We'll MEASURE the inflation, then show the fix (split by GROUP).

This lesson is the difference between "my model scores 0.99" and "my model scores
0.99 and I know why that number is trustworthy."

Run me:  python lessons/lesson_12_adversarial_and_pitfalls.py
"""

from __future__ import annotations

import numpy as np

from _util import step, title
from lesson_08_neural_networks import forward
from lesson_09_backprop import fit_mlp
from lesson_10_text_hashing_ngrams import featurize
from lesson_11_capstone_mini_waf import generate_corpus


# --------------------------------------------------------------------------
# Mutation operators — tiny reimplementations of xorguard_ml/mutate.py.
# Each rewrites an attack to MEAN THE SAME but look different.
# --------------------------------------------------------------------------
def case_swap(s: str) -> str:
    return "".join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(s))


def comment_inject(s: str) -> str:
    # SQL treats /**/ as whitespace; the query still runs.
    return "/**/".join(s.split(" "))


def ifs_sub(s: str) -> str:
    # the shell treats $IFS as a field separator == space.
    return s.replace(" ", "${IFS}")


def main() -> None:
    title("Lesson 12 — Adversarial evasion, and how metrics lie")

    # ======================================================================
    # PART A — Evasion
    # ======================================================================
    step("PART A — train a text MLP, then EVADE it with meaning-preserving rewrites.")
    items = generate_corpus()
    texts = [t for t, _, _ in items]
    y = np.array([l for _, l, _ in items], float)
    X = np.stack([featurize(t) for t in texts])
    w1, b1, w2, b2 = fit_mlp(X, y, hidden=16, epochs=120, lr=0.2, verbose=False)

    def score(text: str) -> float:
        return float(forward(featurize(text)[None, :], w1, b1, w2, b2)[0])

    attacks = [
        "q=' OR 1=1--",
        "id=5' UNION SELECT password FROM users--",
        "q=; cat /etc/passwd",
    ]
    mutators = [("original", lambda s: s),
                ("case-swap", case_swap),
                ("/**/comment", comment_inject),
                ("$IFS", ifs_sub)]
    print("    Each row: the model's attack-score for the SAME attack, rewritten.")
    print(f"    {'payload':45}  " + "  ".join(f"{name:>11}" for name, _ in mutators))
    evaded = 0
    total = 0
    for atk in attacks:
        cells = []
        base = score(atk)
        for name, fn in mutators:
            s = score(fn(atk))
            cells.append(f"{s:11.3f}")
            if name != "original":
                total += 1
                if base >= 0.5 and s < 0.5:
                    evaded += 1
        print(f"    {atk[:45]:45}  " + "  ".join(cells))
    print(f"\n    {evaded}/{total} rewrites pushed a caught attack BELOW 0.5 — evaded.")
    print("    The model never saw these forms, so its n-gram vector shifted enough")
    print("    to slip past. An attacker with mutate.py can hunt for exactly these.")

    step("The fix: ADVERSARIAL TRAINING — feed the mutations in as attacks too.")
    aug_texts, aug_y = list(texts), list(y)
    for t, lab, _ in items:
        if lab == 1:
            for _, fn in mutators[1:]:
                aug_texts.append(fn(t)); aug_y.append(1.0)
    Xa = np.stack([featurize(t) for t in aug_texts])
    ya = np.array(aug_y, float)
    w1b, b1b, w2b, b2b = fit_mlp(Xa, ya, hidden=16, epochs=120, lr=0.2, verbose=False)

    def score2(text):
        return float(forward(featurize(text)[None, :], w1b, b1b, w2b, b2b)[0])

    now_caught = 0
    total2 = 0
    for atk in attacks:
        for name, fn in mutators[1:]:
            total2 += 1
            if score2(fn(atk)) >= 0.5:
                now_caught += 1
    print(f"    After adversarial training: {now_caught}/{total2} of those same")
    print("    rewrites are caught again. This is why mutate.py exists in your repo,")
    print("    and why the engine has a CI 'mutation exam' that must stay >=85%.")

    # ======================================================================
    # PART B — Metric inflation from leaked duplicates
    # ======================================================================
    step("PART B — how duplicate rows across a split INFLATE your metrics.")
    print("    Setup: 80 distinct base payloads, each DUPLICATED 15x with a throwaway")
    print("    suffix — the way hardneg.py repeats each row 41x and wrap_payload emits")
    print("    3 rows per payload. To isolate MEMORIZATION from real learning, each")
    print("    base's label is a per-payload property with no transferable pattern")
    print("    (its distinctive token is the only clue). So any success on UNSEEN")
    print("    bases must be luck, and any success on SEEN bases is pure memorization.")

    rng = np.random.default_rng(0)
    bases = []
    for i in range(80):
        tok = "".join(rng.choice(list("0123456789abcdef"), 8))
        bases.append((f"q=session_{tok}&t={tok}", i % 2))   # label = memorize-only
    rows = []  # (text, label, base_id)
    for bid, (text, lab) in enumerate(bases):
        for k in range(15):
            rows.append((f"{text}&n={k}", lab, bid))

    texts_b = [t for t, _, _ in rows]
    y_b = np.array([l for _, l, _ in rows], float)
    base_id = np.array([b for _, _, b in rows])
    Xb = np.stack([featurize(t) for t in texts_b])

    def nn_accuracy(train_idx, val_idx):
        """1-nearest-neighbour in hashed space. PURE memorization — the perfect
        probe for leakage: if a val row's duplicate is in train, it scores 100%;
        if the val base is unseen, it can only guess."""
        Xt, yt = Xb[train_idx], y_b[train_idx]
        correct = 0
        for i in val_idx:
            sims = Xt @ Xb[i]            # cosine (vectors are unit-normalized)
            pred = yt[int(np.argmax(sims))]
            correct += int(pred == y_b[i])
        return correct / len(val_idx)

    n = len(rows)
    split_rng = np.random.default_rng(3)
    perm = split_rng.permutation(n)
    cut = int(n * 0.8)
    row_train, row_val = perm[:cut], perm[cut:]           # LEAKY: split by row
    acc_row = nn_accuracy(row_train, row_val)

    ubases = np.unique(base_id)
    split_rng.shuffle(ubases)
    tr_bases = set(ubases[: int(len(ubases) * 0.8)].tolist())
    grp_train = np.array([i for i in range(n) if base_id[i] in tr_bases])
    grp_val = np.array([i for i in range(n) if base_id[i] not in tr_bases])
    acc_grp = nn_accuracy(grp_train, grp_val)

    print(f"\n    split-by-ROW   (leaky):  validation accuracy = {acc_row:.1%}")
    print(f"    split-by-GROUP (honest): validation accuracy = {acc_grp:.1%}  (~ chance)")
    print(f"    INFLATION from leakage: {acc_row - acc_grp:+.1%}")
    print("    Same data, same model. Row-splitting let the model find each val row's")
    print("    duplicate in training and copy the answer, so it looks PERFECT. The")
    print("    honest group-split reveals the truth: on payloads it never saw, this")
    print("    memorizer is barely better than a coin flip. That entire 100% was leak.")
    print("    (Real corpora mix a genuine, transferable signal WITH memorizable")
    print("    specifics, so real inflation is partial — but always present, and")
    print("    always in the optimistic direction.)")

    step("Connect to your repo (this is the concrete critique from before)")
    print("    * generate.py invents correlated rows; hardneg.py repeats each base")
    print("      41x; dataset.wrap_payload emits up to 3 requests per payload. All")
    print("      create near-duplicates.")
    print("    * train.split() and neural.train_file() split by ROW, so those copies")
    print("      straddle train/val. The AUC 0.998 in neural.json is therefore an")
    print("      UPPER bound, not the field number.")
    print("    * The fix already half-exists: http_sets.holdout_file() reconstructs")
    print("      the untouched 50% of the corpus. Grade against THAT (a group split),")
    print("      not a 20% slice of the trained sample.")

    step("Takeaways — the practitioner's mindset")
    print("    1. A model is only as robust as the transformations it trained on.")
    print("       Assume attackers mutate; train on mutants (mutate.py).")
    print("    2. Duplicates across a split inflate metrics. Split by GROUP/payload,")
    print("       never by row, whenever rows are correlated.")
    print("    3. Trust a metric only when you know the eval set is truly unseen.")
    print("    4. Defense in depth (Lesson 11) is what makes an imperfect model safe")
    print("       to ship: the gate caps the damage of any single model being wrong.")

    step("Course complete. Go re-read train.py, neural.py, eval.py, mutate.py.")
    print("    You now understand every line — and you can see what you'd improve.")


# ---------------------------------------------------------------------------
# EXERCISES  (solutions/lesson_12_solution.py)
# ---------------------------------------------------------------------------
# 1. Add a new mutation (e.g. URL-encode the payload, or insert a null byte) and
#    see if it evades the adversarially-trained model. The arms race never ends;
#    which transformations generalize and which don't?
#
# 2. In Part B, raise the per-base repeat count from 15 to 60. Does the row-split
#    inflation get WORSE? Explain why more duplication widens the leakage gap.
#
# 3. Reproduce the leakage on the REAL repo's shape: write a group-aware split for
#    lesson_11's generate_corpus (group by family+template) and compare the
#    capstone's detect-recall under row-split vs group-split. Report both honestly.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
