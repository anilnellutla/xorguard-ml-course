"""
Lesson 11 — Capstone: a tiny WAF with BOTH models
==================================================

GURU NOTE
---------
Time to assemble everything. We'll build a miniature xorguard: two models scoring
the same traffic, thresholds calibrated for each, and the crucial 'defense in
depth' wiring where NEITHER model can block on its own. This is your repo in
miniature — every piece is a lesson you've already done:

  Lesson 1  features               Lesson 7  thresholds / calibration
  Lesson 4  logistic training      Lesson 9  MLP training (backprop)
  Lesson 6  class weights          Lesson 10 hashed n-gram featurization

The two models, mirroring the repo:
  * SUPERVISED (logistic) over hand-made indicator features  -> like supervised.json
  * NEURAL (MLP) over hashed char n-grams of the raw text     -> like neural.json

The decision policy, mirroring the repo:
  * DETECT (log it) if the fused score crosses the detect threshold.
  * PREVENT (403) ONLY if it also looks like a CRITICAL family (here: RCE).
    A high neural score on 'alert(' alone must NOT mint a 403. Same rule as the
    real engine.

Run me:  python lessons/lesson_11_capstone_mini_waf.py
"""

from __future__ import annotations

import numpy as np

from _util import step, title
from lesson_02_score_and_sigmoid import sigmoid
from lesson_06_class_imbalance import fit as fit_logistic
from lesson_07_evaluation_and_thresholds import at_threshold, calibrate_for_tnr, roc_auc
from lesson_08_neural_networks import forward
from lesson_09_backprop import fit_mlp
from lesson_10_text_hashing_ngrams import featurize

# --------------------------------------------------------------------------
# 1. A synthetic payload corpus. Templates + random fills give us VOLUME, which
#    the MLP needs. Each item is (text, label, family). family drives the
#    critical-family gate. 'benign' items are label 0.
# --------------------------------------------------------------------------
CRITICAL = {"rce"}   # only RCE may escalate to a 403 in this toy WAF


def generate_corpus(seed: int = 0):
    rng = np.random.default_rng(seed)
    words = ["red", "blue", "winter", "coat", "shoes", "laptop", "phone", "book"]
    names = ["ada", "alice", "bob", "carol", "dave"]
    items = []

    def add(text, label, fam):
        items.append((text, label, fam))

    for _ in range(300):   # benign
        kind = rng.integers(0, 4)
        if kind == 0:
            add(f"q={rng.choice(words)}+{rng.choice(words)}&page={rng.integers(1,9)}", 0, "benign")
        elif kind == 1:
            add(f"email={rng.choice(names)}@example.com", 0, "benign")
        elif kind == 2:
            add(f"id={rng.integers(1, 9999)}", 0, "benign")
        else:
            add(f"sort=price&color={rng.choice(words)}", 0, "benign")

    for _ in range(120):   # sqli
        a = rng.integers(1, 9)
        kind = rng.integers(0, 3)
        if kind == 0:
            add(f"q=' OR {a}={a}--", 1, "sqli")
        elif kind == 1:
            add(f"user=admin' OR '{a}'='{a}", 1, "sqli")
        else:
            add(f"id={a}' UNION SELECT password FROM users--", 1, "sqli")

    for _ in range(120):   # xss (NOT critical here)
        n = rng.integers(1, 99)
        kind = rng.integers(0, 3)
        if kind == 0:
            add(f"q=<script>alert({n})</script>", 1, "xss")
        elif kind == 1:
            add(f"name=<svg onload=alert({n})>", 1, "xss")
        else:
            add(f"c=<img src=x onerror=alert({n})>", 1, "xss")

    for _ in range(120):   # rce (CRITICAL)
        f = rng.choice(["passwd", "shadow", "hosts"])
        kind = rng.integers(0, 3)
        if kind == 0:
            add(f"q=; cat /etc/{f}", 1, "rce")
        elif kind == 1:
            add(f"cmd=$(cat /etc/{f})", 1, "rce")
        else:
            add(f"x=| cat /etc/{f}", 1, "rce")

    rng.shuffle(items)
    return items


# --------------------------------------------------------------------------
# 2. Two featurizations of the same text.
# --------------------------------------------------------------------------
def indicator_features(text: str) -> np.ndarray:
    """Hand-made 'engine-style' indicators + bias. Stand-in for the 32-d vector."""
    t = text.lower()
    sqli = min(1.0, sum(t.count(k) for k in ["' or", "union", "select", "'='", "--"]) / 3)
    xss = min(1.0, sum(t.count(k) for k in ["<script", "onerror", "onload", "<svg", "<img"]) / 2)
    rce = min(1.0, sum(t.count(k) for k in ["cat ", "/etc/", "$(", "; ", "| "]) / 2)
    special = sum(c in "'\"<>;()=|&$/-" for c in text) / max(len(text), 1)
    return np.array([sqli, xss, rce, special, 1.0])


def split_items(items, val_frac=0.25, seed=11):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    cut = int(len(items) * (1 - val_frac))
    tr = [items[i] for i in idx[:cut]]
    va = [items[i] for i in idx[cut:]]
    return tr, va


def main() -> None:
    title("Lesson 11 — Capstone: a tiny WAF with BOTH models")
    items = generate_corpus()
    tr, va = split_items(items)
    print(f"    corpus: {len(items)} requests  ({sum(1 for _,l,_ in items if l)} attacks).")
    print(f"    split : {len(tr)} train / {len(va)} validation (held out).")

    # ---- Train the SUPERVISED logistic on indicator features ----
    step("Train model A — logistic regression on indicator features (Lessons 4/6).")
    Xtr = np.stack([indicator_features(t) for t, _, _ in tr])
    ytr = np.array([l for _, l, _ in tr], float)
    w_log = fit_logistic(Xtr, ytr, balance=True, fp_weight=2.0)
    Xva = np.stack([indicator_features(t) for t, _, _ in va])
    yva = np.array([l for _, l, _ in va], float)
    p_log_va = sigmoid(Xva @ w_log)
    print(f"    logistic AUC on validation = {roc_auc(yva, p_log_va):.3f}")

    # ---- Train the NEURAL MLP on hashed n-grams ----
    step("Train model B — MLP on hashed char n-grams (Lessons 9/10).")
    print("    featurizing text into 512-dim hashed vectors...")
    Htr = np.stack([featurize(t) for t, _, _ in tr])
    Hva = np.stack([featurize(t) for t, _, _ in va])
    print("    training the net (watch accuracy climb)...")
    w1, b1, w2, b2 = fit_mlp(Htr, ytr, hidden=16, epochs=120, lr=0.2, fp_weight=2.0)
    p_mlp_va = forward(Hva, w1, b1, w2, b2)
    print(f"    neural AUC on validation = {roc_auc(yva, p_mlp_va):.3f}")

    # ---- Fuse the two scores (the repo mixes them into Detect) ----
    step("Fuse: the engine blends both scores. We'll average them (Lesson 7 to grade).")
    p_fused_va = 0.5 * p_log_va + 0.5 * p_mlp_va
    print(f"    fused AUC on validation = {roc_auc(yva, p_fused_va):.3f}")
    print("    Two independent views (indicators + raw text) beat either alone,")
    print("    exactly why the engine keeps both models.")

    # ---- Calibrate detect / prevent thresholds (Lesson 7) ----
    step("Calibrate thresholds on validation (detect logs, prevent blocks).")
    prevent_t, _, _ = calibrate_for_tnr(yva, p_fused_va, target_tnr=0.995)
    detect_t, _, _ = calibrate_for_tnr(yva, p_fused_va, target_tnr=0.95)
    if detect_t >= prevent_t:
        detect_t = max(0.5, prevent_t - 0.1)
    print(f"    detect threshold  = {detect_t:.3f}   (TNR>=95%)")
    print(f"    prevent threshold = {prevent_t:.3f}   (TNR>=99.5%)")

    # ---- The decision policy, with the critical-family gate ----
    def decide(text: str):
        """Return one of ALLOW / DETECT / PREVENT for a request."""
        f_ind = indicator_features(text)
        fam = ("rce" if f_ind[2] > 0 else "xss" if f_ind[1] > 0 else
               "sqli" if f_ind[0] > 0 else "benign")
        p_log = float(sigmoid(f_ind @ w_log))
        p_mlp = float(forward(featurize(text)[None, :], w1, b1, w2, b2)[0])
        score = 0.5 * p_log + 0.5 * p_mlp
        if score >= detect_t:
            # PREVENT only if it ALSO looks like a critical family (the repo's rule)
            if score >= prevent_t and fam in CRITICAL:
                return "PREVENT", score, fam
            return "DETECT", score, fam
        return "ALLOW", score, fam

    step("THE KEY WIRING: a high score alone cannot 403. Only critical families do.")
    print("    This is why the repo says 'a high neural score on alert( cannot mint")
    print("    a 403'. XSS scores high -> DETECT (logged), never blocked. RCE can")
    print("    escalate to PREVENT. Watch:")
    demo = [
        "q=red+shoes&page=2",                 # benign  -> ALLOW
        "id=57",                              # benign  -> ALLOW
        "q=<script>alert(1)</script>",        # xss     -> DETECT (not critical)
        "user=admin' OR '1'='1",              # sqli    -> DETECT
        "q=; cat /etc/passwd",                # rce     -> PREVENT (critical)
        "cmd=$(cat /etc/shadow)",             # rce     -> PREVENT
    ]
    print(f"\n    {'decision':8}  {'score':>5}  {'family':7}  request")
    for t in demo:
        d, sc, fam = decide(t)
        print(f"    {d:8}  {sc:.3f}  {fam:7}  {t}")

    step("Grade the whole policy on the held-out validation set.")
    decisions = [decide(t) for t, _, _ in va]
    detected = np.array([d in ("DETECT", "PREVENT") for d, _, _ in decisions])
    prevented = np.array([d == "PREVENT" for d, _, _ in decisions])
    truth = yva >= 0.5
    fams = np.array([f for _, _, f in va])
    crit = np.array([f in CRITICAL for f in fams])
    det_recall = detected[truth].mean()
    det_tnr = (~detected[~truth]).mean()
    print(f"    DETECT : catches {det_recall:.1%} of ALL attacks, "
          f"spares {det_tnr:.1%} of benign.")
    if crit.sum():
        prev_recall_crit = prevented[truth & crit].mean()
        print(f"    PREVENT: blocks {prev_recall_crit:.1%} of CRITICAL (rce) attacks.")
    fp_prevent = prevented[~truth].sum()
    fp_prevent_xss = prevented[(fams == 'xss')].sum()
    print(f"    False blocks on benign : {fp_prevent}  (want 0 — never 403 a customer)")
    print(f"    Blocks on XSS attacks  : {fp_prevent_xss}  (want 0 — not a critical family)")

    step("Connect to your repo (the whole thing, now)")
    print("    * model A = supervised.json (logistic on the 32-d vector).")
    print("    * model B = neural.json (MLP on hashed n-grams of decoded text).")
    print("    * fusion + detect/prevent thresholds + critical-family gate = the")
    print("      engine's Phase-3 decision. You just built a scale model of it.")
    print("    * The gate is a SAFETY property: even a wrong model only over-LOGS;")
    print("      a false 403 needs a critical family too. Blast radius stays small.")

    step("You're done. What you built, from scratch, in one course:")
    print("    features -> logistic (grad descent) -> regularized priors -> class")
    print("    weights -> evaluation/thresholds -> neural nets -> backprop -> hashed")
    print("    text -> a two-model WAF with calibrated, gated decisions.")
    print("    Now re-read train.py, neural.py, eval.py. You will recognize every line.")


# ---------------------------------------------------------------------------
# EXERCISES  (solutions/lesson_11_solution.py)
# ---------------------------------------------------------------------------
# 1. Make XSS a critical family too (add 'xss' to CRITICAL) and re-grade. What
#    happens to PREVENT coverage and to false blocks? This is a real policy lever.
#
# 2. Remove model B (score = p_log only) and re-grade the fused AUC and detect
#    recall. Then remove model A instead. Which attacks does each model catch that
#    the other misses? (Hint: the MLP sees raw text; the logistic sees indicators.)
#
# 3. Add a NEW attack family the models have never seen (e.g. a traversal payload
#    "../../etc/passwd") to the demo list. Does the text-based MLP flag it even
#    though no template trained on it? This is generalization from Lesson 10.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
