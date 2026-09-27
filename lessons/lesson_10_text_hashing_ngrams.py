"""
Lesson 10 — Text to numbers: n-grams and the hashing trick
==========================================================

GURU NOTE
---------
The logistic model ate 32 pre-computed features. The payload MLP is different: it
eats the RAW decoded text of the query and body. But a neural net needs a
fixed-length vector of numbers, and text is variable-length characters. How do we
turn "' OR 1=1--" into exactly 512 numbers, cheaply, and identically in Rust?

Three ideas, and they're all in neural.featurize:

  1. CHARACTER N-GRAMS. Slide a window over the text and collect every substring
     of length 3, 4, 5. "' or 1=1" -> "' o", "' or", "' or ", "or ", ... These
     little fragments are the vocabulary of attacks: "<scr", "=1=1", "un io" etc.

  2. THE HASHING TRICK. There are millions of possible n-grams. Instead of a
     dictionary, hash each one and use (hash mod 512) as its slot in a fixed
     512-length vector, adding 1 each time. No vocabulary file, fixed size, and
     trivially reproducible in Rust. The cost: sometimes two different n-grams
     land in the same slot (a 'collision'). We accept that noise.

  3. TIDY UP. log1p (so a fragment repeated 100x doesn't dominate) then
     L2-normalize (so a 2KB body and a 30-char query are on the same scale).

This lesson builds all three, exactly as neural.py does — including the same
FNV-1a hash with the same constants, so your vectors would match the engine's.

Run me:  python lessons/lesson_10_text_hashing_ngrams.py
"""

from __future__ import annotations

import numpy as np

from _util import step, title

# The SAME constants as neural.py, so this is bit-for-bit the engine's recipe.
HASH_DIM = 512
NGRAMS = (3, 4, 5)
MAX_CHARS = 2048
FNV_OFFSET = 2166136261
FNV_PRIME = 16777619


def fnv1a_32(data: bytes) -> int:
    """A tiny, fast, well-mixing hash. Chosen because it's trivial to reproduce
    in Rust byte-for-byte. XOR each byte in, multiply by a prime, keep 32 bits."""
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFF
    return h


def char_ngrams(text: str, n: int):
    """Every length-n character window. This is the 'vocabulary' extraction."""
    raw = text.encode("utf-8", "replace")[:MAX_CHARS]
    return [raw[i:i + n] for i in range(len(raw) - n + 1)] if len(raw) >= n else []


def featurize(text: str, dim: int = HASH_DIM) -> np.ndarray:
    """Text -> fixed 512-vector. IDENTICAL logic to neural.featurize.

    1) count 3,4,5-gram occurrences into `dim` hashed buckets
    2) log1p to squash repetition
    3) L2-normalize to unit length
    """
    raw = text.encode("utf-8", "replace")[:MAX_CHARS]
    v = np.zeros(dim)
    for n in NGRAMS:
        for i in range(len(raw) - n + 1):
            bucket = fnv1a_32(raw[i:i + n]) % dim
            v[bucket] += 1.0
    v = np.log1p(v)                     # 100 hits -> ~4.6, not 100
    nrm = np.linalg.norm(v)
    if nrm > 0:
        v /= nrm                        # same-length vectors, comparable across sizes
    return v


def main() -> None:
    title("Lesson 10 — Text to numbers: n-grams and the hashing trick")

    step("Step 1 — character n-grams. The payload's 'words' are tiny fragments.")
    sample = "' or 1=1"
    for n in (3, 4):
        grams = char_ngrams(sample, n)
        shown = [g.decode() for g in grams[:6]]
        print(f"    {n}-grams of {sample!r}: {shown} ...  ({len(grams)} total)")
    print("    Fragments like \"1=1\" and \"' o\" are exactly the tells of injection.")

    step("Step 2 — the hash. Each fragment -> a bucket number in [0, 512).")
    for frag in (b"<scr", b"1=1", b"uni", b"sel"):
        h = fnv1a_32(frag)
        print(f"    fnv1a({frag!r}) = {h:>10}  ->  bucket {h % HASH_DIM}")
    print("    No dictionary needed. The hash IS the lookup. Reproducible anywhere.")

    step("The cost of hashing: COLLISIONS. Different fragments, same bucket.")
    # find a natural collision among many english trigrams
    seen = {}
    collision = None
    import itertools, string
    for a, b, c in itertools.product(string.ascii_lowercase, repeat=3):
        g = (a + b + c).encode()
        bkt = fnv1a_32(g) % HASH_DIM
        if bkt in seen and collision is None:
            collision = (seen[bkt], g.decode(), bkt)
        seen[bkt] = g.decode()
    if collision:
        print(f"    {collision[0]!r} and {collision[1]!r} BOTH hash to bucket "
              f"{collision[2]}.")
    print("    With 512 buckets and thousands of fragments, collisions are constant.")
    print("    The model just treats it as a little noise — and it still works great.")
    print("    (Your repo uses 512; more buckets = fewer collisions but bigger model.)")

    step("Step 3 — full featurize: count -> log1p -> normalize.")
    payloads = {
        "benign query": "q=red+dress&page=2",
        "sqli":         "q=' OR 1=1--",
        "sqli variant": "q=' OR 2=2--",
        "xss":          "q=<script>alert(1)</script>",
    }
    vecs = {name: featurize(text) for name, text in payloads.items()}
    for name, v in vecs.items():
        nz = int(np.count_nonzero(v))
        print(f"    {name:14}: 512-vector, {nz} non-zero buckets, ||v||={np.linalg.norm(v):.3f}")

    step("The payoff — similar payloads produce similar vectors.")
    print("    Cosine similarity (1.0 = identical direction, 0 = unrelated):")

    def cos(a, b):
        return float(a @ b)   # already unit-normalized, so dot == cosine

    pairs = [("sqli", "sqli variant"), ("sqli", "xss"), ("sqli", "benign query")]
    for a, b in pairs:
        print(f"      {a:12} vs {b:12}: {cos(vecs[a], vecs[b]):.3f}")
    print("    The two SQLi payloads look alike to the model (high similarity), and")
    print("    unlike the benign query (low). THAT is what lets the MLP generalize")
    print("    from payloads it trained on to new variants it has never seen.")

    step("Connect to your repo")
    print("    * fnv1a_32, char n-grams, HASH_DIM=512, NGRAMS=(3,4,5), log1p +")
    print("      L2-normalize: every line here mirrors neural.featurize.")
    print("    * The model JSON stores hash_dim, ngrams, and the FNV constants so")
    print("      the Rust engine hashes text into the SAME buckets. If it didn't,")
    print("      the learned weights would be meaningless. That's why they're saved.")
    print("    * neural.payload_text first URL-decodes and lowercases — so %3C")
    print("      becomes '<' before featurizing. Decode lives in the engine.")

    step("Takeaways")
    print("    1. Char n-grams turn text into a bag of tell-tale fragments.")
    print("    2. The hashing trick maps unlimited fragments into fixed buckets.")
    print("    3. Collisions are accepted noise; the model tolerates them.")
    print("    4. log1p + L2-normalize make vectors comparable across payload sizes.")
    print("    5. Similar payloads -> similar vectors -> the MLP generalizes.")


# ---------------------------------------------------------------------------
# EXERCISES  (solutions/lesson_10_solution.py)
# ---------------------------------------------------------------------------
# 1. Raise HASH_DIM to 4096 and re-run the collision search. Do collisions get
#    rarer? Featurize the SQLi payload at dim=512 vs dim=4096 and compare the
#    non-zero counts. This is the size/accuracy trade the repo made.
#
# 2. Featurize "' OR 1=1--" and its URL-encoded form "%27%20OR%201=1--" WITHOUT
#    decoding. How similar are they? Now lowercase+urldecode both first (like
#    neural.payload_text) and compare again. Why does decoding matter so much?
#
# 3. Compute the cosine similarity between an XSS payload and the SAME payload
#    with the tag uppercased ("<SCRIPT>"). Then lowercase both first. This
#    previews Lesson 12: case-swapping is a real evasion the repo defends against.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
