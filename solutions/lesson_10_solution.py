"""Lesson 10 solutions."""
import _path  # noqa: F401
import itertools
import string
from urllib.parse import unquote_plus

import numpy as np
from lesson_10_text_hashing_ngrams import HASH_DIM, featurize, fnv1a_32


def collisions(dim):
    seen, n = {}, 0
    for a, b, c in itertools.product(string.ascii_lowercase, repeat=3):
        bkt = fnv1a_32((a + b + c).encode()) % dim
        if bkt in seen:
            n += 1
        seen[bkt] = 1
    return n


if __name__ == "__main__":
    print("Ex1 — bigger hash dim -> far fewer collisions among 17,576 trigrams:")
    for dim in (512, 4096):
        print(f"  dim={dim:<5} collisions={collisions(dim)}")
    sqli = "q=' OR 1=1--"
    print(f"  '{sqli}' non-zero buckets: dim512={int(np.count_nonzero(featurize(sqli,512)))}"
          f"  dim4096={int(np.count_nonzero(featurize(sqli,4096)))}")
    print("  More buckets = less collision noise, but a bigger model. Repo chose 512.")

    print("\nEx2 — DECODING matters: encoded vs decoded look totally different:")
    enc = "%27%20OR%201=1--"
    dec = unquote_plus(enc)
    print(f"  raw-encoded vs raw-decoded cosine = {float(featurize(enc) @ featurize(dec)):.3f}")
    print(f"  '{enc}' decodes to '{dec}'. Without decoding the model sees '%27%20',")
    print("  not \"' OR\" — the attack's tell-tale n-grams are hidden. That's why the")
    print("  engine URL-decodes BEFORE featurizing (neural.payload_text).")

    print("\nEx3 — case-swap changes the vector; lowercasing neutralizes it:")
    a, b = "<script>", "<SCRIPT>"
    print(f"  raw cosine('{a}','{b}') = {float(featurize(a) @ featurize(b)):.3f}")
    print(f"  lowercased cosine       = {float(featurize(a.lower()) @ featurize(b.lower())):.3f}")
    print("  Lowercasing is a cheap, powerful defense against case-swap evasion (L12).")
