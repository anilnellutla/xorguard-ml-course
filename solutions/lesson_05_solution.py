"""Lesson 5 solutions."""
import _path  # noqa: F401
import numpy as np
from lesson_05_regularization_and_priors import FEATURES, fit, make_data

JNDI = FEATURES.index("jndi")


def make_data_njndi(n_jndi, seed=0):
    """Ex1: same as make_data but with a tunable number of JNDI examples."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(200):
        rows.append([rng.uniform(0, .2), rng.uniform(0, .3), 0, 1]); labels.append(0)
    for _ in range(200):
        rows.append([rng.uniform(.6, 1), rng.uniform(.4, .9), 0, 1]); labels.append(1)
    for _ in range(n_jndi):
        rows.append([rng.uniform(0, .2), rng.uniform(0, .3), 1, 1]); labels.append(1)
    X = np.array(rows); y = np.array(labels, float)
    p = rng.permutation(len(y)); return X[p], y[p]


if __name__ == "__main__":
    print("Ex1 — more JNDI data lets L2-to-zero recover the weight (prior needed less):")
    for n in (6, 30, 100):
        X, y = make_data_njndi(n)
        w = fit(X, y, l2=0.5, prior=None)
        print(f"  {n:3d} JNDI examples -> jndi weight = {w[JNDI]:+.3f}")
    print("  With enough data the data speaks for itself; the prior matters most when")
    print("  the signal is RARE — exactly JNDI's situation in real corpora.")

    print("\nEx2 — the data can OVERRULE a wrong prior (prior is a bias, not a cage):")
    X, y = make_data_njndi(200)                 # lots of JNDI attacks
    bad_prior = np.array([1.5, 0.5, -3.0, -1.5])  # prior says jndi is BENIGN evidence
    w = fit(X, y, l2=0.5, prior=bad_prior, init=bad_prior)
    print(f"  wrong prior jndi=-3.0, but with 200 JNDI attacks -> learned {w[JNDI]:+.3f}")
    print("  Enough evidence drags the weight back positive despite the bad prior.")

    print("\nEx3 — l2 sweep: stronger regularization -> smaller ||w|| (humbler model):")
    X, y = make_data_njndi(6)
    for l2 in (0.0, 0.1, 1.0, 10.0):
        w = fit(X, y, l2=l2, prior=None)
        print(f"  l2={l2:<5} ||w||={np.linalg.norm(w):.3f}")
