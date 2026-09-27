"""Lesson 8 solutions."""
import _path  # noqa: F401
import numpy as np
from lesson_08_neural_networks import accuracy, forward, make_xor, relu

if __name__ == "__main__":
    X, y = make_xor()
    w1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    b1 = np.array([0.0, -1.0])
    w2 = np.array([[6.0, -12.0]])

    print("Ex1 — flip b2 to +3.0 and the MLP breaks:")
    for b2v in (-3.0, +3.0):
        acc = accuracy(y, forward(X, w1, b1, w2, np.array([b2v])))
        print(f"  b2={b2v:+.1f}: accuracy={acc:.1%}")
    print("  +3.0 shoves the baseline so high EVERYTHING reads as an attack. The")
    print("  hidden layer still computes the right thing; the output bias ruins it.")

    print("\nEx2 — the hidden layer does the work. h2 only lights up for BOTH-high:")
    for a, b, name in [(0, 1, "attack (one high)"), (1, 1, "benign (both high)")]:
        h = relu(np.array([[a, b]]) @ w1.T + b1)
        print(f"  A={a} B={b} [{name:18}] -> hidden h = {h[0]}")
    print("  Notice h[1] (the 'both-high' detector) is 0 for one-high, positive for")
    print("  both-high. That second unit is what lets the output cancel the both-case.")

    print("\nEx3 — can ONE hidden unit solve XOR? No.")
    print("  One ReLU unit = relu(w.x + b): a single bent line. XOR needs the region")
    print("  'exactly one high', which is between TWO parallel boundaries — you need")
    print("  at least two units (two lines) to fence it off. Try it: any (1,2) w1")
    print("  you pick leaves two XOR corners on the same side.")
