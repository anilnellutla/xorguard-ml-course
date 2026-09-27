"""
Lesson 8 — Neural networks: why a straight line isn't enough
============================================================

GURU NOTE
---------
Logistic regression draws ONE straight boundary through feature space. That's
powerful but limited. Some patterns can't be split by any straight line.

The classic example — and it's not abstract for you — is XOR / "combination"
logic: "feature A alone is fine, feature B alone is fine, but A AND B together is
an attack." A quoted string is normal. A SQL keyword is normal. Both in the same
parameter is suspicious. Your repo even has a `combo` feature for exactly this.
No single straight line can carve out "exactly one of A/B, but not both."

The fix is a NEURAL NETWORK (a multi-layer perceptron, MLP): stack a second layer
with a nonlinearity (ReLU) in between. The hidden layer bends the space so the
output layer CAN separate it with a line. This lesson shows a straight-line model
failing, then a tiny hand-built MLP succeeding, so you SEE what the hidden layer
buys you. Lesson 9 makes it LEARN those hidden weights.

Run me:  python lessons/lesson_08_neural_networks.py
"""

from __future__ import annotations

import numpy as np

from _util import step, title
from lesson_02_score_and_sigmoid import sigmoid
from lesson_06_class_imbalance import fit as fit_logistic


def make_xor(n: int = 400, noise: float = 0.15, seed: int = 0):
    """Points around the 4 corners of a square. Label = XOR of the corner.

    (0,0)->benign  (1,1)->benign     [both-low and both-high are FINE]
    (0,1)->attack  (1,0)->attack     [exactly one high is the ATTACK]
    This is the 'only suspicious in combination' pattern, made 2-D so we can see it.
    """
    rng = np.random.default_rng(seed)
    corners = [((0, 0), 0), ((1, 1), 0), ((0, 1), 1), ((1, 0), 1)]
    rows, labels = [], []
    for (cx, cy), lab in corners:
        for _ in range(n // 4):
            rows.append([cx + rng.normal(0, noise), cy + rng.normal(0, noise)])
            labels.append(lab)
    X = np.array(rows)
    y = np.array(labels, float)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def relu(z):
    return np.maximum(z, 0.0)


def forward(x, w1, b1, w2, b2):
    """A 2-layer MLP forward pass. IDENTICAL to neural.forward in your repo.

    Step 1 (hidden layer): h = relu(x @ w1.T + b1)   -- bends the space
    Step 2 (output layer): p = sigmoid(h @ w2.T + b2) -- draws the line
    Shapes: w1 (hidden,d)  b1 (hidden,)  w2 (1,hidden)  b2 (1,)
    """
    h = relu(x @ w1.T + b1)
    return sigmoid((h @ w2.T + b2).ravel())


def accuracy(y, p):
    return float(np.mean((p >= 0.5) == (y >= 0.5)))


def main() -> None:
    title("Lesson 8 — Neural networks: why a straight line isn't enough")
    X, y = make_xor()

    step("The data: 'attack only in COMBINATION'. Look at the four clusters.")
    print("    benign  '.' sit at (0,0) and (1,1)  — both features agree")
    print("    attack  '#' sit at (0,1) and (1,0)  — exactly one feature high")
    from _util import ascii_scatter
    ascii_scatter(X[:, 0], X[:, 1], y, xlabel="feature A", ylabel="feature B",
                  height=14)
    print("    There is NO straight line with attacks on one side, benign on the")
    print("    other. Convince yourself by trying to draw one. You can't.")

    step("Prove logistic regression (a straight line) fails here.")
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)   # add bias column
    w = fit_logistic(Xb, y, balance=True, fp_weight=1.0)
    p_lin = sigmoid(Xb @ w)
    print(f"    logistic accuracy = {accuracy(y, p_lin):.1%}  (barely better than a")
    print("    coin flip — a line simply cannot express XOR).")

    step("Now a tiny MLP: 2 inputs -> 2 hidden ReLU units -> 1 output.")
    print("    I'll hand-set weights so you can see the MECHANISM (Lesson 9 learns them).")
    print("    The two hidden units compute:")
    print("        h1 = relu(A + B)        (fires when EITHER is high)")
    print("        h2 = relu(A + B - 1)    (fires only when BOTH are high)")
    print("    Output = 6*h1 - 12*h2 - 3. The -12*h2 CANCELS the both-high case,")
    print("    leaving exactly-one-high as the only 'attack'. That's XOR, built from")
    print("    two straight lines bent by ReLU.")
    w1 = np.array([[1.0, 1.0], [1.0, 1.0]])   # both units sum A+B
    b1 = np.array([0.0, -1.0])                # second unit shifted by -1
    w2 = np.array([[6.0, -12.0]])             # combine: +6*h1, -12*h2
    b2 = np.array([-3.0])                      # baseline so benign lands below 0.5
    p_mlp = forward(X, w1, b1, w2, b2)
    print(f"\n    MLP accuracy = {accuracy(y, p_mlp):.1%}  <-- the hidden layer cracked it")

    step("Check it corner by corner (clean inputs).")
    for a, b, name in [(0, 0, "benign"), (1, 1, "benign"),
                       (0, 1, "ATTACK"), (1, 0, "ATTACK")]:
        p = float(forward(np.array([[a, b]]), w1, b1, w2, b2)[0])
        print(f"      A={a} B={b} [{name:6}] -> p={p:.3f}")
    print("    Both-low and both-high score low; exactly-one-high scores high. XOR.")

    step("The big idea")
    print("    A hidden layer + a nonlinearity (ReLU) lets the network build NEW")
    print("    features ('is exactly one high?') out of the raw ones, then separate")
    print("    those with a line. Stack enough and you can carve ANY shape. THIS is")
    print("    why your payload MLP can catch patterns the 32-d logistic cannot.")

    step("Connect to your repo")
    print("    * forward() here == neural.forward exactly: relu then sigmoid, same")
    print("      shapes. The real net is 512 inputs -> 64 hidden -> 1, but the math")
    print("      is what you just ran.")
    print("    * Why 64 hidden units there? More units = more 'bent lines' = more")
    print("      complex payload shapes it can recognize. 2 was enough for XOR.")

    step("Takeaways")
    print("    1. Logistic regression = one straight boundary. Great, but limited.")
    print("    2. XOR / 'malicious only in combination' has no straight boundary.")
    print("    3. A hidden layer + ReLU builds new features -> curved boundaries.")
    print("    4. forward pass = relu(x@W1.T + b1), then sigmoid(h@W2.T + b2).")


# ---------------------------------------------------------------------------
# EXERCISES  (solutions/lesson_08_solution.py)
# ---------------------------------------------------------------------------
# 1. Break the MLP: change b2 from -3.0 to +3.0 and re-run. Why does accuracy
#    collapse? (You moved the baseline so everything reads as an attack.)
#
# 2. The hidden layer is doing the work. Print h = relu(X @ w1.T + b1) for one
#    attack point and one benign point. Confirm h2 (the second column) only lights
#    up for both-high inputs.
#
# 3. Can ONE hidden unit solve XOR? Try w1 with shape (1,2) and a matching w2.
#    Convince yourself (and explain in a comment) why you need at least two.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
