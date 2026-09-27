"""
Lesson 4 — Gradient descent: how the machine actually learns
============================================================

GURU NOTE
---------
This is the heart of the whole course. Everything before was setup; everything
after is a variation on this. We have a loss (Lesson 3). We want the weights that
make it smallest. We will NOT guess. We'll compute which direction reduces the
loss and take a small step that way. Repeat a few hundred times. Done.

THE ONE PIECE OF CALCULUS (in plain words)
------------------------------------------
The "gradient" is just the slope of the loss with respect to each weight: "if I
increase this weight a tiny bit, does the loss go up or down, and how fast?" Walk
in the opposite direction (downhill) and the loss drops.

For logistic regression the gradient is famous for being dead simple. For one
weight j it works out to:

        gradient_j = average over all examples of  (p_i - y_i) * x_ij

In words: (what I predicted minus the truth) times (the feature value). If I
over-predicted (p > y) on examples where feature j was large, then weight j is too
high — push it down. That's the entire idea. In matrix form for all weights at
once:

        grad = X.T @ (p - y) / n

Look at train.py lines 43-45. It is LITERALLY this:
        p = sigmoid(x @ w)
        err = (p - y) * sample_w        # sample_w is Lesson 6; ignore for now
        grad = x.T @ err / n

You already understand the real code. Now let's build it.

Run me:  python lessons/lesson_04_gradient_descent.py
"""

from __future__ import annotations

import numpy as np

from _util import ascii_bars, step, title
from lesson_01_features import build_dataset
from lesson_02_score_and_sigmoid import add_bias, predict_proba, sigmoid
from lesson_03_loss import log_loss


def fit_logistic(X, y, steps: int = 400, lr: float = 0.5, verbose: bool = True):
    """Train logistic-regression weights by gradient descent.

    This is a stripped-down train.fit_logistic: no regularization (Lesson 5) and
    no class weights (Lesson 6) yet, so the CORE learning loop is unmistakable.
    """
    n, d = X.shape
    w = np.zeros(d)          # start from all-zeros: the model knows nothing
    w[-1] = -0.5             # ...except a mild 'assume benign' bias, like the repo

    for step_i in range(steps):
        p = sigmoid(X @ w)            # 1) predict with current weights
        err = p - y                  # 2) how far off, and which sign
        grad = X.T @ err / n         # 3) THE gradient (derived above)
        step_lr = lr / (1.0 + step_i / 150.0)  # 4) decay: big steps early, small late
        w -= step_lr * grad          # 5) step downhill

        if verbose and step_i in (0, 1, 2, 5, 20, 100, steps - 1):
            print(f"      step {step_i:4d}   loss={log_loss(y, p):.4f}   "
                  f"w=[{w[0]:+.2f}, {w[1]:+.2f}, {w[2]:+.2f}]")
    return w


def main() -> None:
    title("Lesson 4 — Gradient descent: how the machine learns")

    X, y = build_dataset()
    Xb = add_bias(X)

    step("Start from zeros. The model has NO idea. Watch the loss fall.")
    print("    Each step: predict -> measure error -> compute gradient -> step down.")
    print()
    w = fit_logistic(Xb, y)

    step("The learned weights (found automatically, not guessed):")
    ascii_bars(["special_ratio", "sqli_hits", "bias"], w)
    print("    Compare to Lesson 2's hand guess [2.0, 4.0, -1.5]. Gradient descent")
    print("    landed somewhere sensible on its own — big +sqli, negative bias.")

    step("How good is the trained model?")
    p = predict_proba(Xb, w)
    acc = float(np.mean((p >= 0.5) == (y >= 0.5)))
    print(f"    final loss = {log_loss(y, p):.4f}")
    print(f"    accuracy   = {acc:.0%} on the training set")

    step("Feel the two big knobs. First, LEARNING RATE (step size).")
    print("    Too small: crawls, barely moves. Too big: overshoots and DIVERGES —")
    print("    the loss goes UP instead of down. There's a healthy middle.")
    for lr in (0.02, 0.5, 200.0):
        w_try = fit_logistic(Xb, y, steps=200, lr=lr, verbose=False)
        p_try = predict_proba(Xb, w_try)
        note = ""
        if lr == 0.02:
            note = "  <- too small: hardly learned"
        elif lr == 200.0:
            note = "  <- too big: overshot, loss got WORSE"
        print(f"      lr={lr:<6} final loss={log_loss(y, p_try):.4f}{note}")
    print("    lr=0.5 is the healthy middle here. Tuning lr is a real part of the job.")

    step("Second knob: NUMBER OF STEPS. More steps -> lower loss, with diminishing")
    print("    returns as it approaches the bottom of the valley.")
    for steps in (5, 50, 500):
        w_try = fit_logistic(Xb, y, steps=steps, lr=0.5, verbose=False)
        p_try = predict_proba(Xb, w_try)
        print(f"      steps={steps:<4} final loss={log_loss(y, p_try):.4f}")

    step("Why the learning rate DECAYS during training (step_lr line):")
    print("    Early on you're far from the bottom — take big strides. Near the")
    print("    bottom, big strides bounce you around the minimum, so shrink them to")
    print("    settle in precisely. train.py does the same: lr/(1 + step/250).")

    step("Connect to your repo")
    print("    * You just wrote the beating heart of train.fit_logistic. The real")
    print("      one adds only: a regularization pull (Lesson 5) and per-sample")
    print("      class weights (Lesson 6). The predict/err/grad/step core is IDENTICAL.")
    print("    * This same loop, with a chain-rule twist, trains the neural net too")
    print("      (Lesson 9). Master this and backprop is just 'this, but twice'.")

    step("Takeaways")
    print("    1. Gradient = slope of the loss w.r.t. each weight.")
    print("    2. grad = X.T @ (p - y) / n  for logistic regression. Memorize it.")
    print("    3. Learning = repeat: predict, error, gradient, step downhill.")
    print("    4. Learning rate and step count are the knobs you'll tune most.")


# ---------------------------------------------------------------------------
# EXERCISES  (solutions/lesson_04_solution.py)
# ---------------------------------------------------------------------------
# 1. Instrument fit_logistic to RECORD the loss at every step into a list, and
#    print it every 25 steps. Confirm it decreases monotonically (with a good lr).
#
# 2. Set lr=50 and steps=200. What happens to the loss? Explain in one sentence
#    what "divergence" looks like and why a too-big step causes it.
#
# 3. Remove the bias initialization (start w entirely at zeros). Does the final
#    model change much? Why is the starting point less important than the lr here?
#
# 4. HARDER: implement gradient checking. Pick weight j, compute the loss at w and
#    at w with w[j] nudged by 1e-5, and estimate the slope numerically as
#    (loss2 - loss1)/1e-5. Confirm it matches grad[j] from the formula. This is
#    how you'd catch a bug in a hand-derived gradient (crucial in Lesson 9).
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
