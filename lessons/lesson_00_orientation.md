# Lesson 0 — Orientation

*Read this one. No code yet. It's the map before the territory.*

---

## What you're actually going to learn

Your `xorguard-ml` repo fits two models. That's the whole repo. Everything else —
downloading payload lists, building mock HTTP, calling the engine — is plumbing to
feed those two models. Here they are, and here is where in this course you'll
build each one from nothing:

**Model 1 — the logistic regression (`xorguard-supervised-v1`).**
It reads 32 numbers describing a request (how strongly SQLi indicators fired, how
random the text looks, how long the path is, ...) and outputs one number: the
probability the request is an attack. It is *32 weights and a formula*. That's it.
You'll build it in **Lessons 2–7**.

**Model 2 — the neural network (`xorguard-neural-v1`).**
It reads the *raw decoded text* of the query and body — the actual payload
characters — and also outputs an attack probability. It's a small neural net:
512 numbers in, 64 hidden units, 1 out. You'll build it in **Lessons 8–11**.

That's the destination. Two files of NumPy, ~300 lines total, and you'll have
written every line and know why it's there.

---

## The three moves (memorize these)

All of machine learning, this whole course, is three moves on repeat:

### Move 1 — Turn the thing into numbers ("features")

A model can't read an HTTP request. It reads numbers. So step one is always:
convert your messy real-world thing into a fixed list of numbers called a
**feature vector**.

```
GET /search?q=' OR 1=1--     ──►     [0.9, 0.0, 0.1, ..., 1.0]
   (a request)                        (32 numbers the model can eat)
```

### Move 2 — Guess with a formula that has knobs ("the model")

The model is just a formula with adjustable numbers in it, called **weights**.
Feed the feature vector in, get a number out. At first the weights are garbage and
the guesses are garbage. That's fine.

```
guess = some_formula(features, weights)
```

### Move 3 — Measure the wrongness, turn the knobs to reduce it ("training")

You have example requests where you *know* the right answer (attack or not). So you
measure how wrong the model's guesses are — a single number called the **loss** —
and then nudge every weight in the direction that makes the loss smaller. Repeat a
few hundred times. The weights stop being garbage. That's learning.

```
loss = how_wrong(guesses, correct_answers)
weights = weights - small_step_toward_less_loss
```

Logistic regression, neural networks, transformers, all of it: **Move 2 changes,
Moves 1 and 3 stay basically the same.** Hold onto that. It keeps the whole field
from feeling like a thousand unrelated tricks.

---

## Vocabulary you'll meet (don't memorize, just recognize)

You'll learn each of these properly when we get to it. This is just so the words
aren't scary the first time:

- **Feature / feature vector** — the input numbers. (Lesson 1)
- **Weights** — the model's adjustable knobs. (Lesson 2)
- **Label** — the known correct answer. In this project: `1` = malicious,
  `0` = benign. (Lesson 1)
- **Sigmoid** — the S-shaped function that turns any number into a probability
  between 0 and 1. (Lesson 2)
- **Loss** — one number measuring total wrongness. Lower is better. (Lesson 3)
- **Gradient descent** — the "nudge the weights downhill" learning loop. (Lesson 4)
- **Overfitting** — memorizing the examples instead of learning the pattern.
  (Lesson 5)
- **Precision / recall / TNR / AUC** — the ways we grade a finished model. (Lesson 7)
- **Neural network / MLP** — a fancier Move-2 formula that can learn curved
  patterns a straight line can't. (Lesson 8)
- **Backpropagation** — gradient descent for neural nets. (Lesson 9)

---

## Why this project uses *small* models (an important point)

You might expect a modern security product to use some giant deep-learning model.
Yours deliberately doesn't, and that's a feature, not a limitation:

- The model runs **inline on every single HTTP request**. It must answer in
  microseconds. A 32-weight formula does.
- The weights are **human-readable**. You can open `models/supervised.json` and
  literally see that `jndi` has weight 3.8 — the model is auditable.
- The Rust engine must run **the exact same math** at inference. Simple formulas
  port to Rust perfectly; a 2-billion-parameter transformer does not.

So this course teaches *classical* ML on purpose. You will not need transformers or
LLMs to understand a single line of your repo. Don't let anyone route you there
first — it would be the wrong tool for this job.

---

## How to study

For each lesson file:

1. **Run it first**, before reading the code: `python lessons/lesson_01_features.py`.
   Watch what it prints.
2. **Then read the code top to bottom**, matching each printed line to the code
   that produced it.
3. **Change one number and re-run.** Break it on purpose. Curiosity is the fastest
   teacher.
4. **Do the exercises** at the bottom before opening `solutions/`.

Ready. Open `lesson_01_features.py` and run it.
