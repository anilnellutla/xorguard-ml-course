# ML from scratch, for xorguard

A hands-on machine-learning course, taught from zero, built around the two models
your `xorguard-ml` lab actually ships. No frameworks — just Python and NumPy, the
same tools the real repo uses. By the end you will have re-built, from scratch and
by hand, both models in `xorguard-ml`:

- the **32-d logistic regression** (`xorguard-supervised-v1`)
- the **hashed n-gram MLP** (`xorguard-neural-v1`)

...and you will understand every line of `train.py`, `neural.py`, and `eval.py`.

---

## How this course works

You are the student. I am your guru. The deal between us:

1. **You run every lesson.** Reading is not learning. Each lesson is a runnable
   Python file that narrates itself as it executes. Run it, read the output next
   to the code, then change a number and run it again.
2. **You do the exercises.** Every lesson ends with `EXERCISES`. Attempt them
   before peeking at `solutions/`. The struggle is the point — that's where the
   understanding actually forms.
3. **We build everything from scratch.** No scikit-learn, no PyTorch. When we
   need gradient descent, we write gradient descent. This is how you *own* the
   knowledge instead of renting it.

There is no math prerequisite beyond high-school algebra. Where calculus appears
(twice: the logistic gradient and backprop), I derive it in plain language first,
then show the one line of code it becomes.

---

## Setup

```bash
cd xorguard-ml-course
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Two ways to take the course

**1. As scripts** (numpy only, runs anywhere, no display needed):

```bash
python lessons/lesson_01_features.py
python lessons/lesson_02_score_and_sigmoid.py
# ...and so on
```

Every script runs with **numpy alone**. `matplotlib` is optional — scripts draw
ASCII charts when it is missing.

**2. As Jupyter notebooks** (interactive, with real inline plots — recommended):

```bash
pip install -r requirements-notebooks.txt   # jupyter + matplotlib
jupyter lab                                  # or just open a notebook in VSCode
```

Open `notebooks/lesson_01_features.ipynb` and run cells top to bottom. The
notebooks are a thin interactive layer: they **import** the canonical functions
from `lessons/*.py` (one source of truth — no forked copies) and show each one's
real source live, so editing a notebook cell and editing the script stay in sync.
They ship pre-executed, so you can also just read them on GitHub/VSCode without a
kernel. Regenerate them any time with `python tools/build_notebooks.py`.

---

## The syllabus

Each lesson maps to real code in your `../xorguard-ml` repo. That column is the
whole point: this is not generic ML, it is *your project's* ML.

| # | Lesson | Core idea | Maps to in `xorguard-ml` |
|---|--------|-----------|--------------------------|
| 0 | [Orientation](lessons/lesson_00_orientation.md) | What ML is, and the two models we're chasing | whole repo |
| 1 | [Features](lessons/lesson_01_features.py) | Turning requests into numbers | `generate.py`, the 32-d contract |
| 2 | [Score & sigmoid](lessons/lesson_02_score_and_sigmoid.py) | The weighted sum that becomes a probability | `train.sigmoid`, `predict_proba` |
| 3 | [Loss](lessons/lesson_03_loss.py) | Measuring wrongness so a machine can fix it | (the objective `fit_logistic` minimizes) |
| 4 | [Gradient descent](lessons/lesson_04_gradient_descent.py) | How the machine actually learns | `train.fit_logistic` |
| 5 | [Regularization & priors](lessons/lesson_05_regularization_and_priors.py) | Not overfitting; the "don't erase JNDI" trick | `train.py:33-34` warm-start prior |
| 6 | [Class imbalance](lessons/lesson_06_class_imbalance.py) | When 95% benign breaks naive training | `train.py:38-41` sample weights, `fp_weight` |
| 7 | [Evaluation & thresholds](lessons/lesson_07_evaluation_and_thresholds.py) | Precision, recall, TNR, AUC, calibration | `eval.py` |
| 8 | [Neural networks](lessons/lesson_08_neural_networks.py) | Why a straight line isn't enough | `neural.forward` |
| 9 | [Backpropagation](lessons/lesson_09_backprop.py) | Training a net, by hand | `neural.fit_mlp` |
| 10 | [Text → numbers](lessons/lesson_10_text_hashing_ngrams.py) | N-grams and the hashing trick | `neural.featurize` |
| 11 | [Capstone: a tiny WAF](lessons/lesson_11_capstone_mini_waf.py) | Both models, end to end | ties the whole repo together |
| 12 | [Adversarial & pitfalls](lessons/lesson_12_adversarial_and_pitfalls.py) | Evasion, and how metrics lie | `mutate.py`, my critique of the eval split |

Work top to bottom. Each lesson assumes the one before it.

---

## A word from your guru

Machine learning has a reputation for being deep magic. It is not. It is three
ideas, repeated:

1. **Turn the thing into numbers** (features).
2. **Guess with a formula that has knobs** (the model).
3. **Measure how wrong you are, and turn the knobs to be less wrong** (training).

Everything else — logistic regression, neural networks, the whole zoo — is a
variation on those three moves. You already write firewalls in Rust; you have
the harder skill. This will feel small by comparison once it clicks.

Let's begin. Open `lessons/lesson_00_orientation.md`.
