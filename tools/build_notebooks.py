"""Generate Jupyter notebooks from the lesson .py modules.

Design: the notebooks are an INTERACTIVE LAYER, not a fork. They import the
canonical functions from lessons/*.py and display each one's real source via
inspect.getsource, so there is a single source of truth. Prose comes straight
from each lesson's module docstring; exercises from its EXERCISES block. A few
lessons get real matplotlib plots replacing the ASCII charts.

Run:  ./.venv/bin/python tools/build_notebooks.py
"""

from __future__ import annotations

import ast
import pathlib
import re

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = pathlib.Path(__file__).resolve().parent.parent
LESSONS = ROOT / "lessons"
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)

# Per-lesson: which functions to SHOW (source), and an optional matplotlib plot.
SHOW = {
    "lesson_01_features": ["two_features", "build_dataset"],
    "lesson_02_score_and_sigmoid": ["sigmoid", "add_bias", "predict_proba"],
    "lesson_03_loss": ["log_loss"],
    "lesson_04_gradient_descent": ["fit_logistic"],
    "lesson_05_regularization_and_priors": ["fit"],
    "lesson_06_class_imbalance": ["fit", "report"],
    "lesson_07_evaluation_and_thresholds": ["at_threshold", "roc_auc", "calibrate_for_tnr"],
    "lesson_08_neural_networks": ["relu", "forward"],
    "lesson_09_backprop": ["fit_mlp"],
    "lesson_10_text_hashing_ngrams": ["fnv1a_32", "featurize"],
    "lesson_11_capstone_mini_waf": ["indicator_features"],
    "lesson_12_adversarial_and_pitfalls": ["case_swap", "comment_inject", "ifs_sub"],
}

PLOTS = {
    "lesson_01_features": """import matplotlib.pyplot as plt
from lesson_01_features import build_dataset
X, y = build_dataset()
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', s=60, label='benign')
ax.scatter(X[y == 1, 0], X[y == 1, 1], marker='x', s=70, label='malicious')
ax.set_xlabel('special_char_ratio'); ax.set_ylabel('sqli_hits')
ax.set_title('The two classes in feature space'); ax.legend()
plt.show()""",
    "lesson_02_score_and_sigmoid": """import numpy as np, matplotlib.pyplot as plt
from lesson_02_score_and_sigmoid import sigmoid
z = np.linspace(-8, 8, 200)
plt.figure(figsize=(6, 3)); plt.plot(z, sigmoid(z))
plt.axhline(0.5, ls='--', c='gray'); plt.axvline(0, ls=':', c='gray')
plt.xlabel('score z = w . x'); plt.ylabel('probability'); plt.title('The sigmoid squash')
plt.show()""",
    "lesson_03_loss": """import numpy as np, matplotlib.pyplot as plt
p = np.linspace(0.01, 0.99, 200)
plt.figure(figsize=(6, 3))
plt.plot(p, -np.log(p), label='penalty when truth = 1 (attack)')
plt.plot(p, -np.log(1 - p), label='penalty when truth = 0 (benign)')
plt.xlabel("model's predicted p"); plt.ylabel('log-loss penalty')
plt.title('Confident mistakes are punished hard'); plt.legend()
plt.show()""",
    "lesson_04_gradient_descent": """import numpy as np, matplotlib.pyplot as plt
from lesson_01_features import build_dataset
from lesson_02_score_and_sigmoid import add_bias, sigmoid
from lesson_03_loss import log_loss
X, y = build_dataset(); Xb = add_bias(X)
w = np.zeros(Xb.shape[1]); w[-1] = -0.5; hist = []
for i in range(400):
    p = sigmoid(Xb @ w); hist.append(log_loss(y, p))
    w -= (0.5 / (1 + i / 150)) * (Xb.T @ (p - y) / len(y))
plt.figure(figsize=(6, 3)); plt.plot(hist)
plt.xlabel('gradient-descent step'); plt.ylabel('log-loss')
plt.title('Watch the loss fall as it learns'); plt.show()""",
    "lesson_07_evaluation_and_thresholds": """import numpy as np, matplotlib.pyplot as plt
from lesson_02_score_and_sigmoid import sigmoid
from lesson_06_class_imbalance import fit
from lesson_07_evaluation_and_thresholds import make_noisy, split, at_threshold
X, y = make_noisy(); Xtr, ytr, Xva, yva = split(X, y)
w = fit(Xtr, ytr, balance=True, fp_weight=1.0); p = sigmoid(Xva @ w)
ts = np.linspace(0.35, 0.75, 80)
rec = [at_threshold(yva, p, t)['recall'] for t in ts]
tnr = [at_threshold(yva, p, t)['tnr'] for t in ts]
plt.figure(figsize=(6, 3))
plt.plot(ts, rec, label='recall (attacks caught)')
plt.plot(ts, tnr, label='TNR (benign spared)')
plt.xlabel('threshold'); plt.title('The threshold is a dial'); plt.legend(); plt.show()""",
    "lesson_08_neural_networks": """import matplotlib.pyplot as plt
from lesson_08_neural_networks import make_xor
X, y = make_xor()
plt.figure(figsize=(5, 4))
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', s=45, label='benign (both agree)')
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='x', s=55, label='attack (exactly one high)')
plt.xlabel('feature A'); plt.ylabel('feature B')
plt.title('XOR: no straight line can separate these'); plt.legend(); plt.show()""",
    "lesson_09_backprop": """import numpy as np, matplotlib.pyplot as plt
from lesson_08_neural_networks import make_xor, forward, relu, accuracy
X, y = make_xor()
rng = np.random.default_rng(0); n, d = X.shape; hid = 8
w1 = rng.normal(0, 0.5, (hid, d)); b1 = np.zeros(hid)
w2 = rng.normal(0, 0.5, (1, hid)); b2 = np.array([-0.5])
accs = []; idx = np.arange(n)
for ep in range(120):
    rng.shuffle(idx)
    for s in range(0, n, 64):
        sl = idx[s:s + 64]; xb = X[sl]; yb = y[sl]
        z1 = xb @ w1.T + b1; h1 = relu(z1); p = 1 / (1 + np.exp(-(h1 @ w2.T + b2).ravel()))
        err = ((p - yb) / max(len(sl), 1))[:, None]
        gw2 = err.T @ h1; gb2 = err.sum(0)
        dz1 = (err @ w2) * (z1 > 0); gw1 = dz1.T @ xb; gb1 = dz1.sum(0)
        w1 -= 0.1 * gw1; b1 -= 0.1 * gb1; w2 -= 0.1 * gw2; b2 -= 0.1 * gb2
    accs.append(accuracy(y, forward(X, w1, b1, w2, b2)))
plt.figure(figsize=(6, 3)); plt.plot(accs)
plt.xlabel('epoch'); plt.ylabel('accuracy')
plt.title('Backprop teaches the net XOR from scratch'); plt.show()""",
}


def docstring_of(path: pathlib.Path) -> str:
    return ast.get_docstring(ast.parse(path.read_text())) or ""


def exercises_of(path: pathlib.Path) -> str:
    lines = path.read_text().splitlines()
    out, grabbing = [], False
    for ln in lines:
        if "EXERCISES" in ln and ln.lstrip().startswith("#"):
            grabbing = True
            continue
        if grabbing:
            if not ln.lstrip().startswith("#"):
                break
            body = ln.lstrip()[1:]           # drop leading '#'
            if set(body.strip()) <= {"-"} and body.strip():
                continue                     # skip dashed separator lines
            out.append(body[1:] if body.startswith(" ") else body)
    return "\n".join(out).strip()


def build(module: str, idx: int) -> nbf.NotebookNode:
    path = LESSONS / f"{module}.py"
    doc = docstring_of(path)
    ex = exercises_of(path)
    title = doc.splitlines()[0].strip() if doc else module

    cells = []
    # 1. Title + full guru intro (the module docstring renders as markdown).
    cells.append(new_markdown_cell(f"# {title}\n\n{doc}"))

    # 2. Setup — find lessons/ and put it on the path (works from any cwd).
    cells.append(new_markdown_cell(
        "## Setup\n\nThese notebooks *import* the real code from `lessons/*.py` "
        "— one source of truth, no forked copies. This cell finds that folder."))
    cells.append(new_code_cell(
        "import sys, pathlib\n"
        "for base in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:\n"
        "    if (base / 'lessons' / '_util.py').exists():\n"
        "        sys.path.insert(0, str(base / 'lessons'))\n"
        "        print('lessons on path:', base / 'lessons'); break\n"
        "else:\n"
        "    raise RuntimeError('Open this notebook inside the xorguard-ml-course folder.')"))

    # 3. Show the key NEW code, live from source.
    funcs = SHOW.get(module, [])
    if funcs:
        cells.append(new_markdown_cell(
            "## The key code (shown live from the source file)\n\n"
            "Read it, then copy any function into a new cell and tweak it — the "
            "originals live in `lessons/" + module + ".py`."))
        imp = ", ".join(funcs)
        cells.append(new_code_cell(
            f"import inspect\nfrom {module} import {imp}\n"
            f"for fn in ({imp}{',' if len(funcs) == 1 else ''}):\n"
            "    print(inspect.getsource(fn)); print('-' * 72)"))

    # 4. Optional real plot.
    if module in PLOTS:
        cells.append(new_markdown_cell(
            "## A picture\n\nA real plot (matplotlib) in place of the terminal's "
            "ASCII. Re-run after changing inputs to build intuition."))
        cells.append(new_code_cell(PLOTS[module]))

    # 5. Run the full narrated lesson.
    cells.append(new_markdown_cell(
        "## Run the full narrated lesson\n\nThis executes the same walkthrough as "
        "`python lessons/" + module + ".py`, printing each step with its result."))
    cells.append(new_code_cell(f"from {module} import main\nmain()"))

    # 6. Exercises.
    if ex:
        cells.append(new_markdown_cell(
            "## Exercises\n\nTry these in new cells below. Worked answers are in "
            f"`solutions/{module.split('_', 2)[0]}_{module.split('_')[1]}_solution.py`.\n\n"
            "```\n" + ex + "\n```"))
        cells.append(new_code_cell("# your work here\n"))

    nb = new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    return nb


def main() -> None:
    modules = sorted(SHOW.keys())
    for i, module in enumerate(modules, 1):
        nb = build(module, i)
        dest = OUT / f"{module}.ipynb"
        nbf.write(nb, dest)
        print(f"wrote {dest.relative_to(ROOT)}")
    print(f"\n{len(modules)} notebooks written to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
