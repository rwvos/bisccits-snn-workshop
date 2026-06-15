"""Assemble the paired content/*.md + scripts/*.py into one Jupyter notebook.

Walks the shared `N.k` cell ids (see CONVENTIONS.md) and interleaves:
  * markdown cells  -> from content/0N_*.md
  * code cells      -> from scripts/0N_*.py  (preceded by their task description
                       from the .md, kept as a small markdown cell)

Run:  python dev/build_notebook.py
Output: BISCCITS_SNN_Workshop.ipynb at the repo root.
Dev-only; not part of the participant deliverable.
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTERS = ["01_defining_snns", "02_training_snns", "03_evaluating_snns"]
OUT = os.path.join(ROOT, "BISCCITS_SNN_Workshop.ipynb")

# Injected as the first two cells: install the workshop package + dependencies so the
# helper modules in snn_workshop/ are importable (especially on Google Colab).
SETUP_MD = """## Setup — run this first

This workshop uses helper modules from its GitHub repository (dataset loading, the
spiking-network models, and plotting utilities). The cell below **installs the
repository as a package** — along with external dependencies such as `aeon` (used to
download the dataset) — so that `import snn_workshop` works on Google Colab. PyTorch,
NumPy and Matplotlib are already available on Colab.

> Running locally from a clone instead? Use `pip install -e .` in the repo root; the
> cell below detects that `snn_workshop` is already importable and skips the install."""

SETUP_CODE = '''# Install the workshop package (and its dependencies, e.g. aeon) if needed.
import importlib.util, subprocess, sys

REPO_URL = "https://github.com/rwvos/bisccits-snn-workshop.git"  # workshop repository

if importlib.util.find_spec("snn_workshop") is None:
    print(f"Installing snn_workshop (+ aeon and other dependencies) from {REPO_URL} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", f"git+{REPO_URL}"],
        check=True,
    )
    print("Installation complete.")
else:
    print("snn_workshop already importable — skipping install.")'''

MD_MARKER = re.compile(r"^<!--\s*CELL\s+(\S+)\s*\|\s*(markdown|code)\b.*-->\s*$")
PY_MARKER = re.compile(r"^#\s*%%\s*CELL\s+(\S+)\s*\|\s*code\b.*$")


def sort_key(cell_id):
    """'2.18c' -> (2, 18, 'c'); '2.15' -> (2, 15, '')."""
    chap, rest = cell_id.split(".")
    m = re.match(r"(\d+)([a-z]*)$", rest)
    return (int(chap), int(m.group(1)), m.group(2))


def parse_markdown(path):
    """Return list of (cell_id, kind, body) in file order."""
    cells, cid, kind, buf = [], None, None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = MD_MARKER.match(line.rstrip("\n"))
            if m:
                if cid is not None:
                    cells.append((cid, kind, "".join(buf).strip("\n")))
                cid, kind, buf = m.group(1), m.group(2), []
            elif cid is not None:
                buf.append(line)
    if cid is not None:
        cells.append((cid, kind, "".join(buf).strip("\n")))
    return cells


def parse_python(path):
    """Return dict cell_id -> code body (marker line dropped)."""
    code, cid, buf = {}, None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = PY_MARKER.match(line.rstrip("\n"))
            if m:
                if cid is not None:
                    code[cid] = "".join(buf).strip("\n")
                cid, buf = m.group(1), []
            elif cid is not None:
                buf.append(line)
    if cid is not None:
        code[cid] = "".join(buf).strip("\n")
    return code


def cell(kind, text):
    src = text.splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        base.update(execution_count=None, outputs=[])
    return base


def main():
    nb_cells = [cell("markdown", SETUP_MD), cell("code", SETUP_CODE)]
    for chap in CHAPTERS:
        md = parse_markdown(os.path.join(ROOT, "content", f"{chap}.md"))
        py = parse_python(os.path.join(ROOT, "scripts", f"{chap}.py"))
        for cid, kind, body in sorted(md, key=lambda c: sort_key(c[0])):
            if kind == "markdown":
                nb_cells.append(cell("markdown", body))
            else:  # code cell: task description (markdown) then the solution code
                if body:
                    nb_cells.append(cell("markdown", body))
                code_body = py.get(cid)
                if code_body is None:
                    nb_cells.append(cell("markdown", f"> _missing code for cell {cid}_"))
                else:
                    nb_cells.append(cell("code", code_body))

    for i, c in enumerate(nb_cells):
        c["id"] = f"cell-{i:03d}"

    notebook = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    n_md = sum(c["cell_type"] == "markdown" for c in nb_cells)
    n_code = sum(c["cell_type"] == "code" for c in nb_cells)
    print(f"wrote {OUT}: {len(nb_cells)} cells ({n_md} markdown, {n_code} code)")


if __name__ == "__main__":
    main()
