"""Assemble the paired content/*.md + scripts/*.py into one Jupyter notebook.

Walks the shared `N.k` cell ids (see CONVENTIONS.md) and interleaves:
  * markdown cells  -> from content/0N_*.md
  * code cells      -> from scripts/0N_*.py  (preceded by their task description
                       from the .md, kept as a small markdown cell)

Run:  python dev/build_notebook.py
Output (both written at the repo root):
  * BISCCITS_SNN_Workshop.ipynb              -- full solutions (answer key)
  * BISCCITS_SNN_Workshop_participant.ipynb  -- task cells blanked for participants

Task cells are the code cells whose `# %% CELL N.k` marker carries `# TASK`. Inside
those cells, the solution logic is wrapped in `# >>> SOLUTION ... # <<< SOLUTION`
markers (an optional `hint="..."` on the opening marker). In the solution notebook the
markers are stripped and the code is kept; in the participant notebook each wrapped
region is replaced by a `# TODO: <hint>` + `# YOUR CODE HERE` placeholder, leaving the
surrounding scaffolding (signatures, docstrings, plotting) intact. See CONVENTIONS.md.

Dev-only; not part of the participant deliverable.
"""

import base64
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTENT = os.path.join(ROOT, "content")
CHAPTERS = ["01_defining_snns", "02_training_snns", "03_evaluating_snns",
            "04_bonus_fgi"]
OUT = os.path.join(ROOT, "BISCCITS_SNN_Workshop.ipynb")
OUT_PARTICIPANT = os.path.join(ROOT, "BISCCITS_SNN_Workshop_participant.ipynb")

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
REPO_REF = "main"   # branch, tag, or commit to install (e.g. "main", "dev", "v0.1.0")

if importlib.util.find_spec("snn_workshop") is None:
    spec = f"git+{REPO_URL}@{REPO_REF}"
    print(f"Installing snn_workshop (+ aeon and other dependencies) from {spec} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", spec],
        check=True,
    )
    print("Installation complete.")
else:
    print("snn_workshop already importable — skipping install.")'''

MD_MARKER = re.compile(r"^<!--\s*CELL\s+(\S+)\s*\|\s*(markdown|code)\b.*-->\s*$")
PY_MARKER = re.compile(r"^#\s*%%\s*CELL\s+(\S+)\s*\|\s*code\b(.*)$")

# Solution-region markers inside a task cell. The opening marker may carry a hint:
#   # >>> SOLUTION hint="convert tau to beta = exp(-dt / tau)"
#   <solution lines>
#   # <<< SOLUTION
SOL_OPEN = re.compile(r'^(\s*)#\s*>>>\s*SOLUTION(?:\s+hint="([^"]*)")?\s*$')
SOL_CLOSE = re.compile(r"^\s*#\s*<<<\s*SOLUTION\s*$")

# Markdown image syntax: ![alt](path "optional-title"). We inline any image that
# resolves inside content/ as a base64 data URI so the notebook is self-contained on
# Colab (external URLs are left untouched). An optional title of the form
# `"width=480"` sets the rendered width.
IMG_MARKER = re.compile(r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"([^"]*)")?\)')
IMG_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp"}


def embed_images(text):
    """Replace markdown image refs to files in content/ with base64 <img> tags."""
    def repl(m):
        alt, rel, title = m.group(1), m.group(2), m.group(3)
        path = os.path.normpath(os.path.join(CONTENT, rel))
        if not os.path.isfile(path):
            return m.group(0)  # external URL or missing file: leave as-is
        mime = IMG_MIME.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        width = ""
        if title and title.startswith("width="):
            width = f' width="{title.split("=", 1)[1]}"'
        return (f'<img alt="{alt}"{width} style="max-width:100%;height:auto;" '
                f'src="data:{mime};base64,{b64}">')
    return IMG_MARKER.sub(repl, text)


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
    """Return (code, tasks): dict cell_id -> code body, and set of task cell ids.

    A cell is a *task* when its `# %% CELL N.k | code` marker line contains `TASK`.
    """
    code, tasks, cid, buf = {}, set(), None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = PY_MARKER.match(line.rstrip("\n"))
            if m:
                if cid is not None:
                    code[cid] = "".join(buf).strip("\n")
                cid, buf = m.group(1), []
                if "TASK" in m.group(2):
                    tasks.add(cid)
            elif cid is not None:
                buf.append(line)
    if cid is not None:
        code[cid] = "".join(buf).strip("\n")
    return code, tasks


def render_code(body, is_task, participant):
    """Resolve `# >>> SOLUTION ... # <<< SOLUTION` regions in a code cell.

    Solution notebook (participant=False): drop the marker lines, keep the code.
    Participant notebook on a task cell: replace each region with a placeholder
    (`# TODO: <hint>` if a hint is given, then `# YOUR CODE HERE`), indented to match
    the first non-blank line of the region so it sits correctly inside a def/for/class.
    Non-task cells are never blanked (their markers, if any, are simply stripped).
    """
    lines = body.split("\n")
    out, i, n = [], 0, len(lines)
    while i < n:
        m = SOL_OPEN.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        marker_indent, hint = m.group(1), m.group(2)
        i += 1
        region = []
        while i < n and not SOL_CLOSE.match(lines[i]):
            region.append(lines[i])
            i += 1
        i += 1  # skip the closing marker (if present)
        if participant and is_task:
            indent = marker_indent
            for ln in region:
                if ln.strip():
                    indent = ln[: len(ln) - len(ln.lstrip())]
                    break
            if hint:
                out.append(f"{indent}# TODO: {hint}")
            out.append(f"{indent}# YOUR CODE HERE")
        else:
            out.extend(region)
    return "\n".join(out).strip("\n")


def cell(kind, text):
    if kind == "markdown":
        text = embed_images(text)
    src = text.splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        base.update(execution_count=None, outputs=[])
    return base


def build_cells(participant):
    """Assemble the notebook cell list. `participant` toggles task-cell blanking."""
    nb_cells = [cell("markdown", SETUP_MD), cell("code", SETUP_CODE)]
    for chap in CHAPTERS:
        md = parse_markdown(os.path.join(ROOT, "content", f"{chap}.md"))
        py, tasks = parse_python(os.path.join(ROOT, "scripts", f"{chap}.py"))
        for cid, kind, body in sorted(md, key=lambda c: sort_key(c[0])):
            if kind == "markdown":
                nb_cells.append(cell("markdown", body))
            else:  # code cell: task description (markdown) then the code
                if body:
                    nb_cells.append(cell("markdown", body))
                code_body = py.get(cid)
                if code_body is None:
                    nb_cells.append(cell("markdown", f"> _missing code for cell {cid}_"))
                else:
                    rendered = render_code(code_body, cid in tasks, participant)
                    nb_cells.append(cell("code", rendered))

    for i, c in enumerate(nb_cells):
        c["id"] = f"cell-{i:03d}"
    return nb_cells


def write_notebook(path, nb_cells):
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    n_md = sum(c["cell_type"] == "markdown" for c in nb_cells)
    n_code = sum(c["cell_type"] == "code" for c in nb_cells)
    print(f"wrote {path}: {len(nb_cells)} cells ({n_md} markdown, {n_code} code)")


def main():
    write_notebook(OUT, build_cells(participant=False))
    write_notebook(OUT_PARTICIPANT, build_cells(participant=True))


if __name__ == "__main__":
    main()
