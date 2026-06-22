# Authoring conventions

The participant-facing deliverable is a **single Jupyter notebook** (three chapters).
We author it as **paired Markdown + Python files** so content and code can be edited
and tested cleanly, then assembled into the notebook.

For each chapter `N`:

- `content/0N_*.md` holds **every markdown cell** (prose, background, task statements).
- `scripts/0N_*.py` holds **every code cell** (the *solution* code we test).

## Cell mapping

Cells are numbered `N.k` where `N` is the chapter and `k` is a strictly increasing
sequence index **shared across both files**. The number alone fixes the order in the
final notebook; the type tag says which file it comes from.

**In the markdown file**, each cell starts with a header line:

```
<!-- CELL N.k | markdown -->
<!-- CELL N.k | code -> scripts/0N_xxx.py -->
```

A `markdown` cell's body is the prose that follows. A `code` cell entry in the
markdown file contains only the **task description** (what the participant must
implement); the runnable solution lives in the script under the same `N.k`.

**In the python file**, each code cell is delimited with a `# %%` marker (Jupytext
"percent" style) carrying the same id:

```python
# %% CELL N.k | code
<solution code>
```

## Assembling the notebook

Run `python dev/build_notebook.py`. It walks cells in ascending `N.k` and, for each,
takes markdown prose from the `.md` and code from the `.py`. Because ids are shared and
ordered, assembly is mechanical. It writes **two** notebooks at the repo root:

- `BISCCITS_SNN_Workshop.ipynb` — full solutions (the answer key).
- `BISCCITS_SNN_Workshop_participant.ipynb` — the participant deliverable, with task
  cells blanked (see below).

Both are committed so they can be diffed; the participant one is what gets handed out.

## Participant vs solution

The scripts contain full solutions so we can test runnability. A code cell is a **task**
when its `# %% CELL N.k` marker line contains `# TASK`. Within a task cell, wrap the
lines the participant should write in solution markers:

```python
# >>> SOLUTION hint="convert tau to beta = exp(-dt / tau), then simulate"
b = np.exp(-dt / tau)
m, s = lif_simulate(step, beta=b)
# <<< SOLUTION
```

- **Solution notebook**: the marker lines are stripped, the code is kept.
- **Participant notebook**: each wrapped region is replaced by `# TODO: <hint>` (only if
  a `hint="..."` is given) followed by `# YOUR CODE HERE`, indented to match the region
  so it sits correctly inside a `def`/`for`/`class`. Everything outside the markers —
  signatures, docstrings, plotting, comparison code — is kept verbatim, which is how a
  complex task still ships with guiding structure.

Non-task code cells (imports, plotting, data loading) are emitted unchanged. A task cell
with no markers would be blanked entirely (it has no kept regions), so always wrap the
solution lines. Markers are plain comments, so the scripts stay runnable and testable.
