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

Walk cells in ascending `N.k`. For each: if `markdown`, take the prose from the `.md`;
if `code`, take the solution from the `.py` (blanked / partially blanked for the
participant version). Because ids are shared and ordered, assembly is mechanical.

## Participant vs solution

The scripts contain full solutions so we can test runnability. When generating the
participant notebook, code cells tagged as **tasks** (marked `# TASK` in the script)
are reduced to a stub + `# YOUR CODE HERE`; non-task code cells (imports, plotting,
data loading) are kept as-is.
