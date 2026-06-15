"""Dev-only: run a chapter script headlessly and dump each figure to figures/.

Usage:  python dev/run_script.py scripts/01_defining_snns.py
Not part of the participant deliverable.
"""
import os
import sys
import runpy

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

script = sys.argv[1]
out_dir = os.path.join(ROOT, "figures", os.path.splitext(os.path.basename(script))[0])
os.makedirs(out_dir, exist_ok=True)

_counter = {"n": 0}
_orig_show = plt.show


def _capture_show(*args, **kwargs):
    fig = plt.gcf()
    if fig.get_axes():
        _counter["n"] += 1
        path = os.path.join(out_dir, f"fig_{_counter['n']:02d}.png")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        print(f"[saved] {path}")
    plt.close("all")


plt.show = _capture_show
runpy.run_path(os.path.join(ROOT, script), run_name="__main__")
print(f"[done] {script}: {_counter['n']} figure(s)")
