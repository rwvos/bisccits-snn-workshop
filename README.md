# bisccits-snn-workshop

A hands-on workshop on **Spiking Neural Networks (SNNs)** — from the equations that
define them, to training them with surrogate gradients, to measuring why they are
computationally efficient at inference. Built for the **BISCCITS** conference.

The workshop is delivered as a single guided **Jupyter notebook** that runs on
**Google Colab** (no local setup required for participants).

> **Audience.** Researchers in biology, bio-inspired computing, and neurobiology —
> *not* assumed to have a machine-learning background. The material keeps a strong
> bio-inspired framing and introduces the ML tooling (autograd, backpropagation,
> PyTorch) from the ground up.

## The three chapters

Each chapter is concise, has 2–3 interactive subtasks, and visualizes both the
**neuron/network dynamics** and **what the model is computing**.

1. **Defining a spiking network** — the leaky integrate-and-fire (LIF) neuron from its
   equations; membrane as a low-pass filter, threshold, hard reset; and the
   *surrogate gradient* (derivative of a sigmoid, with a slope sweep) that makes
   training possible.
2. **Training a spiking network** — stack LIF neurons into a 3-layer SNN; train it on
   the **RacketSports** sensor dataset with surrogate-gradient backprop-through-time;
   then make training fast with **forward-gradient injection** + `torch.compile`.
   Benchmarked against a matched **MLP** and **GRU** (accuracy + wall-clock time).
3. **Evaluating a spiking network** — spike rasters, activity sparsity, and the
   inference-cost story: **multiply-accumulate (MAC)** vs **accumulate (AC)**
   operations, and how the SNN compares with the ANN and the RNN.

## Dataset

[**RacketSports**](https://zenodo.org/records/3742271) (UEA multivariate time-series
archive): smartwatch accelerometer + gyroscope (6 channels), 30 timesteps, 4 classes
(badminton/squash strokes), 151 train / 152 test trials. Downloaded automatically via
[`aeon`](https://www.aeon-toolkit.org/).

## Repository layout

This repo is the **development workspace**. Per the authoring workflow (see
[CONVENTIONS.md](CONVENTIONS.md)), content lives as paired Markdown + runnable Python
so it can be reviewed and tested before being assembled into the participant notebook.

```
.
├── content/            # one markdown file per chapter = the notebook's markdown cells
│   ├── 01_defining_snns.md
│   ├── 02_training_snns.md
│   └── 03_evaluating_snns.md
├── scripts/            # one python file per chapter = the notebook's code cells (solutions)
│   ├── 01_defining_snns.py
│   ├── 02_training_snns.py
│   └── 03_evaluating_snns.py
├── snn_workshop/       # shared plumbing (NOT participant tasks): data, plotting, utils, reference models
├── dev/                # dev-only helper to run scripts headlessly and dump figures
├── CONVENTIONS.md      # the markdown <-> script cell-mapping rules
├── requirements.txt
└── README.md
```

Each markdown cell and code cell carries a shared id (`N.k`) so the two files map
cleanly onto one consistent notebook. See [CONVENTIONS.md](CONVENTIONS.md).

## Getting started (development)

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD          # so `import snn_workshop` resolves

python scripts/01_defining_snns.py
python scripts/02_training_snns.py     # trains the SNN, saves a checkpoint
python scripts/03_evaluating_snns.py   # loads the checkpoint, evaluates

# or run headlessly and save figures to figures/
python dev/run_script.py scripts/01_defining_snns.py
```

> **Note on `torch.compile`.** The compile speed-up in Chapter 2 needs an
> Inductor/Triton backend (present on the Colab GPU runtime). On platforms without it
> (e.g. Windows) the code falls back to eager mode automatically.

## License

Copyright 2026 Reinier Vos & Stein Stroobants