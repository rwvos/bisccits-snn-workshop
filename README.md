# bisccits-snn-workshop

A hands-on workshop on spiking neural networks (SNNs), written for BISCCITS 2026 in Wageningen, The Netherlands. You start from the equations that define a spiking neuron, train a small network with surrogate gradients, and then measure why spiking models can be cheap to run at inference time.

It's a single Jupyter notebook meant to run on Google Colab, so participants don't need
to install anything locally.

The workshop is aimed at researchers in biology, bio-inspired computing, and
neurobiology. We don't assume a machine-learning background: the bio-inspired framing
stays front and center, and the ML tooling (autograd, backpropagation, PyTorch) is
introduced as we go.

## What you'll work through

Three chapters, each with a couple of interactive subtasks. Throughout, we plot
both the network dynamics and what the model is actually computing.

1. **Defining a spiking network.** The leaky integrate-and-fire (LIF) neuron, built up
   from its equations: the membrane as a low-pass filter, the firing threshold, and the
   hard reset. Then the surrogate gradient — a sigmoid derivative whose slope you can
   sweep — which is the trick that makes a spiking network trainable.
2. **Training a spiking network.** Stack LIF neurons into a 3-layer SNN and train it on
   the RacketSports sensor dataset using surrogate-gradient backprop-through-time. Then
   speed training up with forward-gradient injection and `torch.compile`, and compare
   against a matched MLP and GRU on both accuracy and wall-clock time.
3. **Evaluating a spiking network.** Spike rasters, activity sparsity, and the cost of
   inference: multiply-accumulate (MAC) vs. accumulate (AC) operations, and how the SNN
   stacks up against the ANN and the RNN.

Ready to start? Open the notebook in Colab and work straight through — everything below
is for people setting up or developing the workshop, and you don't need it.

[![Participant notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rwvos/bisccits-snn-workshop/blob/main/BISCCITS_SNN_Workshop_participant.ipynb)
&nbsp;participant notebook (start here)

[![Solutions notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rwvos/bisccits-snn-workshop/blob/main/BISCCITS_SNN_Workshop.ipynb)
&nbsp;solutions notebook (filled-in code)

## The dataset

We use [RacketSports](https://zenodo.org/records/3742271) from the UEA
multivariate time-series archive: smartwatch accelerometer and gyroscope readings
(6 channels, 30 timesteps) labelled with one of 4 badminton/squash strokes, split into
151 training and 152 test trials. It downloads automatically through
[`aeon`](https://www.aeon-toolkit.org/), so there's nothing to fetch by hand.

## Development

### How this repo is organized

This is the development workspace, not the notebook participants will use. We keep the
content as paired Markdown and runnable Python — one of each per chapter — so every
cell can be reviewed and actually run before it's assembled into the final notebook.
[CONVENTIONS.md](CONVENTIONS.md) explains the cell-mapping rules; the short version is
that each markdown cell and code cell shares an id (`N.k`) so the two files line up into
one consistent notebook.

```
.
├── content/            # one markdown file per chapter → the notebook's markdown cells
│   ├── 01_defining_snns.md
│   ├── 02_training_snns.md
│   └── 03_evaluating_snns.md
├── scripts/            # one python file per chapter → the notebook's code cells (solutions)
│   ├── 01_defining_snns.py
│   ├── 02_training_snns.py
│   └── 03_evaluating_snns.py
├── snn_workshop/       # shared plumbing (data, plotting, utils, reference models) — not participant tasks
├── dev/                # helper to run scripts headlessly and dump figures
├── CONVENTIONS.md      # the markdown ↔ script cell-mapping rules
├── requirements.txt
└── README.md
```

### Running it locally

If you're developing or just want to run the chapters outside Colab:

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD          # so `import snn_workshop` resolves

python scripts/01_defining_snns.py
python scripts/02_training_snns.py     # trains the SNN, saves a checkpoint
python scripts/03_evaluating_snns.py   # loads the checkpoint, evaluates

# or run headlessly and save the figures to figures/
python dev/run_script.py scripts/01_defining_snns.py
```

A note on `torch.compile`: the Chapter 2 speed-up needs an Inductor/Triton backend,
which the Colab GPU runtime has. Where it's missing (Windows, for instance) the code
quietly falls back to eager mode, so things still run — just without the compile boost.

## License

Released under the [MIT License](LICENSE).

Copyright © 2026 Reinier Vos & Stein Stroobants
