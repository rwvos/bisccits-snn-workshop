# bisccits-snn-workshop

A hands-on workshop on **Spiking Neural Networks (SNNs)** — from the equations that
define them, to training them with surrogate gradients, to measuring why they are
computationally efficient at inference.

The workshop is delivered as a single guided **Jupyter notebook** that runs on
**Google Colab** (no local setup required for participants).

> **Audience.** Researchers in biology, bio-inspired computing, and neurobiology —
> *not* assumed to have a machine-learning background. The material keeps a strong
> bio-inspired framing and introduces the ML tooling (autograd, backpropagation,
> PyTorch) from the ground up.

## What you will learn

The notebook is organized into three concise chapters, each with a few interactive
subtasks and visualizations of both the **neuron dynamics** and **what the model is
actually computing**.

1. **Defining a spiking network.** The governing equations (leaky integrate-and-fire
   and friends) and how a spiking neuron / recurrent spiking layer is expressed in
   PyTorch. Visual contrast between single-neuron LIF dynamics and the recurrent
   network dynamics participants build.
2. **Training a spiking network.** Why the spike is non-differentiable, and how
   **surrogate gradients** make backpropagation-through-time work. We also look at
   **forward-mode gradients** and `torch.compile` to train faster.
3. **Recovering and evaluating the trained network.** Measuring the efficiency gain
   of an SNN versus an ANN and a continuous-valued RNN, and discussing the operation
   profile at inference — accumulate-only (SNN) versus multiply-accumulate (ANN) —
   and how that scales.

## Repository layout

This repo is the **development workspace** for the workshop. Content is authored and
tested as Markdown + runnable Python scripts first; the participant-facing notebook is
assembled from that verified material.

```
.
├── README.md
└── (content + scripts to be added)
```

> Layout is a work in progress and will be filled in as the material is built.

## Getting started (development)

```bash
git clone <this-repo-url>
cd bisccits-snn-workshop
# environment / dependency setup to be added
```

## License

TBD.
