<!-- CELL 2.0 | markdown -->
# Chapter 2 — Training a Spiking Neural Network

In Chapter 1 we built a single LIF neuron and met the surrogate gradient. Now we will:

1. stack LIF neurons into a **deep spiking network**,
2. train it on a real movement-sensor dataset with **backpropagation through time** using
   the surrogate gradient, and
3. make training **fast** with *forward-gradient injection* + `torch.compile`,
   benchmarking against a conventional MLP and a GRU.

> **Objective.** Train a 3-layer SNN end-to-end, and understand *why* the spike's
> non-differentiability is not an obstacle — and how to keep training efficient.

<!-- CELL 2.1 | markdown -->
## From one neuron to a deep network

A **layer** of LIF neurons is just many neurons in parallel. Each neuron receives a
weighted sum of the layer's inputs as its input current:

$$I^{(\ell)}[t] = W^{(\ell)} \, s^{(\ell-1)}[t], \qquad
  V^{(\ell)}[t] = \beta\,V^{(\ell)}[t-1] + I^{(\ell)}[t],$$

then thresholds and resets exactly as before, emitting a spike vector
$s^{(\ell)}[t]$. We stack three such layers. The network is **feedforward in space**
(layer $\ell$ feeds layer $\ell+1$) but **recurrent in time**: each neuron's membrane
carries state from one timestep to the next. *This temporal recurrence is the
"recurrent dynamics" of the SNN* — there are no explicit lateral weights here.

Our input is a length-$T$ multivariate time series. At every timestep we feed one
sample $x[t]$ into the first layer. After the last spiking layer we apply a linear
**readout** at each timestep and **average the logits over time**; the time-averaged
logits go into a standard **cross-entropy** classification loss.

> Why average over time? Each timestep produces a noisy, spike-driven vote for each
> class. Averaging integrates evidence across the whole sequence into one prediction.

<!-- CELL 2.2 | code -> scripts/02_training_snns.py -->
**Setup.** Imports and device selection (GPU if available — useful for the compile
demo later).

<!-- CELL 2.3 | markdown -->
## Subtask 1 — The spike with a surrogate gradient

Recall the problem: the spike is a Heaviside step whose derivative is zero, so no
gradient flows. The classic fix in PyTorch is a **custom `autograd.Function`**: we
define the forward pass (the hard spike) and *override* the backward pass to use the
smooth surrogate from Chapter 1 (the derivative of a sigmoid).

This is the most explicit way to write it, and it makes the "different forward vs
backward" idea concrete. Its one drawback — which motivates Subtask 3 — is that a
custom `autograd.Function` cannot be traced by `torch.compile`.

<!-- CELL 2.4 | code -> scripts/02_training_snns.py -->
**TASK.** Implement `SpikeFunction(torch.autograd.Function)`: `forward` returns
`(x >= 0).float()`; `backward` multiplies the incoming gradient by the
sigmoid-derivative surrogate `slope * σ(slope·x) · (1 − σ(slope·x))`. Wrap it in a
helper `spike_autograd(x, slope)`.

<!-- CELL 2.5 | markdown -->
## Subtask 2 — The LIF layer and the deep SNN

Now we assemble the network. `LIFLayer` wraps a `nn.Linear` (the weights $W$) and
unrolls the LIF recurrence over time, calling our spike function each step and
applying the hard reset. `DeepSNN` stacks several `LIFLayer`s and adds the
time-averaged linear readout.

<!-- CELL 2.6 | code -> scripts/02_training_snns.py -->
**TASK.** Implement `LIFLayer` (loop over time: integrate `v = beta*v + current`,
spike, hard-reset `v = v*(1-s)`, collect spikes) and `DeepSNN` (stack layers, then
`readout(spikes).mean(dim=1)`). Expose a `return_spikes`/`return_mem` flag — we reuse
it for the visualisations in Chapter 3.

<!-- CELL 2.7 | markdown -->
## The dataset — RacketSports

We use **RacketSports** from the UEA multivariate time-series archive. University
students played **badminton** or **squash** while wearing a **smartwatch**; the watch
streamed its accelerometer and gyroscope.

| property | value |
|---|---|
| channels (C) | **6** — 3-axis accelerometer + 3-axis gyroscope |
| timesteps (T) | **30** — sampled at 10 Hz over ~3 seconds |
| classes | **4** — Badminton Clear, Badminton Smash, Squash Forehand Boast, Squash Backhand Boast |
| train / test | **151 / 152** trials |

Each trial is one stroke; the task is to identify the sport **and** the stroke. It is
small (fast to train) yet genuinely temporal — a good fit for spiking models. We
z-score each channel using training-set statistics, feed the 6 channels as the input
current at each of the 30 timesteps, and read out 4 class logits.

<!-- CELL 2.8 | code -> scripts/02_training_snns.py -->
**Load the data** with the helper (`aeon` downloads it automatically) and move the
tensors to the device.

<!-- CELL 2.9 | markdown -->
## Training

`train_model` is a standard mini-batch Adam loop with cross-entropy loss. The only
thing that makes this "spiking" is the model — backpropagation through time and the
optimiser are exactly what you would use for any recurrent network, because the
surrogate gradient lets gradients flow through the spikes. We record **train
accuracy**, **test accuracy** (the dataset ships with its own split), and **wall-clock
training time**.

<!-- CELL 2.10 | code -> scripts/02_training_snns.py -->
*(No task — train the SNN built from your custom-autograd spike and record its
accuracy and training time.)*

<!-- CELL 2.11 | markdown -->
## Baselines — MLP and GRU

To judge the SNN we train two conventional networks of matched depth/width (3 layers,
64 units):

- **MLP** — flattens the time axis into one long feature vector and classifies it in a
  single forward pass. Simple and fast, but it is *not* a sequence model: it sees the
  whole trial at once and ignores temporal order.
- **GRU** — a continuous-valued **recurrent** network, the natural non-spiking
  counterpart of our SNN (it too processes the sequence step by step). This is the
  apples-to-apples baseline we will return to in Chapter 3.

<!-- CELL 2.12 | code -> scripts/02_training_snns.py -->
*(No task — train the MLP and GRU baselines.)*

<!-- CELL 2.13 | markdown -->
## Subtask 3 — Faster training: forward-gradient injection + `torch.compile`

Our custom `autograd.Function` works, but it forces a "graph break": `torch.compile`
cannot fuse the unrolled time loop, so each of the 30 steps pays Python/kernel-launch
overhead. The fix is to express the surrogate **without** a custom Function, as a
single differentiable expression — *forward-gradient injection*:

```python
spike = (x >= 0).float().detach() + (surr - surr.detach())
```

- **Forward:** `surr - surr.detach() == 0`, so the value is exactly the hard spike.
- **Backward:** the only non-detached term is `surr`, so the gradient is `d surr/dx` —
  our surrogate.

With `surr = x` this is the plain *straight-through estimator*; with
`surr = σ(slope·x)` we recover the smooth sigmoid surrogate. Because it is one ordinary
expression (no custom Function), the whole model is now traceable, and
`torch.compile` can fuse the time loop for a large speed-up.

> **Note.** `torch.compile` needs a backend (Inductor/Triton) that ships on the Colab
> GPU runtime. On some local setups (e.g. Windows) it is unavailable; the code then
> falls back to eager mode automatically. Run this on Colab to see the speed-up.

<!-- CELL 2.14 | code -> scripts/02_training_snns.py -->
**TASK.** Implement `spike_fgi(x, slope)` as the single-line forward-gradient
injection above, rebuild the SNN with it, wrap it in `torch.compile`, and train.
Compare its wall-clock time and accuracy with the custom-autograd version — accuracy
should match (same surrogate), but the compiled model trains faster.

<!-- CELL 2.15 | markdown -->
## Results & visualizations

Each `train_model` call returned a **history** (per-epoch train/test loss and
accuracy) and a training-only **wall-clock time**. We now read those out as a summary
table and three plots. Headlines to expect:

- The **SNN reaches accuracy competitive with the MLP**, a little below the GRU — a
  good result for a spiking network on a small dataset.
- **The two SNN variants (custom-autograd vs forward-gradient injection) reach the
  same accuracy** — they use the identical surrogate. Any small difference is just
  random fluctuation from seeding. They differ only in *runtime*.
- On Colab, the **compiled** FGI SNN trains markedly faster than the eager one.

We also save the trained SNN to `checkpoints/` — Chapter 3 loads it.

<!-- CELL 2.16 | code -> scripts/02_training_snns.py -->
*(No task — print the summary table and save the checkpoint.)*

<!-- CELL 2.17 | markdown -->
### Training curves

For each model type, plot loss and accuracy **against epoch**, with the **training**
split solid and the **test** split dashed. These show *how* learning progresses: the
loss falling, the accuracy rising, and the gap between train and test (a read on
over-fitting — expected here, since the dataset is small).

<!-- CELL 2.18 | code -> scripts/02_training_snns.py -->
*(No task — plot the train/test loss and accuracy curves for the SNN, MLP and GRU.)*

<!-- CELL 2.19 | markdown -->
### Training time

Training time is a **single number per run** — the total wall-clock seconds for the
fixed number of epochs. A bar plot makes the comparison clear. Note the two SNN
variants: custom-autograd vs forward-gradient injection (and, on Colab, the compiled
variant), which is where the FGI + `torch.compile` speed-up shows up. The MLP and GRU
are fast because their layers are single fused ops; the eager SNN pays for its
Python-level unrolled time loop — exactly what compilation removes.

<!-- CELL 2.20 | code -> scripts/02_training_snns.py -->
*(No task — bar plot of per-run training time.)*

<!-- CELL 2.21 | markdown -->
### Confusion matrices

Accuracy is one number; a **confusion matrix** shows *which* classes get confused. We
plot one per model type (MLP, SNN, GRU), normalized per true class (rows sum to 1), so
the diagonal is the per-class recall. Look for which strokes are hardest — e.g.
badminton vs squash should separate easily, while the two strokes *within* a sport may
be confused.

<!-- CELL 2.22 | code -> scripts/02_training_snns.py -->
*(No task — compute and plot the three confusion matrices on the test set.)*
