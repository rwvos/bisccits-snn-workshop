# %% [chapter 3] Evaluating Spiking Neural Networks -- solution script
# Paired with content/03_evaluating_snns.md (see CONVENTIONS.md for the cell mapping).
#
# In the assembled notebook this chapter reuses the trained `snn` model and the
# classes/data already defined in Chapter 2 (single kernel). To keep this script
# runnable on its own we import the reference model and reload the saved checkpoint.

# %% CELL 3.2 | code  (setup -- not a task)
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from snn_workshop import set_seed, get_device
from snn_workshop.data import load_racket_sports
from snn_workshop.models import DeepSNN, spike_autograd
from snn_workshop.viz import plot_spike_raster, plot_firing_rates

set_seed(0)
DEVICE = get_device()

ds = load_racket_sports(normalize=True)
X_test = torch.tensor(ds.X_test, device=DEVICE)
y_test = torch.tensor(ds.y_test, device=DEVICE)

# Rebuild the SNN and load the weights trained in Chapter 2.
ckpt_path = "checkpoints/snn_racketsports.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    snn = DeepSNN(spike_fn=spike_autograd, **ckpt["config"]).to(DEVICE)
    snn.load_state_dict(ckpt["state_dict"])
    print("loaded trained SNN from", ckpt_path)
else:
    print("WARNING: no checkpoint found -- run Chapter 2 first. Using an untrained SNN.")
    snn = DeepSNN(ds.n_channels, hidden=64, n_layers=3, n_classes=ds.n_classes,
                  spike_fn=spike_autograd).to(DEVICE)
snn.eval()


# %% CELL 3.4 | code  # TASK: spike raster for a single input
# Run one test trial through the SNN and visualise *when* each hidden neuron fires.
# Pick the first correctly-classified test trial so the raster matches its label.

with torch.no_grad():
    preds_all = snn(X_test).argmax(dim=1)
correct = (preds_all == y_test).nonzero(as_tuple=True)[0]
sample_idx = int(correct[0]) if len(correct) else 0

x = X_test[sample_idx:sample_idx + 1]
with torch.no_grad():
    logits, per_layer = snn(x, return_spikes=True)
pred = logits.argmax(dim=1).item()
print(f"sample {sample_idx}: true={ds.class_names[y_test[sample_idx]]}, "
      f"pred={ds.class_names[pred]}")

# >>> SOLUTION hint="one raster per hidden layer: for each layer's spikes, take spikes[0].cpu().numpy() (shape T x hidden) and call plot_spike_raster(sp, ax=...). A subplot row per layer, sharing the time axis, works well."
fig, axes = plt.subplots(len(per_layer), 1, figsize=(9, 7), sharex=True)
for l, spikes in enumerate(per_layer):
    sp = spikes[0].cpu().numpy()          # (T, hidden)
    plot_spike_raster(sp, title=f"Hidden layer {l + 1} spikes", ax=axes[l])
axes[-1].set_xlabel("timestep")
fig.suptitle(f"Spike raster — true: {ds.class_names[y_test[sample_idx]]}")
fig.tight_layout()
plt.show()
# <<< SOLUTION


# %% CELL 3.6 | code  # TASK: how sparse is the network? (firing rates over the test set)
# >>> SOLUTION hint="get the per-layer spikes for the whole test set, then a per-neuron rate is the mean over the (sample, timestep) dims"
with torch.no_grad():
    _, per_layer = snn(X_test, return_spikes=True)
# Mean firing rate per neuron = average over samples and timesteps.
rates_per_layer = [s.mean(dim=(0, 1)).cpu().numpy() for s in per_layer]
# <<< SOLUTION
for l, r in enumerate(rates_per_layer):
    print(f"layer {l + 1}: mean firing rate {r.mean():.3f} spikes/step "
          f"({100 * r.mean():.1f}% of neurons active per step)")

# >>> SOLUTION hint="visualise the per-layer rates, e.g. plot_firing_rates(rates_per_layer, title=...)"
plot_firing_rates(rates_per_layer, title="Per-neuron firing rates (test set)")
plt.show()
# <<< SOLUTION


# %% CELL 3.8 | code  # TASK: inference cost -- MAC vs AC
# A conventional layer computes a dense matrix-vector product: every output is a sum of
# weight x activation MULTIPLY-ACCUMULATE (MAC) operations. In an SNN, a presynaptic
# spike is binary: the synapse just ADDS its weight -> an ACCUMULATE (AC), and only
# when a spike actually occurs. So SNN cost scales with the NUMBER OF SPIKES.
T, C, H, K = ds.n_timesteps, ds.n_channels, 64, ds.n_classes
L = len(per_layer)
mean_rate = [float(r.mean()) for r in rates_per_layer]   # per hidden layer

# All three models run per timestep and pool over time; we count ops over the full
# T-step sequence so the comparison is like-for-like.
# SNN: first layer sees real-valued input (MACs); hidden layers + readout are
# spike-driven (ACs), so their cost scales with the measured firing rates.
# >>> SOLUTION hint="SNN: MACs = dense input projection C*H*T; ACs = sum over spike-driven hidden layers (H*T*rate)*H plus the (H*T*rate)*K readout, scaling with the measured firing rates"
snn_mac = C * H * T                                       # dense input projection
snn_ac = 0.0
for l in range(1, L):                                     # layers 2..L: spike-driven
    snn_ac += (H * T * mean_rate[l - 1]) * H
snn_ac += (H * T * mean_rate[L - 1]) * K                  # spike-driven readout
# <<< SOLUTION

# MLP (memoryless): the same dense net runs at EACH of the T timesteps, all MACs.
# >>> SOLUTION hint="T copies of a dense net: input C*H + hidden H*H + readout H*K"
mlp_mac = T * (C * H + H * H + H * K)
# <<< SOLUTION
# GRU: 3 gates x (input->hidden + hidden->hidden) per layer per timestep, plus a
# per-step readout, all MACs.
# >>> SOLUTION hint="per step: 3 gates over (input->hidden + hidden->hidden) for the first layer, the same for each further layer, plus an H*K readout; times T"
gru_mac = T * (3 * (C * H + H * H) + (L - 1) * 3 * (H * H + H * H) + H * K)
# <<< SOLUTION

# Energy proxy (45nm, Horowitz 2014): 32-bit FP MAC ~ 4.6 pJ, AC ~ 0.9 pJ.
E_MAC, E_AC = 4.6, 0.9
def energy(mac, ac):
    return mac * E_MAC + ac * E_AC

rows = [
    ("MLP", mlp_mac, 0.0),
    ("GRU", gru_mac, 0.0),
    ("SNN", snn_mac, snn_ac),
]
print(f"\n{'model':6s} {'MACs':>12s} {'ACs':>12s} {'energy (pJ)':>14s} {'rel.':>7s}")
base = energy(mlp_mac, 0.0)
for name, mac, ac in rows:
    e = energy(mac, ac)
    print(f"{name:6s} {mac:12,.0f} {ac:12,.0f} {e:14,.0f} {e / base:7.2f}x")

labels = [r[0] for r in rows]
energies = [energy(r[1], r[2]) for r in rows]
# >>> SOLUTION hint="bar chart of energies per model; use a log y-scale since the GRU dwarfs the others on a linear axis"
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels, energies, color=["tab:gray", "tab:orange", "tab:green"])
ax.set_yscale("log")        # GRU dwarfs the others on a linear axis
ax.set_ylabel("estimated inference energy (pJ, log scale)")
ax.set_title("Inference cost: MAC-heavy ANN/RNN vs spike-driven SNN")
for b, e in zip(bars, energies):
    ax.text(b.get_x() + b.get_width() / 2, e, f"{e/1e3:.0f}k",
            ha="center", va="bottom", fontsize=9)
plt.show()
# <<< SOLUTION
