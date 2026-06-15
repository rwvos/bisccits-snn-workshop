# %% [chapter 2] Training Spiking Neural Networks -- solution script
# Paired with content/02_training_snns.md (see CONVENTIONS.md for the cell mapping).
# The classes defined here are the canonical "task solutions"; an identical copy lives
# in snn_workshop/models.py so Chapter 3 / the test runner can import them.

# %% CELL 2.2 | code  (setup -- not a task)
import os
import time
import numpy as np
import torch
import torch.nn as nn

from snn_workshop import set_seed, get_device

set_seed(0)
DEVICE = get_device()
print("device:", DEVICE)


# %% CELL 2.4 | code  # TASK: the spike with a surrogate gradient (custom autograd)
class SpikeFunction(torch.autograd.Function):
    """Heaviside in the forward pass; sigmoid-derivative surrogate in the backward."""

    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x >= 0).float()           # hard spike: 1 if membrane >= threshold

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sig = torch.sigmoid(ctx.slope * x)
        surrogate = ctx.slope * sig * (1.0 - sig)   # d/dx sigmoid(slope * x)
        return grad_output * surrogate, None         # None: no grad w.r.t. slope


def spike_autograd(x, slope=10.0):
    return SpikeFunction.apply(x, slope)


# %% CELL 2.6 | code  # TASK: LIF layer and deep SNN
class LIFLayer(nn.Module):
    """Linear -> leaky integrate-and-fire, unrolled over time (feedforward in space)."""

    def __init__(self, in_features, out_features, beta=0.9, threshold=1.0,
                 slope=10.0, spike_fn=spike_autograd):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.beta = beta
        self.threshold = threshold
        self.slope = slope
        self.spike_fn = spike_fn

    def forward(self, x, return_mem=False):
        B, T, _ = x.shape
        v = torch.zeros(B, self.fc.out_features, device=x.device, dtype=x.dtype)
        spikes, mems = [], []
        for t in range(T):
            current = self.fc(x[:, t, :])                       # input current I[t]
            v = self.beta * v + current                          # leaky integration
            s = self.spike_fn(v - self.threshold, self.slope)    # spike
            if return_mem:
                mems.append(v)
            v = v * (1.0 - s)                                    # hard reset to 0
            spikes.append(s)
        out = torch.stack(spikes, dim=1)                        # (B, T, out_features)
        if return_mem:
            return out, torch.stack(mems, dim=1)
        return out


class DeepSNN(nn.Module):
    """Stack of LIF layers + a linear readout averaged over time."""

    def __init__(self, n_in, hidden=64, n_layers=3, n_classes=4,
                 beta=0.9, threshold=1.0, slope=10.0, spike_fn=spike_autograd):
        super().__init__()
        sizes = [n_in] + [hidden] * n_layers
        self.layers = nn.ModuleList([
            LIFLayer(sizes[i], sizes[i + 1], beta=beta, threshold=threshold,
                     slope=slope, spike_fn=spike_fn)
            for i in range(n_layers)
        ])
        self.readout = nn.Linear(hidden, n_classes)

    def forward(self, x, return_spikes=False):
        s = x
        per_layer = []
        for layer in self.layers:
            s = layer(s)
            per_layer.append(s)
        logits = self.readout(s).mean(dim=1)    # readout per step, then mean over time
        if return_spikes:
            return logits, per_layer
        return logits


# %% CELL 2.8 | code  (load RacketSports -- not a task)
from snn_workshop.data import load_racket_sports

ds = load_racket_sports(normalize=True)
print(f"train {ds.X_train.shape}, test {ds.X_test.shape}")
print(f"T={ds.n_timesteps} timesteps, C={ds.n_channels} channels, "
      f"{ds.n_classes} classes: {ds.class_names}")

X_train = torch.tensor(ds.X_train, device=DEVICE)
y_train = torch.tensor(ds.y_train, device=DEVICE)
X_test = torch.tensor(ds.X_test, device=DEVICE)
y_test = torch.tensor(ds.y_test, device=DEVICE)


# %% CELL 2.10 | code  (shared training / evaluation loop -- not a task)
def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()


def train_model(model, epochs=80, lr=2e-3, batch_size=32, seed=0):
    """Mini-batch Adam training. Returns (train_acc, test_acc, wall_time_seconds)."""
    set_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = X_train.shape[0]

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return accuracy(model, X_train, y_train), accuracy(model, X_test, y_test), wall


# Train the SNN with the custom-autograd surrogate.
snn = DeepSNN(ds.n_channels, hidden=64, n_layers=3, n_classes=ds.n_classes,
              beta=0.9, threshold=1.0, slope=10.0, spike_fn=spike_autograd)
tr, te, wall = train_model(snn)
results = {"SNN (autograd surrogate)": (tr, te, wall)}
print(f"SNN (autograd): train {tr:.3f}  test {te:.3f}  time {wall:.1f}s")


# %% CELL 2.12 | code  (non-spiking baselines -- not a task)
class MLP(nn.Module):
    def __init__(self, n_in, n_timesteps, hidden=64, n_layers=3, n_classes=4):
        super().__init__()
        dims = [n_in * n_timesteps] + [hidden] * (n_layers - 1)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(dims[-1], n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(start_dim=1))


class GRUClassifier(nn.Module):
    def __init__(self, n_in, hidden=64, n_layers=3, n_classes=4):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=n_layers, batch_first=True)
        self.readout = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.readout(out[:, -1, :])


mlp = MLP(ds.n_channels, ds.n_timesteps, 64, 3, ds.n_classes)
tr, te, wall = train_model(mlp)
results["MLP (flatten time)"] = (tr, te, wall)
print(f"MLP: train {tr:.3f}  test {te:.3f}  time {wall:.1f}s")

gru = GRUClassifier(ds.n_channels, 64, 3, ds.n_classes)
tr, te, wall = train_model(gru)
results["GRU"] = (tr, te, wall)
print(f"GRU: train {tr:.3f}  test {te:.3f}  time {wall:.1f}s")


# %% CELL 2.14 | code  # TASK: forward-gradient injection + torch.compile
def spike_fgi(x, slope=10.0):
    """Same surrogate as SpikeFunction, but as one compilable expression.

    forward  : (x >= 0)                  because (surr - surr.detach()) == 0
    backward : d/dx sigmoid(slope * x)   it flows through the non-detached surr term
    """
    hard = (x >= 0).float()
    surr = torch.sigmoid(slope * x)
    return hard.detach() + (surr - surr.detach())


# Identical architecture, swap in the compilable spike, then torch.compile it.
snn_fgi = DeepSNN(ds.n_channels, hidden=64, n_layers=3, n_classes=ds.n_classes,
                  beta=0.9, threshold=1.0, slope=10.0, spike_fn=spike_fgi)

# torch.compile needs a working Inductor/Triton backend. That ships on the Colab
# (Linux) GPU runtime where this workshop runs; on Windows it is typically absent, so
# we skip it there (set SNN_FORCE_COMPILE=1 to try anyway).
import sys
can_compile = sys.platform != "win32" or os.environ.get("SNN_FORCE_COMPILE") == "1"

if can_compile:
    try:
        compiled = torch.compile(snn_fgi)
        _ = compiled(X_train[:8])        # trigger compilation (slow the first time)
        tr, te, wall = train_model(compiled)
        results["SNN (FGI + compile)"] = (tr, te, wall)
        print(f"SNN (FGI+compile): train {tr:.3f}  test {te:.3f}  time {wall:.1f}s")
    except Exception as e:
        print("torch.compile failed; training FGI model eagerly instead.")
        print("  reason:", repr(e)[:200])
        tr, te, wall = train_model(snn_fgi)
        results["SNN (FGI, eager)"] = (tr, te, wall)
        print(f"SNN (FGI, eager): train {tr:.3f}  test {te:.3f}  time {wall:.1f}s")
else:
    print("Skipping torch.compile on this platform (no backend); training eagerly.")
    print("On the Colab GPU runtime this cell trains a compiled model -- much faster.")
    tr, te, wall = train_model(snn_fgi)
    results["SNN (FGI, eager)"] = (tr, te, wall)
    print(f"SNN (FGI, eager): train {tr:.3f}  test {te:.3f}  time {wall:.1f}s")


# %% CELL 2.15b | code  (summary table + save checkpoint for Chapter 3 -- not a task)
print("\n=== Summary ===")
print(f"{'model':28s} {'train':>7s} {'test':>7s} {'time (s)':>9s}")
for name, (tr, te, wall) in results.items():
    print(f"{name:28s} {tr:7.3f} {te:7.3f} {wall:9.1f}")

os.makedirs("checkpoints", exist_ok=True)
ckpt = {
    "state_dict": snn_fgi.state_dict(),
    "config": dict(n_in=ds.n_channels, hidden=64, n_layers=3, n_classes=ds.n_classes,
                   beta=0.9, threshold=1.0, slope=10.0),
    "class_names": ds.class_names,
}
torch.save(ckpt, "checkpoints/snn_racketsports.pt")
print("\nsaved trained SNN -> checkpoints/snn_racketsports.pt")
