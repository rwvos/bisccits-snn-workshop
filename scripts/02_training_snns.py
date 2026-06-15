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
import matplotlib.pyplot as plt

from snn_workshop import set_seed, get_device
from snn_workshop.viz import (
    plot_training_curves, plot_metric_comparison, plot_method_grid,
    plot_runtime_bar, plot_confusion_matrices,
)

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
@torch.no_grad()
def evaluate(model, X, y, loss_fn):
    model.eval()
    out = model(X)
    return loss_fn(out, y).item(), (out.argmax(dim=1) == y).float().mean().item()


def train_model(model, epochs=80, lr=2e-3, batch_size=32, seed=0):
    """Mini-batch Adam training with per-epoch logging.

    Returns a dict: ``history`` (per-epoch train/test loss & accuracy), the final
    ``train_acc``/``test_acc``, and ``wall`` = training-only wall-clock seconds
    (the per-epoch evaluation used for the curves is deliberately *excluded* from the
    timing, so the runtime comparison reflects training compute only).
    """
    set_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = X_train.shape[0]
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    train_time = 0.0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        train_time += time.perf_counter() - t0

        # Per-epoch logging (not timed).
        trl, tra = evaluate(model, X_train, y_train, loss_fn)
        tel, tea = evaluate(model, X_test, y_test, loss_fn)
        history["train_loss"].append(trl); history["train_acc"].append(tra)
        history["test_loss"].append(tel); history["test_acc"].append(tea)

    return {"history": history, "train_acc": history["train_acc"][-1],
            "test_acc": history["test_acc"][-1], "wall": train_time}


# `results` collects metrics per run; `models` keeps the trained models for later cells.
results = {}
models = {}


def report(name, res):
    results[name] = res
    print(f"{name:26s}: train {res['train_acc']:.3f}  test {res['test_acc']:.3f}  "
          f"time {res['wall']:.1f}s")


# Train the SNN with the custom-autograd surrogate.
snn = DeepSNN(ds.n_channels, hidden=64, n_layers=3, n_classes=ds.n_classes,
              beta=0.9, threshold=1.0, slope=10.0, spike_fn=spike_autograd)
report("SNN (autograd)", train_model(snn))


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
report("MLP", train_model(mlp))

gru = GRUClassifier(ds.n_channels, 64, 3, ds.n_classes)
report("GRU", train_model(gru))


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
        report("SNN (FGI + compile)", train_model(compiled))
    except Exception as e:
        print("torch.compile failed; training FGI model eagerly instead.")
        print("  reason:", repr(e)[:200])
        report("SNN (FGI, eager)", train_model(snn_fgi))
else:
    print("Skipping torch.compile on this platform (no backend); training eagerly.")
    print("On the Colab GPU runtime this cell trains a compiled model -- much faster.")
    report("SNN (FGI, eager)", train_model(snn_fgi))

# Keep the trained models for the confusion-matrix cell (use the FGI SNN as "SNN").
models = {"MLP": mlp, "SNN": snn_fgi, "GRU": gru}


# %% CELL 2.16 | code  (summary table + save checkpoint for Chapter 3 -- not a task)
print("\n=== Summary ===")
print(f"{'model':22s} {'train':>7s} {'test':>7s} {'time (s)':>9s}")
for name, res in results.items():
    print(f"{name:22s} {res['train_acc']:7.3f} {res['test_acc']:7.3f} {res['wall']:9.1f}")

os.makedirs("checkpoints", exist_ok=True)
ckpt = {
    "state_dict": snn_fgi.state_dict(),
    "config": dict(n_in=ds.n_channels, hidden=64, n_layers=3, n_classes=ds.n_classes,
                   beta=0.9, threshold=1.0, slope=10.0),
    "class_names": ds.class_names,
}
torch.save(ckpt, "checkpoints/snn_racketsports.pt")
print("saved trained SNN -> checkpoints/snn_racketsports.pt")


# %% CELL 2.18 | code  (training curves -- not a task)
# Loss and accuracy vs epoch (train solid, test dashed) for each model type.
for name in ["SNN (autograd)", "MLP", "GRU"]:
    plot_training_curves(results[name]["history"], title=f"{name} — training curves")
    plt.show()


# %% CELL 2.18c | code  (cross-method comparison on the training set -- not a task)
# Overlay every method's TRAIN accuracy on one axis to compare how fast/high each
# learns on the data it is trained on.
histories = {name: res["history"] for name, res in results.items()}
plot_metric_comparison(histories, "train_acc", ylabel="accuracy",
                       title="Training-set accuracy — all methods")
plt.show()


# %% CELL 2.18d | code  (2x2 method comparison grid -- not a task)
# Rows: train (top) vs test (bottom). Columns: loss (left) vs accuracy (right).
# Every panel overlays all methods so they can be compared directly.
plot_method_grid(histories, title="Method comparison — loss & accuracy, train & test")
plt.show()


# %% CELL 2.20 | code  (training time comparison -- not a task)
# Single number per run: total training wall-clock time. The two SNN variants differ
# only in runtime (and tiny seeding fluctuations), not in what they learn.
plot_runtime_bar({name: res["wall"] for name, res in results.items()},
                 title=f"Training time for {len(results)} runs (same epochs)")
plt.show()


# %% CELL 2.22 | code  (confusion matrices -- not a task)
def confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


short_names = [n.replace("Badminton", "Bad").replace("Squash", "Squ")
                .replace("Backhand Boast", "BH").replace("Forehand Boast", "FH")
               for n in ds.class_names]

y_true = y_test.cpu().numpy()
cms, names = [], []
for name, model in models.items():
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).argmax(dim=1).cpu().numpy()
    cms.append(confusion(y_true, y_pred, ds.n_classes))
    names.append(name)

plot_confusion_matrices(cms, short_names, names, normalize=True)
plt.show()
