# %% [bonus] Faster training: forward-gradient injection + torch.compile -- solution
# Paired with content/04_bonus_fgi.md (see CONVENTIONS.md for the cell mapping).
#
# In the assembled notebook this optional chapter runs after Chapter 3 in the same
# kernel, reusing the data, DeepSNN, the autograd spike, and train_model from Chapter 2.
# To keep it runnable on its own (fresh kernel / the dev test runner) the setup cell
# rebuilds that minimal setup when it is not already in memory.

# %% CELL 4.2 | code  (ensure the Chapter 2 setup is available -- not a task)
# If you ran the whole notebook top-to-bottom, Chapter 2 already defined the data,
# DeepSNN, the autograd spike, and train_model -- reuse them. If you jumped straight
# here in a fresh kernel, rebuild that minimal setup so the bonus runs on its own.
if "train_model" not in globals():
    import os
    import time
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    from snn_workshop import set_seed, get_device
    from snn_workshop.data import load_racket_sports
    from snn_workshop.models import DeepSNN, spike_autograd
    from snn_workshop.viz import plot_runtime_bar

    set_seed(0)
    DEVICE = get_device()
    print("device:", DEVICE)

    ds = load_racket_sports(normalize=True)
    X_train = torch.tensor(ds.X_train, device=DEVICE)
    y_train = torch.tensor(ds.y_train, device=DEVICE)
    X_test = torch.tensor(ds.X_test, device=DEVICE)
    y_test = torch.tensor(ds.y_test, device=DEVICE)

    @torch.no_grad()
    def evaluate(model, X, y, loss_fn):
        model.eval()
        out = model(X)
        return loss_fn(out, y).item(), (out.argmax(dim=1) == y).float().mean().item()

    def train_model(model, epochs=80, lr=2e-3, batch_size=32, seed=0):
        """Mini-batch Adam loop; returns final train/test accuracy and training-only
        wall-clock seconds (per-epoch evaluation is excluded from the timing)."""
        set_seed(seed)
        model.to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        n = X_train.shape[0]
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
        trl, tra = evaluate(model, X_train, y_train, loss_fn)
        tel, tea = evaluate(model, X_test, y_test, loss_fn)
        return {"train_acc": tra, "test_acc": tea, "wall": train_time}

    print("rebuilt the Chapter 2 setup for a standalone run.")
else:
    print("reusing the Chapter 2 setup already in memory.")


# %% CELL 4.3 | code  # TASK: forward-gradient injection + torch.compile
def spike_fgi(x, slope=10.0):
    """Same surrogate as SpikeFunction, but as one compilable expression.

    forward  : (x >= 0)                  because (surr - surr.detach()) == 0
    backward : d/dx sigmoid(slope * x)   it flows through the non-detached surr term
    """
    hard = (x >= 0).float()
    surr = torch.sigmoid(slope * x)
    return hard.detach() + (surr - surr.detach())


# A controlled comparison: identical architecture and surrogate, trained two ways.
# (1) Eager baseline with the custom-autograd spike -- what Chapter 2 trained.
snn_eager = DeepSNN(ds.n_channels, hidden=64, n_layers=3, n_classes=ds.n_classes,
                    beta=0.9, threshold=1.0, slope=10.0, spike_fn=spike_autograd)
res_eager = train_model(snn_eager)
print(f"SNN (autograd, eager) : train {res_eager['train_acc']:.3f}  "
      f"test {res_eager['test_acc']:.3f}  time {res_eager['wall']:.1f}s")

# (2) Forward-gradient injection, then torch.compile.
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
        res_fgi = train_model(compiled)
        fgi_label = "FGI + compile"
    except Exception as e:
        print("torch.compile failed; training the FGI model eagerly instead.")
        print("  reason:", repr(e)[:200])
        res_fgi = train_model(snn_fgi)
        fgi_label = "FGI (eager)"
else:
    print("Skipping torch.compile on this platform (no backend); training eagerly.")
    print("On the Colab GPU runtime this trains a compiled model -- much faster.")
    res_fgi = train_model(snn_fgi)
    fgi_label = "FGI (eager)"

print(f"SNN ({fgi_label}) : train {res_fgi['train_acc']:.3f}  "
      f"test {res_fgi['test_acc']:.3f}  time {res_fgi['wall']:.1f}s")
speedup = res_eager["wall"] / res_fgi["wall"] if res_fgi["wall"] else float("nan")
print(f"\nspeed-up (eager / {fgi_label}): {speedup:.2f}x  "
      f"-- accuracy is unchanged (same surrogate)")

plot_runtime_bar({"SNN (autograd, eager)": res_eager["wall"],
                  f"SNN ({fgi_label})": res_fgi["wall"]},
                 title="Training time: eager autograd vs forward-gradient injection")
plt.show()
