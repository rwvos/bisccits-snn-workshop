"""Reference SNN / baseline models for the workshop.

This is the **same code participants implement in Chapter 2**, packaged here so that
later chapters and the test runner can import it. In the assembled participant
notebook these classes are defined directly in Chapter 2's cells (single kernel), so
Chapter 3 does not import anything — it reuses the in-memory objects.

Design choices (see Chapter 2 markdown for the reasoning):
* Feedforward LIF layers: ``Linear -> LIF``; the only recurrence is the membrane
  state evolving over timesteps.
* Readout = ``mean over time`` of a linear projection of the last layer's spikes.
* Two spike mechanisms with identical forward behaviour but different backward paths:
  a custom ``autograd.Function`` (not compilable) and forward-gradient injection
  (a single expression -> ``torch.compile``-friendly).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------------
# Spike mechanisms
# --------------------------------------------------------------------------------
class SpikeFunction(torch.autograd.Function):
    """Heaviside spike in forward; sigmoid-derivative surrogate in backward.

    Custom autograd.Function: explicit, easy to read -- but it forces a graph break,
    so a model using it **cannot** be ``torch.compile``d.
    """

    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sig = torch.sigmoid(ctx.slope * x)
        surrogate = ctx.slope * sig * (1.0 - sig)  # d/dx sigmoid(slope * x)
        return grad_output * surrogate, None


def spike_autograd(x, slope: float = 10.0):
    return SpikeFunction.apply(x, slope)


def spike_fgi(x, slope: float = 10.0):
    """Forward-gradient injection: same surrogate, but as a single expression.

        forward  : (x >= 0)                       (the surr - surr.detach() term is 0)
        backward : d/dx sigmoid(slope * x)        (flows through the non-detached surr)

    Because it is one differentiable line (no custom Function), a model built on it is
    fully traceable by ``torch.compile``. The straight-through estimator is the special
    case ``surr = x``; here ``surr = sigmoid(slope * x)`` gives the smooth surrogate.
    """
    hard = (x >= 0).float()
    surr = torch.sigmoid(slope * x)
    return hard.detach() + (surr - surr.detach())


# --------------------------------------------------------------------------------
# Spiking network
# --------------------------------------------------------------------------------
class LIFLayer(nn.Module):
    """Linear -> leaky integrate-and-fire, unrolled over time (feedforward in space)."""

    def __init__(self, in_features, out_features, beta=0.9, threshold=1.0,
                 slope=10.0, spike_fn=spike_fgi):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.beta = beta
        self.threshold = threshold
        self.slope = slope
        self.spike_fn = spike_fn

    def forward(self, x, return_mem=False):
        # x: (B, T, in_features)
        B, T, _ = x.shape
        out_features = self.fc.out_features
        v = torch.zeros(B, out_features, device=x.device, dtype=x.dtype)
        spikes = []
        mems = []
        for t in range(T):
            current = self.fc(x[:, t, :])
            v = self.beta * v + current                          # leaky integration
            s = self.spike_fn(v - self.threshold, self.slope)    # spike (surrogate bwd)
            if return_mem:
                mems.append(v)
            v = v * (1.0 - s)                                     # hard reset to 0
            spikes.append(s)
        out = torch.stack(spikes, dim=1)                         # (B, T, out_features)
        if return_mem:
            return out, torch.stack(mems, dim=1)
        return out


class DeepSNN(nn.Module):
    """Stack of LIF layers + a linear readout averaged over time."""

    def __init__(self, n_in, hidden=64, n_layers=3, n_classes=4,
                 beta=0.9, threshold=1.0, slope=10.0, spike_fn=spike_fgi):
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
        logits_t = self.readout(s)          # (B, T, n_classes)
        logits = logits_t.mean(dim=1)       # average over time
        if return_spikes:
            return logits, per_layer
        return logits


# --------------------------------------------------------------------------------
# Non-spiking baselines (matched depth / width)
# --------------------------------------------------------------------------------
class MLP(nn.Module):
    """Memoryless per-timestep MLP (the honest "no temporal state" baseline).

    The *same* small network is applied independently to every timestep's observation
    vector, giving per-step logits that are averaged over time -- the same readout
    contract as the SNN/GRU. Unlike them it carries **no state across time**.
    """

    def __init__(self, n_in, hidden=64, n_layers=3, n_classes=4):
        super().__init__()
        dims = [n_in] + [hidden] * (n_layers - 1)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(dims[-1], n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, C) -> shared net at each timestep -> (B, T, K) -> mean over time
        B, T, C = x.shape
        logits_t = self.net(x.reshape(B * T, C)).reshape(B, T, -1)
        return logits_t.mean(dim=1)


class GRUClassifier(nn.Module):
    """3-layer GRU with a per-timestep readout averaged over time.

    Same input/output contract as the SNN (per-step logits, mean over time); the only
    difference from the SNN is the continuous-valued recurrent cell.
    """

    def __init__(self, n_in, hidden=64, n_layers=3, n_classes=4):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=n_layers, batch_first=True)
        self.readout = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, _ = self.gru(x)                    # (B, T, hidden)
        return self.readout(out).mean(dim=1)    # per-step logits, averaged over time
