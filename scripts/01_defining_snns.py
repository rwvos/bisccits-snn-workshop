# %% [chapter 1] Defining Spiking Neural Networks -- solution script
# Paired with content/01_defining_snns.md (see CONVENTIONS.md for the cell mapping).
# Run top-to-bottom; each `# %% CELL 1.k` block is one notebook code cell.

# %% CELL 1.2 | code  (setup -- not a task)
import numpy as np
import torch
import matplotlib.pyplot as plt

from snn_workshop import set_seed
from snn_workshop.viz import plot_membrane_dynamics

set_seed(0)


# %% CELL 1.4 | code  # TASK: simulate a single LIF neuron
def lif_simulate(current, beta, threshold=1.0, v_reset=0.0):
    """Simulate one leaky integrate-and-fire neuron over time.

    Discrete-time LIF as a *low-pass filter* of the input current with a *hard reset*:

        V[t] = beta * V[t-1] + (1 - beta) * I[t]     # leaky integration
        S[t] = 1  if V[t] >= threshold  else 0       # threshold -> spike
        V[t] <- v_reset  if S[t] == 1                # hard reset

    `beta = exp(-dt/tau_mem)` is the membrane decay (0 < beta < 1): larger beta =
    longer memory = stronger low-pass smoothing.

    Parameters
    ----------
    current : array (T,)   input current I(t)
    beta    : float        membrane decay in (0, 1)
    threshold, v_reset : float

    Returns
    -------
    mem   : array (T,)   membrane potential recorded *at* each step (peaks at spikes)
    spikes: array (T,)   binary spike train
    """
    current = np.asarray(current, dtype=np.float64)
    T = len(current)
    mem = np.zeros(T)
    spikes = np.zeros(T)

    v = v_reset  # state carried between steps
    for t in range(T):
        v = beta * v + (1.0 - beta) * current[t]   # leaky integration
        s = 1.0 if v >= threshold else 0.0          # spike?
        mem[t] = v                                  # record the value that crossed
        spikes[t] = s
        if s > 0.5:
            v = v_reset                             # hard reset for the next step
    return mem, spikes


# %% CELL 1.6 | code  (visualise the dynamics -- not a task)
T = 200
dt = 1.0

# A current that is off, then a constant drive, then off again.
current = np.zeros(T)
current[50:150] = 1.5

beta = np.exp(-dt / 20.0)  # tau_mem = 20 steps
mem, spikes = lif_simulate(current, beta=beta, threshold=1.0)

fig, _ = plot_membrane_dynamics(
    current, mem, spikes, threshold=1.0, dt=dt,
    title=f"LIF neuron  (beta={beta:.3f}, tau_mem=20 steps)",
)
plt.show()
print(f"spike count: {int(spikes.sum())}")


# %% CELL 1.8 | code  # TASK: how does the membrane time constant shape the dynamics?
# Drive the neuron with the same step current for several values of tau_mem and
# compare both the membrane trace and the resulting firing rate.
taus = [5.0, 20.0, 60.0]
step = np.zeros(T)
step[20:] = 1.2

fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
for tau in taus:
    b = np.exp(-dt / tau)
    m, s = lif_simulate(step, beta=b, threshold=1.0)
    axes[0].plot(np.arange(T) * dt, m, label=f"tau={tau:.0f}  (beta={b:.3f})")
    axes[1].vlines(np.arange(T)[s > 0.5] * dt, 0, 1, label=f"tau={tau:.0f}")
    print(f"tau={tau:>4.0f}  rate={s.mean():.3f} spikes/step")
axes[0].axhline(1.0, color="tab:red", ls="--", lw=1)
axes[0].set_ylabel("V(t)")
axes[0].legend(fontsize=8)
axes[0].set_title("Effect of membrane time constant on LIF dynamics")
axes[1].set_yticks([])
axes[1].set_ylabel("spikes")
axes[1].set_xlabel("time")
plt.show()


# %% CELL 1.10 | code  # TASK: the surrogate gradient (derivative of a sigmoid)
# The spike is a Heaviside step: its derivative is 0 everywhere (and undefined at 0),
# so gradients cannot flow. During the *backward* pass we replace it with a smooth
# surrogate: the derivative of a sigmoid sigma(beta_s * v), peaked at threshold.
def heaviside(v, threshold=1.0):
    return (v >= threshold).astype(np.float64)


def sigmoid_surrogate_grad(v, threshold=1.0, slope=5.0):
    """d/dv of sigma(slope * (v - threshold)) -- a bump centred at the threshold."""
    x = slope * (v - threshold)
    sig = 1.0 / (1.0 + np.exp(-x))
    return slope * sig * (1.0 - sig)


v = np.linspace(-1.0, 3.0, 400)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(v, heaviside(v), color="black", lw=2, label="spike S = H(V - thr)")
for slope in [1.0, 5.0, 25.0]:
    ax.plot(v, sigmoid_surrogate_grad(v, slope=slope), label=f"surrogate grad, slope={slope:g}")
ax.axvline(1.0, color="tab:red", ls="--", lw=1, label="threshold")
ax.set_xlabel("membrane potential V")
ax.set_ylabel("value")
ax.set_title("Heaviside spike vs. sigmoid-derivative surrogate gradient")
ax.legend(fontsize=8)
plt.show()
