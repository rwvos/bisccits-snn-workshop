<!-- CELL 1.0 | markdown -->
# Chapter 1 — Defining a Spiking Neural Network

**BISCCITS workshop · Spiking Neural Networks**

Welcome! Over three chapters we will go from a single biologically-inspired neuron to a
*trained, efficient* spiking neural network (SNN), and we will measure exactly where
that efficiency comes from.

- **Chapter 1 (this one) — Defining SNNs.** What equations govern a spiking neuron,
  and how do we write them down so a computer can simulate them?
- **Chapter 2 — Training SNNs.** How do we train a network of spiking neurons with
  the same machinery (backpropagation) used for ordinary deep nets, despite the spike
  being non-differentiable?
- **Chapter 3 — Evaluating SNNs.** Once trained, what does the network actually do at
  inference, and how much cheaper is it than a conventional network?
  **Bonus — Can we make it faster?** 

This workshop assumes you are slightly comfortable with the *biology* of neurons but not necessarily 
with machine learning and PyTorch.

> **Objective of Chapter 1.** Implement a leaky integrate-and-fire (LIF) neuron from
> its equations, see its dynamics, and understand the trick that will later let us
> train spiking networks: the *surrogate gradient*.

<!-- CELL 1.1 | markdown -->
## From a biological neuron to a leaky integrate-and-fire model

A biological neuron integrates incoming current in its membrane potential. The membrane leaks
charge over time, so it behaves like a **low-pass filter**: steady input charges it
up, and when the input stops it relaxes back toward rest. When the membrane potential
crosses a **threshold**, the neuron emits a **spike** and its potential is **reset**.

The simplest model capturing this is the **leaky integrate-and-fire (LIF)** neuron.
In continuous time:

$$\tau_\text{mem}\,\frac{dV(t)}{dt} = -V(t) + I(t), \qquad \text{spike when } V \ge V_\text{thr},\ \text{then } V \leftarrow V_\text{reset}.$$

For simulation (and for PyTorch later) we use the **discrete-time** version. With a
step `dt` and decay `beta = exp(-dt / tau_mem)`:

$$
\begin{aligned}
V[t] &= \beta\, V[t-1] + (1-\beta)\, I[t] && \text{(leaky integration / low-pass filter)}\\
S[t] &= \begin{cases}1 & V[t] \ge V_\text{thr}\\ 0 & \text{otherwise}\end{cases} && \text{(threshold $\rightarrow$ spike)}\\
V[t] &\leftarrow V_\text{reset} \quad \text{if } S[t]=1 && \text{(hard reset)}
\end{aligned}
$$

`beta` controls memory: `beta -> 1` means a long membrane time constant (slow,
strongly smoothing), `beta -> 0` means the neuron almost instantly follows its input.

<!-- CELL 1.2 | code -> scripts/01_defining_snns.py -->
**Setup.** Import NumPy, PyTorch and Matplotlib, plus the workshop helpers, and fix a
random seed for reproducibility.

<!-- CELL 1.3 | markdown -->
## Subtask 1 — Simulate one LIF neuron

We will integrate the three equations above over time, given an input current `I(t)`.
Note the order *inside* each timestep: **integrate → check threshold → reset**. The
reset applies to the state carried into the *next* step, which produces the
characteristic "sawtooth" membrane trace.

<!-- CELL 1.4 | code -> scripts/01_defining_snns.py -->
**TASK.** Implement `lif_simulate(current, beta, threshold, v_reset)` returning the
membrane trace `mem` and the binary `spikes` array for an array `current` that contains input currents for a certain number of timesteps. Use a plain Python loop over
timesteps — clarity first; we move to tensors in Chapter 2.

<!-- CELL 1.5 | markdown -->
## Visualising the dynamics

We drive the neuron with a current that is off, then a constant pulse, then off
again, and plot three panels: the input current, the membrane potential (with the
threshold), and the output spikes. You should see the membrane charge up like a
low-pass filter, fire periodically while driven, reset after each spike, and decay
once the input stops.

<!-- CELL 1.6 | code -> scripts/01_defining_snns.py -->
*(No task — run this cell to produce the dynamics plot.)*

<!-- CELL 1.7 | markdown -->
The slope of the charge-up and the firing frequency are both set by the membrane time
constant `tau_mem` (equivalently `beta`). A longer `tau_mem` integrates input over a
longer window — more smoothing, slower firing.

<!-- CELL 1.8 | code -> scripts/01_defining_snns.py -->
**TASK.** Drive the neuron with the *same* step current for several `tau_mem` values
(e.g. 5, 20, 60 steps) and compare the membrane traces and firing rates. Confirm that
a larger `tau_mem` charges more slowly and fires less often.

<!-- CELL 1.9 | markdown -->
## The problem we will face in Chapter 2 — and the surrogate gradient

To *train* a network we need gradients: how does the loss change if we change a particular weight?
Gradients flow backward through every operation — but the spike is a **Heaviside step
function** of the membrane potential, and its derivative is **zero everywhere** (and
undefined exactly at the threshold). No gradient can flow through a spike.

The fix used throughout modern SNN training is the **surrogate gradient**: keep the
Heaviside step in the *forward* pass, but in the *backward* pass pretend the spike was a
smooth function (that approximates the spiking function). A common choice is the **derivative of a sigmoid**,
$\frac{d}{dV}\,\sigma\!\big(k\,(V - V_\text{thr})\big)$, a bump centred on the
threshold. The **slope** `k` controls its width: small `k` spreads gradient over a
wide range of membrane potentials (smooth but biased), large `k` concentrates it near
the threshold (sharp, closer to the true step function but with vanishing gradient away from
threshold).

<!-- CELL 1.10 | code -> scripts/01_defining_snns.py -->
**TASK.** Plot the Heaviside spike together with the sigmoid-derivative surrogate
gradient for a few slope values `k` (e.g. 1, 5, 25). Observe how the slope trades
gradient *width* against *sharpness*. We will plug exactly this surrogate into the
backward pass in Chapter 2.

<!-- CELL 1.11 | markdown -->
### Recap

You implemented a LIF neuron, saw that it is a leaky low-pass integrator that fires
and resets, and met the surrogate gradient that makes spiking networks trainable.
**Next:** stack these neurons into layers, build a deep SNN, and train it on real
sensor data.
