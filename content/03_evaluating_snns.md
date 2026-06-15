<!-- CELL 3.0 | markdown -->
# Chapter 3 — Evaluating a Spiking Neural Network

We have a trained SNN. The interesting question is no longer "how accurate is it?"
(we saw that in Chapter 2) but **"what does it actually do at inference, and why is it
efficient?"** In this chapter we:

1. look *inside* the network with **spike raster** plots,
2. measure how **sparse** its activity is, and
3. quantify the inference-time **compute cost** — the difference between
   **multiply-accumulate (MAC)** and **accumulate (AC)** operations — and compare the
   SNN with the MLP and the GRU.

> **Objective.** Build intuition for *where the efficiency of SNNs comes from*: sparse,
> binary, event-driven communication that replaces dense multiplications with sparse
> additions.

<!-- CELL 3.1 | markdown -->
## Recover the trained network

In the full notebook the trained `snn` from Chapter 2 is still in memory. This chapter
reloads it from the checkpoint so it can also be run on its own. We then push the test
set through it and study the spikes.

<!-- CELL 3.2 | code -> scripts/03_evaluating_snns.py -->
**Setup.** Load the test data, rebuild the SNN, and load the trained weights.

<!-- CELL 3.3 | markdown -->
## Subtask 1 — Look inside: the spike raster

An SNN does not pass real numbers between layers — it passes **spikes**: binary events
in time. A **spike raster** shows, for every neuron (rows) and every timestep
(columns), when it fired. This is exactly how neuroscientists visualise recorded
neural activity, so it should feel familiar.

<!-- CELL 3.4 | code -> scripts/03_evaluating_snns.py -->
**TASK.** Take one (correctly classified) test trial, run it through the SNN with
`return_spikes=True`, and plot a raster per hidden layer. Notice how activity is
*sparse* and *distributed* — most neurons are silent at any given timestep.

<!-- CELL 3.5 | markdown -->
## Subtask 2 — How sparse is the network?

The raster suggested sparsity; let us quantify it. The **firing rate** of a neuron is
the fraction of timesteps on which it spikes (a number in [0, 1]). Averaged over the
test set, low firing rates mean most neurons are silent most of the time — and, as we
will see next, *silence is free* in a spiking network.

<!-- CELL 3.6 | code -> scripts/03_evaluating_snns.py -->
**TASK.** Compute the mean firing rate per neuron for each layer over the whole test
set and plot the distributions. Typical rates here are well below 50% — the network
has learned a sparse code.

<!-- CELL 3.7 | markdown -->
## Subtask 3 — The cost of inference: MAC vs AC

Here is the crux of SNN efficiency.

- A conventional (ANN/RNN) layer computes a dense matrix–vector product. Every output
  is a sum of **weight × activation** terms — each a **multiply-accumulate (MAC)**.
  The cost is fixed by the layer dimensions and paid *in full, every time*.
- In an SNN, a presynaptic activation is a **spike: either 0 or 1**. When it is 1 the
  synapse simply **adds** its weight to the postsynaptic sum — an **accumulate (AC)**,
  *no multiply*. When it is 0 there is **nothing to do at all**. So the SNN's cost
  scales with the **number of spikes**, not the number of synapses.

Two things make this cheap: (1) an AC is several times less energy than a MAC (in a
classic 45 nm estimate, ~0.9 pJ vs ~4.6 pJ for 32-bit FP), and (2) sparsity means few
ACs are ever performed. On event-driven *neuromorphic* hardware the "do nothing for a
0" part is literal — unused synapses consume no energy.

> **A fair comparison.** Our **MLP** flattens time into a single dense pass, so in raw
> op-count it looks cheap — but it is *not* a streaming/temporal model and its first
> weight matrix grows with sequence length. The **GRU** is the like-for-like baseline:
> a continuous-valued recurrent network that, like the SNN, processes the sequence
> step by step. That is the comparison to watch.

<!-- CELL 3.8 | code -> scripts/03_evaluating_snns.py -->
**TASK.** Using the measured firing rates, estimate the operation counts:
the SNN's first layer sees real-valued input (MACs) while its hidden/readout layers
are spike-driven (ACs); the MLP and GRU are all MACs. Convert to an energy proxy and
compare. **Expect the SNN to use a tiny fraction of the GRU's energy** — the headline
result — while the flatten-MLP, for the caveats above, is a different kind of model.

<!-- CELL 3.9 | markdown -->
## Discussion — what this does and does not show

- **vs the RNN (GRU):** the SNN does the same job — streaming temporal classification —
  for a fraction of the energy, because it replaces dense per-step MACs with sparse
  ACs. This is the result that matters and it grows with sequence length.
- **vs the MLP:** raw op-counts can favour a flatten-once MLP on a short, fixed-length
  trial, but that model is non-causal (needs the whole sequence at once) and scales
  poorly as $T$ grows. The SNN is causal and event-driven.
- **It is a proxy, not a hardware measurement.** Real energy depends on memory traffic,
  dataflow, precision and the actual chip; the MAC/AC counts capture the *arithmetic*
  story, which is where neuromorphic hardware wins.
- **More to explore:** asynchronicity (no global clock; neurons act on events),
  dense matrix multiply vs sparse **pop-count**/address-event routing, and trading a
  little accuracy for far fewer spikes via a **firing-rate regularizer** during
  training (try adding a small penalty on the mean spike count and re-running
  Chapter 2 — accuracy often holds while energy drops).

<!-- CELL 3.10 | markdown -->
## Wrap-up

Across three chapters you defined a spiking neuron from its equations, trained a deep
SNN with surrogate gradients (and made it compile-friendly with forward-gradient
injection), and saw concretely where its inference efficiency comes from. The same
recipe scales to larger networks and to neuromorphic hardware, where sparse,
event-driven computation turns into real energy savings.

**Thank you for joining the BISCCITS SNN workshop!**
