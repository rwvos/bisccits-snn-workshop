<!-- CELL 4.0 | markdown -->
# Bonus — Faster training: forward-gradient injection + `torch.compile`

You reached the bonus — nice. This optional chapter is about **speed**, not accuracy.

In Chapter 2 the SNN trained correctly but *slowly*: its custom `autograd.Function`
forces a "graph break", so `torch.compile` cannot fuse the unrolled time loop and each
of the 30 timesteps pays Python/kernel-launch overhead. Here we rewrite the spike so the
whole model becomes traceable, compile it, and measure the speed-up — at **identical
accuracy**, since it is the same surrogate.

> **Objective.** Express the surrogate without a custom Function, compile the model, and
> see the training time drop while accuracy stays put.

<!-- CELL 4.1 | markdown -->
## The idea — forward-gradient injection

The fix is to express the surrogate as a **single differentiable expression** instead of
a custom Function — *forward-gradient injection*:

```python
spike = (x >= 0).float().detach() + (surr - surr.detach())
```

- **Forward:** `surr - surr.detach() == 0`, so the value is exactly the hard spike.
- **Backward:** the only non-detached term is `surr`, so the gradient is `d surr/dx` —
  our surrogate.

With `surr = x` this is the plain *straight-through estimator*; with
`surr = σ(slope·x)` we recover the smooth sigmoid surrogate from Chapter 1. Because it is
one ordinary expression (no custom Function), the whole model is now traceable, and
`torch.compile` can fuse the time loop for a large speed-up.

> **Note.** `torch.compile` needs a backend (Inductor/Triton) that ships on the Colab
> GPU runtime. On some local setups (e.g. Windows) it is unavailable; the code then
> falls back to eager mode automatically. Run this on Colab to see the speed-up.

<!-- CELL 4.2 | code -> scripts/04_bonus_fgi.py -->
*(No task — make sure the Chapter 2 setup is available. If you ran the whole notebook
top-to-bottom this is a no-op; if you jumped straight here in a fresh kernel it rebuilds
the data and training loop so the bonus runs on its own.)*

<!-- CELL 4.3 | code -> scripts/04_bonus_fgi.py -->
**TASK.** Implement `spike_fgi(x, slope)` as the single-line forward-gradient injection
above. We then run a controlled comparison: the **same architecture** trained two ways —
eagerly with the custom-autograd spike, and with `spike_fgi` under `torch.compile`.
Accuracy should match (same surrogate); the compiled model should train faster.

<!-- CELL 4.4 | markdown -->
## Takeaway

- **Same accuracy, less time.** Both models use the identical sigmoid-derivative
  surrogate, so they learn the same thing. The only difference is *how fast* — on the
  Colab GPU runtime the compiled forward-gradient-injection model trains markedly faster
  than the eager custom-autograd one. (On a CPU-only or backend-less setup the code falls
  back to eager and the two times are similar.)
- **Why it works:** removing the custom `autograd.Function` removes the graph break, so
  `torch.compile` can fuse the whole unrolled time loop into efficient kernels.
- This is the same surrogate you would now plug into larger SNNs: fast to train, ready
  for `torch.compile`.

**Thanks again for joining the BISCCITS SNN workshop!**
