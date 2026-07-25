# WAX-ML Roadmap

## Current state

WAX-ML has two backends: Haiku (legacy) and Flax (new). Both have full module
coverage (27 modules each) with 430+ passing tests across the project.

**What works today:**
- Core streaming transforms: `@streaming_transform_with_state`, `@update_on_event`, `@streaming_scan`, `@streaming_optimizer`
- All Haiku modules ported to Flax with numerical consistency verified
- JIT/scan compatibility, transform compositions, advanced state patterns
- Debugging/profiling tools, visualization, interactive dashboards
- The `.wax` pandas/xarray accessor (Haiku backend only)

**What doesn't:**
- The `.wax` accessor and `Stream` class are hardwired to Haiku. Flax modules can only be used through the lower-level `flax_transform_with_state` / `flax_unroll_transform` API.
- `update_on_event` caches output correctly but does not roll back child module state (EWMA etc. still advance on non-events). The Haiku `UpdateOnEvent` does full state rollback.
- `streaming_optimizer` optimizes a scalar scale parameter, not the full model graph.

---

## Phase 1 -- Flax as primary backend

Haiku is in maintenance mode. The goal is to make Flax the default backend so
users get the same `data.wax.stream().apply(fn)` experience with Flax modules.

### 1.1 Flax accessor integration
Wire `flax_transform_with_state` and `flax_unroll_transform` into the existing
pandas/xarray `.wax` accessor so that Flax modules work with
`data.wax.stream().apply(fn)`.

- [ ] Add `backend="flax"` option to `Stream` and accessor classes
- [ ] Adapt `unroll` path to dispatch to `flax_unroll_transform` when backend is Flax
- [ ] Preserve output format (DataFrame/Dataset) for Flax results
- [ ] Integration tests: same pipeline, same data, both backends, same output

### 1.2 ConditionalComputation state rollback
The current `update_on_event` always runs the inner module (so state advances
even when the event doesn't fire). Implement state save/restore around the
inner call using Flax's variable collections.

- [ ] Before calling `update_fn`, snapshot all mutable variable collections
- [ ] After calling `update_fn`, use `jnp.where(should_update, new, old)` on every state variable
- [ ] Test: EWMA state does not change on non-event steps
- [ ] Test: numerical equivalence with Haiku `UpdateOnEvent`

### 1.3 StreamingOptimizer full-parameter gradients
The current implementation differentiates through a single scale parameter.
Extend it to differentiate through the model's own parameters, matching the
Flax `OnlineOptimizer` module's approach.

- [ ] Use `nn.apply` with `mutable` collections to extract model params as a pytree
- [ ] Compute `jax.value_and_grad` of the loss w.r.t. the model params
- [ ] Apply optax updates and write back to the variable collection
- [ ] Test: gradient norms decrease over a training sequence
- [ ] Test: predictions improve on a simple regression problem

### 1.4 Haiku deprecation path
- [ ] Add deprecation warnings to Haiku-only code paths in `accessors.py` and `stream.py`
- [ ] Document migration guide: Haiku module -> Flax equivalent
- [ ] Ensure all notebooks and examples have Flax versions

---

## Phase 2 -- Correctness and polish

### 2.1 Test hardening
- [ ] Buffer tests: assert on returned output (ordered), not internal circular state
- [ ] Property-based tests for Buffer, EWMA, ARMA (hypothesis or hand-rolled)
- [ ] Cross-backend consistency: run every Haiku module test against its Flax counterpart
- [ ] CI: run full suite on Python 3.11+ with JAX CPU

### 2.2 Performance benchmarks
- [ ] Benchmark Flax vs Haiku: throughput, memory, JIT compile time
- [ ] Benchmark Buffer: old `jnp.roll` vs new write-pointer (vary `maxlen`)
- [ ] Publish results in a notebook or docs page

### 2.3 Documentation
- [ ] API reference for `wax.flax.core` and `wax.flax.modules`
- [ ] Tutorial: building a streaming pipeline from scratch (Flax)
- [ ] Tutorial: migrating from Haiku to Flax
- [ ] Docstrings: fill gaps in Flax modules (match Haiku quality)

---

## Phase 3 -- Targeted extensions

Only items that build on existing code and serve demonstrated use cases.

### 3.1 Compressed buffer improvements
`CompressedBuffer` already exists (518 lines, 4 strategies). Harden it.

- [ ] Test numerical accuracy of each compression strategy over long sequences
- [ ] Add `auto` strategy that picks compression based on data statistics
- [ ] Benchmark memory savings vs. accuracy trade-off

### 3.2 Online learning improvements
Build on the fixed `StreamingOptimizer` and existing `OnlineOptimizer`.

- [ ] Streaming learning rate schedules (warmup, decay, cyclical)
- [ ] Gradient clipping and NaN protection in the streaming optimizer
- [ ] Example: online linear regression with convergence plot
- [ ] Example: online EWMA parameter tuning via gradient descent

### 3.3 Streaming anomaly detection
A natural extension of the EWMA/buffer primitives already in the library.

- [ ] CUSUM (cumulative sum) change-point detector module
- [ ] Streaming z-score anomaly detector
- [ ] Example: real-time anomaly detection on sensor data

---

## Non-goals

The following are explicitly out of scope to keep the project focused:

- **Distributed streaming / Kafka / Pulsar integration** -- WAX-ML is a computation library, not a data pipeline framework. Use Kafka consumers to feed data into WAX-ML, not the other way around.
- **Federated learning** -- orthogonal concern, better served by dedicated libraries.
- **Custom JAX primitives / compiler passes** -- JAX's XLA compiler already optimizes `jax.lax.scan` and fuses operations. Manual primitives add maintenance burden with marginal gain.
- **Hardware-specific optimizations (TPU/GPU)** -- JAX handles device placement. Library code should be device-agnostic.
- **MLOps / Kubernetes / monitoring infrastructure** -- deployment concerns belong in the deployment layer, not the computation library.
- **HFT latency optimization** -- Python/JAX is not the right stack for microsecond latency. WAX-ML serves research and medium-frequency applications.

---

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Optax Documentation](https://optax.readthedocs.io/)
- [WAX-ML Paper](https://arxiv.org/abs/2106.06110)
