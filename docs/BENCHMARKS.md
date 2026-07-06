# Benchmarks

All benchmarks measured on the specified hardware. Every result includes the exact command, model, and hardware configuration so numbers are reproducible.

**Hardware:**
- CPU: Intel Core i3-5005U @ 2.00GHz (4 cores, AVX2)
- RAM: 11GB
- GPU: None

**Model:** [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- Parameters: 135M
- Precision: FP32 (CPU)
- Format: Safetensors

**Prompt:** `"the best thing about rust is"` (6 tokens)
**Max tokens:** 200

## M0 — Baseline

No batching, no KV cache, no semantic cache. Single request processed synchronously. Every decode step re-processes the full growing sequence.

| Metric | Value |
|---|---|
| Prompt processing speed | 30.27 tok/s |
| Generation speed | 1.17 tok/s |
| TTFT (time to first token) | 0.20s |
| Total time | 171.76s |

## M1 — Naive KV cache

KV cache enabled. First forward pass (prompt) fills the cache; subsequent decode steps process only the single new token.

| Metric | Value |
|---|---|
| Prompt processing speed | 26.91 tok/s |
| Generation speed | 9.30 tok/s |
| TTFT (time to first token) | 0.22s |
| Total time | 21.72s |

### Improvement

| Metric | M0 → M1 |
|---|---|---|
| Generation speed | 1.17 → 9.30 tok/s (**7.9x**) |
| Total time | 171.76s → 21.72s (**7.9x faster**) |

## M2 — Naive batching

Naive batched prefill + decode. Processes B=4 sequences concurrently in the same forward pass. Model output is `(B, vocab_size)` — the Llama forward pass internally extracts only the last position's logits.

**Prompt:** `"the best thing about rust is"` (6 tokens each)
**Max tokens:** 200
**Measurements taken with same-input replicates** (all sequences use the identical prompt). This is a best-case benchmark — real traffic mixes different prompts with varying lengths, which adds padding overhead and reduces efficiency.

### B=1 vs B=4 comparison

| Metric | B=1 | B=4 |
|---|---|---|
| Wall clock time | 29.03s | 27.60s |
| Total generated tokens | 200 | 298 |
| Total throughput | 6.95 tok/s | 10.80 tok/s (+55%) |
| Per-sequence speed | 6.95 tok/s | ~2.70 tok/s |
| TTFT | 0.25s | 0.41s |
| Users served in parallel | 1 | 4 |

**What this means in practice:**

- **B=1 serially (4 users):** First user waits 0.25s, finishes at 29s. Last user gets their first token at 87s. All 4 finished by **116s**.
- **B=4 (4 users simultaneously):** All 4 users wait 0.41s for TTFT. Each generates at ~2.70 tok/s. All 4 finished by **28s**.

B=4 serves 4 users in **less wall time** than B=1 serves 1 user, but each individual user experiences slower token rate (2.70 vs 6.95 tok/s). This is the standard throughput-vs-latency tradeoff in batching.

### Caveat: same prompt limitation

These numbers use identical prompts across all B=4 slots. In production, each user sends a different prompt (different lengths, different content), requiring:
- Padding to the longest prompt in the batch (waste)
- Left-padding for causal masking (extra compute)
- More complex KV cache management

M3 (continuous batching) addresses dynamic request scheduling with varying prompt lengths.

### Analysis

Total throughput improved 55% over B=1 (6.95 → 10.80 tok/s), but per-sequence speed dropped from 6.95 to 2.70 tok/s. On a CPU with 4 cores and no GPU, batched matmuls compete for memory bandwidth rather than compute. This is expected — the bottleneck is memory bandwidth, not FLOPs. The real value of batching will show on GPU hardware where compute is the bottleneck.
