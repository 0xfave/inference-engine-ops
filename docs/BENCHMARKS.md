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
|---|---|
| Generation speed | 1.17 → 9.30 tok/s (**7.9x**) |
| Total time | 171.76s → 21.72s (**7.9x faster**) |
