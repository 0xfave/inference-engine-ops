# Benchmarks

All benchmarks measured on the specified hardware. Every result includes the exact command, model, and hardware configuration so numbers are reproducible.

## M0 — Baseline single-request serving

No batching, no KV cache, no semantic cache. Single request processed synchronously.

**Hardware:**
- CPU: Intel Core i3-5005U @ 2.00GHz (4 cores, AVX2)
- RAM: 11GB (1.8GB available during test)
- GPU: None

**Model:** [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- Parameters: 135M
- Precision: FP32 (CPU)
- Format: Safetensors

**Prompt:** `"the best thing about rust is"` (6 tokens)
**Max tokens:** 200

| Metric | Value |
|---|---|
| Prompt tokens processed | 6 |
| Prompt processing speed | 30.27 tok/s |
| Tokens generated | 200 |
| Generation speed | 1.17 tok/s |
| TTFT (time to first token) | 0.20s |
| Total time | 171.76s |
