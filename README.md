# inference-engine-ops

A Rust-native LLM inference server built from scratch — continuous batching, KV cache management, semantic caching, and quantization benchmarking, with every optimization measured and documented. Not a wrapper around vLLM; an implementation of what vLLM does internally.

**Status:** 🚧 In development — M0 (baseline single-request serving)

## Why this exists

Most portfolio LLM projects call an API and call it a day. This one answers a narrower, harder question: what actually happens between "a request comes in" and "tokens stream out," and how do you make that fast and cheap under concurrent load — without touching the model itself.

## Headline results

*(filled in as milestones complete — see [docs/BENCHMARKS.md](docs/BENCHMARKS.md))*

| Metric | Baseline | Optimized | Source |
|---|---|---|---|
| Throughput (req/sec) | TBD | TBD | `docs/BENCHMARKS.md` |
| Latency (TTFT) | TBD | TBD | `docs/BENCHMARKS.md` |
| GPU cost / 1k tokens | TBD | TBD | `docs/BENCHMARKS.md` |
| Cache hit rate | — | TBD | `docs/BENCHMARKS.md` |

## Architecture

```
Client → Axum HTTP layer → Request validator → Scheduler/Batcher (in-process)
                                                      ↓
                                        Semantic cache check (Redis)
                                              ↓ (miss)
                                        KV cache manager ← → Model executor (Candle)
                                                      ↓
                                        Token stream (SSE) → Client
                                                      ↓
                                        Metrics → Prometheus → Grafana
```

Full design rationale in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Components

| Component | What it does | Design doc |
|---|---|---|
| HTTP layer | OpenAI-compatible `/v1/chat/completions`, streaming via SSE | — |
| Scheduler | Continuous batching — requests join/leave a batch mid-flight | `docs/BATCHING_DESIGN.md` |
| KV cache manager | Naive → paged block-based cache allocation | `docs/KV_CACHE_DESIGN.md` |
| Semantic cache | Redis-backed similarity match, skips redundant model calls | — |
| Quantization bench | FP16 vs GGUF Q4/Q8 comparison, real numbers | `docs/BENCHMARKS.md` |
| Observability | Prometheus (TTFT, TPOT, queue depth, cache hit rate) + Grafana | — |
| Load testing | k6/vegeta concurrent load simulation | `load_test/` |

## Quick start

```bash
# Clone and build
git clone https://github.com/0xfave/inference-engine-ops.git
cd inference-engine-ops
cargo build --release

# Run the server (single model, CPU or CUDA feature flag)
cargo run --release --features cuda -- --model ./models/<model> --port 8080

# Talk to it
curl http://localhost:8080/health
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

## Repo structure

```
inference-engine-ops/
├── src/
│   ├── api/            # Axum routes, request/response types
│   ├── scheduler/       # continuous batching loop
│   ├── kv_cache/        # naive.rs, paged.rs
│   ├── model/           # Candle model executor wrapper
│   ├── cache/           # Redis semantic cache
│   └── metrics/         # Prometheus instrumentation
├── benches/             # quantization + throughput benchmarks
├── load_test/           # k6/vegeta scripts
├── docs/                # architecture + design docs, sourced benchmarks
└── docker/
```

## Tech stack

Rust · Axum · Tokio · Candle · Redis · Prometheus · Grafana · Docker

## Non-goals (v1)

Multi-model serving, SGLang/TensorRT-LLM integration, autoscaling, billing simulation, A/B testing, canary deploys, Kubernetes, agent orchestration. See [docs/PRD.md](docs/PRD.md#non-goals) for rationale.

## Writeups

- Dev phase (architecture, batching/cache design decisions) — *coming soon*
- Prod phase (AWS deploy, cost analysis) — *coming soon*

## License

MIT
