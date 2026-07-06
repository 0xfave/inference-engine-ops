# PRD: inference-engine-ops

## Problem statement

Building an LLM inference server that goes beyond "call a model, return output" — implementing the production serving mechanisms (batching, caching, memory management) that separate a working demo from a system that holds up under concurrent load, and measuring the impact of each one honestly.

## Goal

One finished, benchmarked, Rust-native inference server. Depth over breadth — a small number of components built and measured properly beats a long feature list half-done.

## Non-goals

Explicitly out of scope for v1. Each of these is a legitimate feature of a real serving platform, but adding them here would trade a finishable, defensible project for an unfinished, unconvincing one.

- Multi-model serving / dynamic model routing
- SGLang / TensorRT-LLM integration (Candle or llama.cpp only)
- Autoscaling, Kubernetes
- Billing simulation, usage dashboards, multi-tenancy
- A/B testing between models, canary deploys
- Agent orchestration / multi-agent systems

If any of these get built, they're a v2 project, not scope creep into v1.

## Success criteria (per milestone)

| Milestone | Definition of done | Key metric to capture |
|---|---|---|
| M0 — Baseline serving | Single request in, single response out, no batching, no cache. Correct output vs. reference. | Baseline latency (single request) |
| M1 — Naive KV cache | Cache implemented, output still correct vs. M0 reference (no regressions from caching bugs) | Latency vs. M0 (should improve on multi-turn) |
| M2 — Naive batching | Fixed-size batch, waits for full batch before running | Throughput at fixed concurrency, GPU idle time during padding |
| M3 — Continuous batching | Requests join/leave batch mid-flight, no waiting for stragglers | Throughput improvement over M2 (this is your first headline number) |
| M4 — Paged KV cache | Block-based allocation, shared pool, eviction on completion | Max concurrent requests supported at fixed memory budget, vs. M1 |
| M5 — Semantic cache | Redis-backed similarity match live in request path | Cache hit rate, % of model calls avoided |
| M6 — Quantization bench | FP16 vs. GGUF Q4/Q8 compared on identical prompts | Latency/throughput/memory table, no runtime switching needed |
| M7 — Observability + load test | Prometheus metrics + Grafana dashboard + k6/vegeta suite producing throughput curve | Full benchmark suite output, screenshot-able dashboard |
| M8 — Deploy + writeup | Dockerized, running on single GPU EC2 instance, both articles published | Cost per 1k tokens on target instance |

A milestone isn't done until its metric is captured and written down in `docs/BENCHMARKS.md` — a milestone with no number attached doesn't count as finished.

## Sequencing risk

- **M4 may force a rework of M1's cache interface.** Naive cache uses one contiguous buffer per request; paged cache needs a block-table abstraction. Design M1's interface with this in mind (don't hard-code contiguous-buffer assumptions into the scheduler) to avoid a full rewrite at M4.
- **M3 depends on M1 being correct first**, not fast first — batching bugs on top of a broken cache are hard to debug. Don't start M3 until M1's output is verified correct.
- **M5 (semantic cache) is independent** of M2-M4 and can be built in parallel if you want a break from scheduler/cache work.

## Learning checkpoints

Tied to the build sequence, not front-loaded — read just enough for the next milestone, not the whole project at once.

| Before | Learn |
|---|---|
| M0 | Nothing new — model loading + generation loop |
| M1 | What KV cache is and why (why recompute is wasteful across decode steps) |
| M2/M3 | Continuous batching (Anyscale post; vLLM paper's batching section) |
| M4 | vLLM paper's memory management / PagedAttention section — read this one twice before coding |
| M5–M8 | Minimal new theory — mostly applying existing Redis/Prometheus/Docker skills |

## Out of scope for this PRD

Business/product concerns (pricing, multi-tenant auth, marketplace features) — this is an engineering artifact, not a product. Any commercial framing is for the writeup's positioning, not the system's feature set.

## Deliverables

1. Working Rust codebase, milestones M0–M8
2. `docs/BENCHMARKS.md` — sourced, honest numbers per milestone
3. Two articles: dev phase (design decisions) and prod phase (AWS deploy + cost)