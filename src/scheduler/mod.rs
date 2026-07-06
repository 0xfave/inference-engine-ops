// Naive batching: process multiple prompts together in one model forward pass.
//
// Why batch? The model's matrix operations work on tensors. A single tensor of
// shape (4, 10) — 4 sequences, 10 tokens each — processes all 40 tokens in one
// call, which is more efficient than 4 separate calls of (1, L1)...(1, L4).
//
// "Naive" means we wait until the batch is full, pad all sequences to the same
// length, and process them together in lockstep. If one sequence finishes early,
// the others keep going but the finished one still pays the compute cost.
// (Continuous batching — M3 — fixes this.)

use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::llama::Cache,
};

use crate::model::ModelExecutor;

pub struct NaiveBatcher {
    pub batch_size: usize,
    pub executor: ModelExecutor,
}

pub struct BatchResult {
    pub outputs: Vec<String>,
    pub total_prompt_tokens: usize,
    pub total_generated: usize,
    pub prefill_time_s: f64,
    pub decode_time_s: f64,
}

impl NaiveBatcher {
    pub fn new(executor: ModelExecutor, batch_size: usize) -> Self {
        Self { executor, batch_size }
    }

    // Takes multiple prompts, processes them as a batch.
    //
    // Phases:
    //   1. TOKENIZE + PAD — all prompts become the same length (left-padded with EOS)
    //   2. PREFILL — single forward pass processes all sequences, fills KV cache
    //   3. FIRST TOKEN — extract logits at each sequence's last position, sample
    //   4. DECODE LOOP — each step feeds one new token per sequence in a batch tensor
    //   5. DECODE — convert token IDs back to text
    pub fn run(&mut self, prompts: &[String], max_tokens: usize, temperature: f64) -> Result<BatchResult> {
        let batch_size = prompts.len();
        // Each batch gets its own Cache. This stores K/V tensors with shape
        // (batch_size, seq_len, n_heads, head_dim) for all sequences together.
        let mut cache = Cache::new(true, DType::F32, &self.executor.config, &self.executor.device)?;
        let eos_id = self.executor.eos_token_id;

        // ── Phase 1: Tokenize ──────────────────────────────────────────────
        // Convert each prompt string into an array of token IDs
        let mut all_token_ids: Vec<Vec<u32>> = Vec::new();
        for prompt in prompts {
            let tokens =
                self.executor.tokenizer.encode(prompt.as_str(), true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
            all_token_ids.push(tokens);
        }

        // ── Phase 2: Left-pad to uniform length ─────────────────────────────
        // Model needs a rectangular tensor (B, max_len). Shorter prompts get
        // padded on the LEFT with EOS tokens so the last real token is always at
        // position (max_len - 1) for every sequence in the batch.
        let max_len = all_token_ids.iter().map(|t| t.len()).max().unwrap_or(0);
        let padded: Vec<u32> = all_token_ids
            .iter()
            .flat_map(|t| {
                let pad_count = max_len - t.len();
                let mut v = vec![eos_id; pad_count]; // padding goes first
                v.extend_from_slice(t); // real tokens go after
                v
            })
            .collect();

        // ── Phase 3: Prefill ────────────────────────────────────────────────
        // Single forward pass processes all B sequences at once.
        // Input shape: (B, max_len)
        // Output shape: (B, max_len, vocab_size) — logits for every position
        let prefill_start = std::time::Instant::now();
        let input = Tensor::from_slice(&padded, (batch_size, max_len), &self.executor.device)?;
        let logits = self.executor.model.forward(&input, 0, &mut cache)?;
        let prefill_time = prefill_start.elapsed();

        // ── Phase 4: First token per sequence ───────────────────────────────
        // Each sequence has its own LogitsProcessor (separate random state).
        // The model returns logits of shape (B, vocab_size) — it internally
        // extracts the last sequence position. We narrow dim 0 to get each
        // sequence's logits individually.
        let sampling = if temperature <= 0. { Sampling::ArgMax } else { Sampling::All { temperature } };
        let mut processors: Vec<LogitsProcessor> =
            (0..batch_size).map(|i| LogitsProcessor::from_sampling(42 + i as u64, sampling.clone())).collect();

        let mut seq_tokens: Vec<Vec<u32>> = all_token_ids.clone();
        let mut alive = vec![true; batch_size]; // Tracks which sequences are still generating
        let mut generated = 0usize;

        for i in 0..batch_size {
            // logits shape is (B, vocab_size) — model already extracts the last
            // position internally. narrow(0, i, 1) picks the i-th sequence,
            // squeeze removes the batch dimension of size 1.
            let seq_logit = logits.narrow(0, i, 1)?.squeeze(0)?;
            let next_token = processors[i].sample(&seq_logit)?;
            seq_tokens[i].push(next_token);
            generated += 1;
            if next_token == eos_id {
                alive[i] = false;
            }
        }

        // Track the latest generated token for each sequence
        let mut next_tokens: Vec<u32> = seq_tokens.iter().map(|t| *t.last().unwrap()).collect();
        let decode_start = std::time::Instant::now();

        // ── Phase 5: Batched decode loop ────────────────────────────────────
        // Each step: run ONE forward pass with ALL B sequences in a fixed-size
        // (B, 1) tensor. Dead sequences feed EOS tokens (won't change output since
        // they've already hit EOS, but keeps the cache's batch dimension stable).
        // Cache stores K/V with shape (B, n_heads, seq_len, head_dim) — changing B
        // mid-generation would panic because cat() requires matching batch dims.
        for step in 1..max_tokens {
            // Always use full batch size: alive tokens for active sequences,
            // EOS filler for dead ones (so cache batch dimension stays constant)
            let batch_next: Vec<u32> =
                (0..batch_size).map(|i| if alive[i] { next_tokens[i] } else { eos_id }).collect();

            if !alive.iter().any(|&a| a) {
                break; // All sequences finished
            }

            // Batched forward pass: (B, 1) tensor — B never changes
            let input = Tensor::from_slice(&batch_next, (batch_size, 1), &self.executor.device)?;
            let logits = self.executor.model.forward(&input, max_len + step, &mut cache)?;

            // logits shape: (B, vocab_size) — model internally extracts last position
            for i in 0..batch_size {
                if !alive[i] {
                    continue;
                }
                let seq_logit = logits.narrow(0, i, 1)?.squeeze(0)?;
                let next_token = processors[i].sample(&seq_logit)?;
                seq_tokens[i].push(next_token);
                next_tokens[i] = next_token;
                generated += 1;
                if next_token == eos_id {
                    alive[i] = false;
                }
            }
        }
        let decode_time = decode_start.elapsed();

        // ── Phase 6: Decode tokens to text ──────────────────────────────────
        let outputs: Vec<String> = seq_tokens
            .iter()
            .map(|tokens| self.executor.tokenizer.decode(tokens, true).map_err(anyhow::Error::msg))
            .collect::<Result<Vec<_>>>()?;

        Ok(BatchResult {
            outputs,
            total_prompt_tokens: all_token_ids.iter().map(|t| t.len()).sum(),
            total_generated: generated,
            prefill_time_s: prefill_time.as_secs_f64(),
            decode_time_s: decode_time.as_secs_f64(),
        })
    }
}
