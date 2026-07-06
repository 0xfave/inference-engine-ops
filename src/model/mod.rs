// This module wraps Candle (HuggingFace's Rust ML framework) into a ModelExecutor
// that can load a model and generate text one token at a time.
//
// The key concept: transformer models process tokens in sequence. Each step
// produces "logits" (scores for every word in the vocabulary), and we pick
// the next token by sampling from those scores.

use anyhow::{Context, Error, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::llama::{Cache, Config, Llama, LlamaConfig, LlamaEosToks},
};
use hf_hub::HFClient;
use std::io::Write;
use tokenizers::Tokenizer;

// Holds everything needed to run inference on one model.
// Note: Cache is NOT stored here — it's passed as a parameter because
// different batch operations need different cache instances.
pub struct ModelExecutor {
    pub model: Llama,         // The actual transformer with all its weights
    pub tokenizer: Tokenizer, // Converts text ↔ token IDs (numbers the model understands)
    pub config: Config,       // Model architecture params (hidden size, layers, heads, etc.)
    pub device: Device,       // CPU or CUDA — where tensors live
    pub eos_token_id: u32,    // Token ID for end-of-sequence (stops generation)
}

// Downloads a model from HuggingFace Hub and returns a ready-to-use ModelExecutor.
// Uses hf-hub crate which caches files in ~/.cache/huggingface/hub/.
pub async fn load(repo_id: &str) -> Result<ModelExecutor> {
    // repo_id format is "owner/name" e.g. "HuggingFaceTB/SmolLM2-135M-Instruct"
    let (owner, name) = repo_id.split_once('/').context("repo_id must be in owner/name format")?;

    // Create a HuggingFace API client and point it at the model repo
    let api = HFClient::new()?;
    let repo = api.model(owner, name);

    // Download three essential files from the repo
    let config_path = repo.download_file().filename("config.json").send().await?;
    let tokenizer_path = repo.download_file().filename("tokenizer.json").send().await?;
    let model_path = repo.download_file().filename("model.safetensors").send().await?;

    // config.json tells us the model architecture (how many layers, heads, etc.)
    let hf_config: LlamaConfig =
        serde_json::from_str(&std::fs::read_to_string(&config_path)?).context("failed to parse config.json")?;

    // Candle has its own Config type. into_config(false) — false = don't use flash attention
    let config = hf_config.into_config(false);

    // No GPU available, run on CPU with 32-bit floats
    let device = Device::Cpu;
    let dtype = DType::F32;

    // Load model weights from the safetensors file into memory
    // Unsafe because it memory-maps the file (OS-level file mapping)
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device)? };

    // Construct the actual transformer model from weights + config
    let model = Llama::load(vb, &config)?;

    // Load the tokenizer — this converts text ↔ token IDs
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

    // Find the EOS token. Some models list it in config, others we look up from the tokenizer.
    let eos_token_id = match &config.eos_token_id {
        Some(LlamaEosToks::Single(id)) => *id,
        Some(LlamaEosToks::Multiple(ids)) => ids[0],
        None => tokenizer.token_to_id("<|endoftext|>").unwrap_or(2),
    };

    Ok(ModelExecutor { model, tokenizer, config, device, eos_token_id })
}

// BPE tokenizers produce tokens like "Ġyou" that contain special encoding artifacts.
// This helper fixes that by using the tokenizer's decode() which handles merging correctly.
// It tracks what we've already output and only returns the new portion on each call.
pub struct TokenOutputStream {
    tokenizer: Tokenizer,
    prev_len: usize,      // How many characters of decoded output we've already emitted
    all_tokens: Vec<u32>, // All token IDs seen so far (used for context-aware decoding)
}

impl TokenOutputStream {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer, prev_len: 0, all_tokens: Vec::new() }
    }

    // Feed a new generated token. Returns the clean text portion if any is available.
    // None means the token hasn't formed a complete output yet (still buffering).
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        self.all_tokens.push(token);
        // Decode ALL tokens accumulated so far into a single string
        let decoded = self.tokenizer.decode(&self.all_tokens, true).map_err(anyhow::Error::msg)?;

        // If the decoded text got longer, emit only the NEW part
        if decoded.len() > self.prev_len {
            let new = &decoded[self.prev_len..];
            self.prev_len = decoded.len();
            Ok(Some(new.to_string()))
        } else {
            Ok(None)
        }
    }

    // Flush any remaining buffered text
    pub fn decode_rest(&self) -> Result<Option<String>> {
        if self.all_tokens.is_empty() {
            return Ok(None);
        }
        let text = self.tokenizer.decode(&self.all_tokens, true).map_err(anyhow::Error::msg)?;
        Ok(Some(text))
    }
}

// Stats returned by a single generate() call
pub struct GenerateResult {
    pub output: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_time_s: f64,
    pub generation_time_s: f64,
}

impl ModelExecutor {
    // Generates text for ONE prompt. Cache is passed in so the caller manages
    // its lifecycle (important for batching where one cache serves many sequences).
    pub fn generate(
        &mut self,
        cache: &mut Cache,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
    ) -> Result<GenerateResult> {
        // Step 1: Convert prompt text to token IDs the model understands
        let tokens = self.tokenizer.encode(prompt, true).map_err(anyhow::Error::msg)?.get_ids().to_vec();

        // Sampling strategy: how do we pick the next token from probabilities?
        // temperature=0: always pick the most likely token (greedy/argmax)
        // temperature>0: sample, higher = more random
        let sampling = Sampling::All { temperature };
        let mut logits_processor = LogitsProcessor::from_sampling(299792458, sampling);

        // Setup streaming decoder for clean text output
        let mut all_tokens = tokens.clone();
        let mut stream = TokenOutputStream::new(self.tokenizer.clone());
        let prompt_text = self.tokenizer.decode(&tokens, true).map_err(anyhow::Error::msg)?;
        stream.all_tokens = tokens.clone();
        stream.prev_len = prompt_text.len();

        // Step 2: FIRST FORWARD PASS — feed the full prompt
        // This processes all prompt tokens and fills the KV cache for the first time.
        // Without the KV cache, every subsequent step would reprocess these tokens.
        let prompt_start = std::time::Instant::now();
        let input = Tensor::new(all_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0, cache)?;
        let logits = logits.squeeze(0)?;
        let mut next_token = logits_processor.sample(&logits)?;
        let prompt_time = prompt_start.elapsed();
        all_tokens.push(next_token);

        // Emit first token via streaming decoder
        if let Some(t) = stream.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }

        // Step 3: DECODE LOOP — generate remaining tokens one at a time
        // With KV cache enabled, each step only processes ONE new token.
        // The cache stores previous keys/values, so the model "remembers"
        // earlier tokens without re-processing them.
        let gen_start = std::time::Instant::now();
        let mut generated = 1;
        for index in 1..max_tokens {
            // Feed only the single most recent token (not the full sequence)
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, tokens.len() + index, cache)?;
            let logits = logits.squeeze(0)?;

            // Sample next token from the probability distribution
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            generated += 1;

            // Stop if we hit the end-of-sequence token
            if next_token == self.eos_token_id {
                break;
            }

            // Print this token via streaming decoder (handles BPE artifacts)
            if let Some(t) = stream.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let gen_time = gen_start.elapsed();

        // Decode the generated tokens back to clean text for the result
        let output = self.tokenizer.decode(&all_tokens[tokens.len()..], true).map_err(anyhow::Error::msg)?;

        Ok(GenerateResult {
            output,
            prompt_tokens: tokens.len(),
            generated_tokens: generated,
            prompt_time_s: prompt_time.as_secs_f64(),
            generation_time_s: gen_time.as_secs_f64(),
        })
    }

    // Process a prompt and fill the KV cache, without generating any output tokens.
    // This is used by the batcher to prepare all sequences before the shared decode loop.
    pub fn prefill(&self, cache: &mut Cache, prompt: &str) -> Result<Vec<u32>> {
        let tokens = self.tokenizer.encode(prompt, true).map_err(Error::msg)?.get_ids().to_vec();

        // Run forward pass — the model processes all tokens, fills cache
        let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let _logits = self.model.forward(&input, 0, cache)?;

        Ok(tokens)
    }
}
