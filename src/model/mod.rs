use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig, LlamaEosToks};
use hf_hub::HFClient;
use std::io::Write;
use tokenizers::Tokenizer;

pub struct ModelExecutor {
    pub model: Llama,
    pub tokenizer: Tokenizer,
    pub cache: Cache,
    pub config: Config,
    pub device: Device,
    pub eos_token_id: u32,
}

pub async fn load(repo_id: &str) -> Result<ModelExecutor> {
    let (owner, name) = repo_id.split_once('/').context("repo_id must be in owner/name format")?;

    let api = HFClient::new()?;
    let repo = api.model(owner, name);

    let config_path = repo.download_file().filename("config.json").send().await?;

    let tokenizer_path = repo.download_file().filename("tokenizer.json").send().await?;

    let model_path = repo.download_file().filename("model.safetensors").send().await?;

    let hf_config: LlamaConfig =
        serde_json::from_str(&std::fs::read_to_string(&config_path)?).context("failed to parse config.json")?;

    let config = hf_config.into_config(false);

    let device = Device::Cpu;
    let dtype = DType::F32;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device)? };

    let cache = Cache::new(true, dtype, &config, &device)?;
    let model = Llama::load(vb, &config)?;

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

    let eos_token_id = match &config.eos_token_id {
        Some(LlamaEosToks::Single(id)) => *id,
        Some(LlamaEosToks::Multiple(ids)) => ids[0],
        None => tokenizer.token_to_id("<|endoftext|>").unwrap_or(2),
    };

    Ok(ModelExecutor { model, tokenizer, cache, config, device, eos_token_id })
}

/// Streaming token decoder that handles BPE merge boundaries correctly.
/// Uses `decode` instead of `id_to_token` to get clean text output.
pub struct TokenOutputStream {
    tokenizer: Tokenizer,
    prev_len: usize,
    all_tokens: Vec<u32>,
}

impl TokenOutputStream {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer, prev_len: 0, all_tokens: Vec::new() }
    }

    /// Process a new token and return the clean text if enough context is available.
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        self.all_tokens.push(token);
        let decoded = self
            .tokenizer
            .decode(&self.all_tokens, true)
            .map_err(anyhow::Error::msg)?;

        if decoded.len() > self.prev_len {
            let new = &decoded[self.prev_len..];
            self.prev_len = decoded.len();
            Ok(Some(new.to_string()))
        } else {
            Ok(None)
        }
    }

    /// Flush any remaining partial decode state.
    pub fn decode_rest(&self) -> Result<Option<String>> {
        if self.all_tokens.is_empty() {
            return Ok(None);
        }
        let text = self
            .tokenizer
            .decode(&self.all_tokens, true)
            .map_err(anyhow::Error::msg)?;
        Ok(Some(text))
    }
}

pub struct GenerateResult {
    pub output: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_time_s: f64,
    pub generation_time_s: f64,
}

impl ModelExecutor {
    pub fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f64) -> Result<GenerateResult> {
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();

        let sampling = Sampling::All { temperature };
        let mut logits_processor = LogitsProcessor::from_sampling(299792458, sampling);

        let mut all_tokens = tokens.clone();
        let mut stream = TokenOutputStream::new(self.tokenizer.clone());
        // Pre-load stream with prompt tokens so decode baseline starts after prompt
        let prompt_text = self.tokenizer.decode(&tokens, true).map_err(anyhow::Error::msg)?;
        stream.all_tokens = tokens.clone();
        stream.prev_len = prompt_text.len();

        let prompt_start = std::time::Instant::now();
        let input = Tensor::new(all_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0, &mut self.cache)?;
        let logits = logits.squeeze(0)?;
        let mut next_token = logits_processor.sample(&logits)?;
        let prompt_time = prompt_start.elapsed();
        all_tokens.push(next_token);

        // Emit first token
        if let Some(t) = stream.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }

        let gen_start = std::time::Instant::now();
        let mut generated = 1;
        for index in 1..max_tokens {
            // Subsequent steps: only the new token (cache handles the rest)
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, tokens.len() + index, &mut self.cache)?;
            let logits = logits.squeeze(0)?;

            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            generated += 1;

            if next_token == self.eos_token_id {
                break;
            }

            if let Some(t) = stream.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let gen_time = gen_start.elapsed();

        let output = self
            .tokenizer
            .decode(&all_tokens[tokens.len()..], true)
            .map_err(anyhow::Error::msg)?;

        Ok(GenerateResult {
            output,
            prompt_tokens: tokens.len(),
            generated_tokens: generated,
            prompt_time_s: prompt_time.as_secs_f64(),
            generation_time_s: gen_time.as_secs_f64(),
        })
    }
}
