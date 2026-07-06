// Entry point for the inference engine.
// Two modes:
//   --batch-size 1 (default): single request via ModelExecutor::generate()
//   --batch-size N:           batch requests via NaiveBatcher::run()

use candle_core::DType;
use candle_transformers::models::llama::Cache;
use clap::Parser;
use inference_engine_ops::{model, scheduler::NaiveBatcher};

#[derive(Parser)]
struct Args {
    // HuggingFace model ID in "owner/name" format
    #[arg(long, default_value = "HuggingFaceTB/SmolLM2-135M-Instruct")]
    model: String,

    // Prompt text. If not provided, a default is used.
    #[arg(long)]
    prompt: Option<String>,

    // Maximum number of tokens to generate (not counting the prompt)
    #[arg(long, default_value_t = 200)]
    max_tokens: usize,

    // Sampling temperature. 0 = greedy (always pick most likely).
    // Higher = more random/creative output.
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    // Number of requests to batch together. 1 = single request path.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Load model once. Both single and batch modes share the same loaded model.
    println!("Loading model: {}", args.model);
    let executor = model::load(&args.model).await?;
    println!("Model loaded.\n");

    if args.batch_size > 1 {
        // ── Batch mode ─────────────────────────────────────────────────
        // Create one prompt per batch slot (all using the same prompt text
        // for benchmarking, but they could be different).
        let prompts: Vec<String> = (0..args.batch_size)
            .map(|_i| args.prompt.clone().unwrap_or_else(|| "The capital of France is".to_string()))
            .collect();

        // Run the batch. The batcher owns the executor temporarily.
        let mut batcher = NaiveBatcher::new(executor, args.batch_size);
        let result = batcher.run(&prompts, args.max_tokens, args.temperature)?;

        // Print each sequence's output
        for (i, output) in result.outputs.iter().enumerate() {
            println!("--- Result {} ---", i + 1);
            println!("{}\n", output);
        }

        // Print aggregate batch metrics
        println!("---");
        println!("Total prompt tokens: {}", result.total_prompt_tokens);
        println!("Total generated:     {} tokens", result.total_generated);
        println!("Prefill:             {:.2}s", result.prefill_time_s);
        println!("Decode:              {:.2}s", result.decode_time_s);
    } else {
        // ── Single request mode ────────────────────────────────────────
        let prompt = args.prompt.unwrap_or_else(|| "The capital of France is".to_string());

        // Cache is created per-request (or per-batch, in batch mode).
        // It stores K/V tensors so decode steps don't reprocess the prompt.
        let mut cache = Cache::new(true, DType::F32, &executor.config, &executor.device)?;
        let mut executor = executor;
        let result = executor.generate(&mut cache, &prompt, args.max_tokens, args.temperature)?;
        println!();

        // Print per-request metrics
        let prompt_tok_s = result.prompt_tokens as f64 / result.prompt_time_s;
        let gen_tok_s = result.generated_tokens as f64 / result.generation_time_s;

        println!("---");
        println!("Prompt tokens:  {} ({:.2} tok/s)", result.prompt_tokens, prompt_tok_s);
        println!("Generated:      {} tokens ({:.2} tok/s)", result.generated_tokens, gen_tok_s);
        println!("TTFT:           {:.2}s", result.prompt_time_s);
        println!("Total time:     {:.2}s", result.prompt_time_s + result.generation_time_s);
    }

    Ok(())
}
