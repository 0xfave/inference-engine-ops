use clap::Parser;
use inference_engine_ops::model;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "HuggingFaceTB/SmolLM2-135M-Instruct")]
    model: String,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long, default_value_t = 200)]
    max_tokens: usize,

    #[arg(long, default_value_t = 0.7)]
    temperature: f64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let prompt = args.prompt.unwrap_or_else(|| "The capital of France is".to_string());

    println!("Loading model: {}", args.model);
    let mut executor = model::load(&args.model).await?;
    println!("Model loaded. Generating...\n");

    let result = executor.generate(&prompt, args.max_tokens, args.temperature)?;
    println!();

    let prompt_tok_s = result.prompt_tokens as f64 / result.prompt_time_s;
    let gen_tok_s = result.generated_tokens as f64 / result.generation_time_s;

    println!("---");
    println!("Prompt tokens:  {} ({:.2} tok/s)", result.prompt_tokens, prompt_tok_s);
    println!("Generated:      {} tokens ({:.2} tok/s)", result.generated_tokens, gen_tok_s);
    println!("TTFT:           {:.2}s", result.prompt_time_s);
    println!("Total time:     {:.2}s", result.prompt_time_s + result.generation_time_s);

    Ok(())
}
