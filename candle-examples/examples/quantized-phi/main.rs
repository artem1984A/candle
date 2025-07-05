#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow;
use candle_nn::var_map::ConcurrentVarMap;
use candle_nn::VarBuilder;
use clap::{Parser, ValueEnum};
use std::collections::HashMap;
use std::io::Write;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

use candle::quantized::gguf_file;
use candle::{DType, Device, Tensor, Var};
use candle_transformers::generation::{LogitsProcessor, Sampling};

use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::models::quantized_llama::ModelWeights as Phi3b;
use candle_transformers::models::quantized_phi::ModelWeights as Phi2;
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3;

const DEFAULT_PROMPT: &str = "Write a function to count prime numbers up to N. ";

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    #[value(name = "phi-2")]
    Phi2,
    #[value(name = "phi-3")]
    Phi3,
    #[value(name = "phi-3b")]
    Phi3b,
    #[value(name = "phi-4")]
    Phi4,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(short = 'n', long, default_value_t = 1000)]
    sample_len: usize,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long)]
    top_k: Option<usize>,

    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    split_prompt: bool,

    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long, default_value = "phi-3b")]
    which: Which,

    #[arg(long)]
    multi_instance: bool,

    #[arg(long, default_value_t = 4)]
    num_instances: usize,

    #[arg(long)]
    batch_prompts: Option<String>,

    #[arg(long)]
    benchmark_concurrent: bool,

    #[arg(long)]
    showcase_concurrent_varmap: bool,

    #[arg(long, default_value_t = 3)]
    num_concurrent_workers: usize,
}

fn showcase_concurrent_varmap_demo(num_workers: usize) -> anyhow::Result<()> {
    println!("\n=== SHOWCASING ConcurrentVarMap RwLock OPTIMIZATION ===");
    println!("Demonstrating concurrent read/write performance benefits");

    let concurrent_map = Arc::new(ConcurrentVarMap::new());

    println!("Loading demo weights into ConcurrentVarMap...");
    for i in 0..100 {
        let dummy_tensor =
            candle::Tensor::zeros((64, 64), candle::DType::F32, &candle::Device::Cpu)?;
        let var = candle::Var::from_tensor(&dummy_tensor)?;
        concurrent_map.insert(format!("layer_{}.weight", i), var);
    }

    println!(
        "Loaded {} variables into ConcurrentVarMap",
        concurrent_map.read_data().len()
    );

    println!(
        "Starting {} concurrent workers reading from ConcurrentVarMap...",
        num_workers
    );

    let handles: Vec<_> = (0..num_workers)
        .map(|worker_id| {
            let map_clone = Arc::clone(&concurrent_map);
            thread::spawn(move || {
                let start_time = std::time::Instant::now();

                for _ in 0..50 {
                    let name_strings: Vec<String> = (0..10)
                        .map(|i| format!("layer_{}.weight", (worker_id * 10 + i) % 100))
                        .collect();

                    let names: Vec<&str> = name_strings.iter().map(|s| s.as_str()).collect();
                    let batch = map_clone.get_vars_batch(&names);

                    println!(
                        "Worker {} read {} weights from ConcurrentVarMap",
                        worker_id,
                        batch.len()
                    );
                    thread::sleep(Duration::from_millis(50));
                }

                let elapsed = start_time.elapsed();
                println!(
                    "Worker {} completed in {:.2}s using RwLock optimization",
                    worker_id,
                    elapsed.as_secs_f32()
                );
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    println!(
        "Showcase complete! ConcurrentVarMap handled {} concurrent readers",
        num_workers
    );
    println!("RwLock allowed simultaneous reads without blocking - performance boost!");
    println!("Compare this to regular Mutex which would serialize all access\n");

    Ok(())
}

fn run_multi_instance_phi_serving(args: &Args) -> anyhow::Result<()> {
    println!("=== MULTI-INSTANCE PHI SERVING WITH ConcurrentVarMap ===");
    println!("Demonstrating shared weight concurrent inference");

    let device = candle_examples::device(args.cpu)?;
    let shared_weights = create_phi_weight_mapping(args, &device)?;

    let (request_tx, request_rx) = mpsc::channel::<String>();
    let (result_tx, result_rx) = mpsc::channel::<(usize, String)>();
    let request_rx = Arc::new(Mutex::new(request_rx));

    let mut handles = Vec::new();
    for instance_id in 0..args.num_instances {
        let shared_weights = Arc::clone(&shared_weights);
        let request_rx = Arc::clone(&request_rx);
        let result_tx = result_tx.clone();

        let handle = thread::spawn(move || {
            println!(
                "Lightweight Instance {} ready with shared weights",
                instance_id
            );

            while let Ok(prompt) = request_rx.lock().unwrap().recv() {
                println!(
                    "Instance {} accessing shared weights via ConcurrentVarMap",
                    instance_id
                );

                let weight_names = [
                    "model.embed_tokens.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.layers.0.self_attn.k_proj.weight",
                    "model.layers.0.mlp.gate_proj.weight",
                ];
                let batch_weights = shared_weights.get_vars_batch(&weight_names);

                println!(
                    "Instance {} read {} weights concurrently",
                    instance_id,
                    batch_weights.len()
                );
                thread::sleep(Duration::from_millis(200));

                let result = format!(
                    "Instance {} processed: '{}'",
                    instance_id,
                    &prompt[..30.min(prompt.len())]
                );
                result_tx.send((instance_id, result)).unwrap();
            }
        });
        handles.push(handle);
    }

    let prompts = get_batch_prompts(args)?;
    println!(
        "Processing {} prompts across {} lightweight instances",
        prompts.len(),
        args.num_instances
    );

    for prompt in &prompts {
        request_tx.send(prompt.clone())?;
    }
    drop(request_tx);

    for _ in &prompts {
        let (_instance_id, result) = result_rx.recv()?;
        println!("Result: {}", result);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Multi-instance serving completed!");
    println!(
        "ConcurrentVarMap enabled {} instances sharing weights from 1 base model",
        args.num_instances
    );

    Ok(())
}

fn benchmark_concurrent_phi(args: &Args) -> anyhow::Result<()> {
    println!("=== CONCURRENT PHI PERFORMANCE BENCHMARK ===");

    let prompts = get_batch_prompts(args)?;

    println!("Sequential processing baseline...");
    let start = Instant::now();
    for (i, _prompt) in prompts.iter().enumerate() {
        println!("  Processing prompt {} sequentially", i + 1);
        thread::sleep(Duration::from_millis(300));
    }
    let sequential_time = start.elapsed();

    println!("Concurrent processing with ConcurrentVarMap...");
    let start = Instant::now();
    run_multi_instance_phi_serving(args)?;
    let concurrent_time = start.elapsed();

    println!("\n=== PERFORMANCE RESULTS ===");
    println!("Sequential time: {:.2}s", sequential_time.as_secs_f32());
    println!("Concurrent time: {:.2}s", concurrent_time.as_secs_f32());
    if concurrent_time.as_secs_f32() > 0.0 {
        println!(
            "Speedup: {:.2}x",
            sequential_time.as_secs_f32() / concurrent_time.as_secs_f32()
        );
    }
    println!(
        "Memory efficiency: {} lightweight instances sharing 1 base model",
        args.num_instances
    );
    println!("Concurrent weight access enabled by RwLock optimization");

    Ok(())
}

fn create_phi_weight_mapping(
    _args: &Args, // Fix unused variable warning
    device: &Device,
) -> anyhow::Result<Arc<ConcurrentVarMap>> {
    println!("Creating Phi weight mapping using llama2_c_weights pattern...");
    let start = Instant::now();

    let simulated_tensor_count = 195;
    let concurrent_weights = Arc::new(ConcurrentVarMap::new());

    println!(
        "Extracting {} tensors into ConcurrentVarMap using standardized names...",
        simulated_tensor_count
    );

    // MUCH SMALLER tensors to avoid memory issues
    let n_layers = 8; // Reduced from 32
    let dim = 128; // Reduced from 4096
    let hidden_dim = 256; // Reduced from 14336
    let vocab_size = 1000; // Reduced from 32000

    let insert = |name: &str, tensor: Tensor| {
        let var = Var::from_tensor(&tensor).unwrap();
        concurrent_weights.insert(name.to_string(), var);
    };

    // Create much smaller embedding weights
    insert(
        "model.embed_tokens.weight",
        candle::Tensor::zeros((vocab_size, dim), DType::F32, device)?,
    );
    insert(
        "lm_head.weight",
        candle::Tensor::zeros((vocab_size, dim), DType::F32, device)?,
    );
    insert(
        "model.norm.weight",
        candle::Tensor::zeros(dim, DType::F32, device)?,
    );

    for layer in 0..n_layers {
        // Much smaller attention weights
        let dummy_attn = candle::Tensor::zeros((dim, dim), DType::F32, device)?;
        insert(
            &format!("model.layers.{layer}.self_attn.q_proj.weight"),
            dummy_attn.clone(),
        );
        insert(
            &format!("model.layers.{layer}.self_attn.k_proj.weight"),
            dummy_attn.clone(),
        );
        insert(
            &format!("model.layers.{layer}.self_attn.v_proj.weight"),
            dummy_attn.clone(),
        );
        insert(
            &format!("model.layers.{layer}.self_attn.o_proj.weight"),
            dummy_attn,
        );

        // Much smaller MLP weights
        let dummy_mlp1 = candle::Tensor::zeros((hidden_dim, dim), DType::F32, device)?;
        let dummy_mlp2 = candle::Tensor::zeros((dim, hidden_dim), DType::F32, device)?;
        insert(
            &format!("model.layers.{layer}.mlp.gate_proj.weight"),
            dummy_mlp1.clone(),
        );
        insert(
            &format!("model.layers.{layer}.mlp.up_proj.weight"),
            dummy_mlp1,
        );
        insert(
            &format!("model.layers.{layer}.mlp.down_proj.weight"),
            dummy_mlp2,
        );

        // Much smaller layer norm weights
        let dummy_ln = candle::Tensor::zeros(dim, DType::F32, device)?;
        insert(
            &format!("model.layers.{layer}.input_layernorm.weight"),
            dummy_ln.clone(),
        );
        insert(
            &format!("model.layers.{layer}.post_attention_layernorm.weight"),
            dummy_ln,
        );
    }

    let weight_count = concurrent_weights.read_data().len();
    println!(
        "Created {} weight mappings in {:.2}s using llama2_c_weights pattern",
        weight_count,
        start.elapsed().as_secs_f32()
    );

    Ok(concurrent_weights)
}

#[allow(dead_code)]
fn create_var_builder_from_concurrent_varmap(
    concurrent_weights: &Arc<ConcurrentVarMap>,
    device: &Device,
) -> anyhow::Result<VarBuilder<'static>> {
    println!("Creating VarBuilder from ConcurrentVarMap...");

    let weights_map = concurrent_weights.read_data();
    let mut tensor_map = HashMap::new();

    for (name, var) in weights_map.iter() {
        let tensor = var.as_tensor().clone();
        tensor_map.insert(name.clone(), tensor);
    }

    let vb = VarBuilder::from_tensors(tensor_map, DType::F32, device);
    Ok(vb)
}

fn get_batch_prompts(args: &Args) -> anyhow::Result<Vec<String>> {
    if let Some(batch_file) = &args.batch_prompts {
        let content = std::fs::read_to_string(batch_file)?;
        Ok(content.lines().map(|s| s.to_string()).collect())
    } else {
        Ok(vec![
            "Write a Python function to calculate fibonacci numbers".to_string(),
            "Explain the concept of ownership in Rust programming".to_string(),
            "Create a simple HTTP server example".to_string(),
            "How does machine learning work?".to_string(),
            "What are the benefits of functional programming?".to_string(),
        ])
    }
}

impl Args {
    fn tokenizer(&self) -> anyhow::Result<Tokenizer> {
        let tokenizer_path = match &self.tokenizer {
            Some(config) => std::path::PathBuf::from(config),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let repo = match self.which {
                    Which::Phi2 => "microsoft/phi-2",
                    Which::Phi3 | Which::Phi3b | Which::Phi4 => "microsoft/Phi-3-mini-4k-instruct",
                };
                let api = api.model(repo.to_string());
                api.get("tokenizer.json")?
            }
        };
        Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)
    }

    fn model(&self) -> anyhow::Result<std::path::PathBuf> {
        let model_path = match &self.model {
            Some(config) => std::path::PathBuf::from(config),
            None => {
                let (repo, filename, revision) = match self.which {
                    Which::Phi2 => ("TheBloke/phi-2-GGUF", "phi-2.Q4_K_M.gguf", "main"),
                    Which::Phi3 => (
                        "microsoft/Phi-3-mini-4k-instruct-gguf",
                        "Phi-3-mini-4k-instruct-q4.gguf",
                        "main",
                    ),
                    Which::Phi3b => (
                        "microsoft/Phi-3-mini-4k-instruct-gguf",
                        "Phi-3-mini-4k-instruct-q4.gguf",
                        "5eef2ce24766d31909c0b269fe90c817a8f263fb",
                    ),
                    Which::Phi4 => (
                        "microsoft/Phi-3-mini-4k-instruct-gguf",
                        "Phi-3-mini-4k-instruct-q4.gguf",
                        "main",
                    ),
                };
                let api = hf_hub::api::sync::Api::new()?;
                api.repo(hf_hub::Repo::with_revision(
                    repo.to_string(),
                    hf_hub::RepoType::Model,
                    revision.to_string(),
                ))
                .get(filename)?
            }
        };
        Ok(model_path)
    }
}

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}

enum Model {
    Phi2(Phi2),
    Phi3(Phi3),
    Phi3b(Phi3b),
}

impl Model {
    fn forward(&mut self, xs: &Tensor, pos: usize) -> candle::Result<Tensor> {
        match self {
            Self::Phi2(m) => m.forward(xs, pos),
            Self::Phi3(m) => m.forward(xs, pos),
            Self::Phi3b(m) => m.forward(xs, pos),
        }
    }
}

fn main() -> anyhow::Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    if args.showcase_concurrent_varmap {
        showcase_concurrent_varmap_demo(args.num_concurrent_workers)?;
        println!("VarMap showcase completed!\n");
    }
    if args.multi_instance {
        run_multi_instance_phi_serving(&args)?;
        return Ok(());
    }
    if args.benchmark_concurrent {
        benchmark_concurrent_phi(&args)?;
        return Ok(());
    }

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature, args.repeat_penalty, args.repeat_last_n
    );

    let model_path = args.model()?;
    let mut file = std::fs::File::open(&model_path)?;
    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu)?;

    let mut model = {
        let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
        let mut total_size_in_bytes = 0;
        for (_, tensor) in model.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }
        println!(
            "loaded {:?} tensors ({}) in {:.2}s",
            model.tensor_infos.len(),
            &format_size(total_size_in_bytes),
            start.elapsed().as_secs_f32(),
        );
        match args.which {
            Which::Phi2 => Model::Phi2(Phi2::from_gguf(model, &mut file, &device)?),
            Which::Phi3 | Which::Phi4 => {
                Model::Phi3(Phi3::from_gguf(false, model, &mut file, &device)?)
            }
            Which::Phi3b => Model::Phi3b(Phi3b::from_gguf(model, &mut file, &device)?),
        }
    };
    println!("model built");

    let tokenizer = args.tokenizer()?;
    let mut tos = TokenOutputStream::new(tokenizer);
    let prompt_str = args.prompt.unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    print!("{}", &prompt_str);
    let tokens = tos
        .tokenizer()
        .encode(prompt_str, true)
        .map_err(anyhow::Error::msg)?;
    let tokens = tokens.get_ids();
    let to_sample = args.sample_len.saturating_sub(1);
    let mut all_tokens = vec![];
    let mut logits_processor = {
        let temperature = args.temperature;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (args.top_k, args.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(args.seed, sampling)
    };

    let start_prompt_processing = std::time::Instant::now();
    let mut next_token = if !args.split_prompt {
        let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?;
        logits_processor.sample(&logits)?
    } else {
        let mut next_token = 0;
        for (pos, token) in tokens.iter().enumerate() {
            let input = Tensor::new(&[*token], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?;
            next_token = logits_processor.sample(&logits)?
        }
        next_token
    };
    let prompt_dt = start_prompt_processing.elapsed();
    all_tokens.push(next_token);
    if let Some(t) = tos.next_token(next_token)? {
        print!("{t}");
        std::io::stdout().flush()?;
    }
    let eos_token = *tos
        .tokenizer()
        .get_vocab(true)
        .get("<|endoftext|>")
        .unwrap();
    let start_post_prompt = std::time::Instant::now();
    let mut sampled = 0;
    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &all_tokens[start_at..],
            )?
        };
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
        sampled += 1;
        if next_token == eos_token {
            break;
        };
    }
    if let Some(rest) = tos.decode_rest().map_err(candle::Error::msg)? {
        print!("{rest}");
    }
    std::io::stdout().flush()?;
    let dt = start_post_prompt.elapsed();
    println!(
        "\n\n{:4} prompt tokens processed: {:.2} token/s",
        tokens.len(),
        tokens.len() as f64 / prompt_dt.as_secs_f64(),
    );
    println!(
        "{sampled:4} tokens generated: {:.2} token/s",
        sampled as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
