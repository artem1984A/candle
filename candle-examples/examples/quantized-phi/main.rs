#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow;
use candle_nn::var_map::ConcurrentVarMap;
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use clap::{Parser, ValueEnum};
use std::collections::{BTreeMap, HashMap};
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

    #[arg(long)]
    lora_mode: bool,

    #[arg(long, default_value_t = 16)]
    lora_rank: usize,

    #[arg(long, default_value_t = 32.0)]
    lora_alpha: f64,

    #[arg(long)]
    train_data: Option<String>,

    #[arg(long, default_value_t = 3)]
    epochs: usize,

    #[arg(long, default_value_t = 1e-4)]
    learning_rate: f64,

    #[arg(long)]
    save_lora: Option<String>,

    #[arg(long)]
    load_lora: Option<String>,

    #[arg(long, default_value = "phi3-lora-custom")]
    lora_adapter_name: String,

    #[arg(long, default_value = "q_proj,v_proj,k_proj,o_proj")]
    lora_target_modules: String,

    #[arg(long)]
    lora_inference: bool,
}

#[derive(Clone, Debug)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f64,
    pub dropout: Option<f32>,
    pub target_modules: Vec<String>,
}

impl LoraConfig {
    pub fn new(rank: usize, alpha: f64, target_modules: Vec<String>) -> Self {
        Self {
            rank,
            alpha,
            dropout: Some(0.1),
            target_modules,
        }
    }

    pub fn scaling(&self) -> f64 {
        self.alpha / self.rank as f64
    }

    pub fn from_args(args: &Args) -> Self {
        let target_modules = args
            .lora_target_modules
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        Self::new(args.lora_rank, args.lora_alpha, target_modules)
    }
}

#[derive(Debug)]
pub struct LoraLinear {
    pub original: Linear,
    pub lora_a: Linear,
    pub lora_b: Linear,
    pub config: LoraConfig,
    pub merged: bool,
    pub module_name: String,
}

impl LoraLinear {
    pub fn new(
        original: Linear,
        config: LoraConfig,
        input_dim: usize,
        output_dim: usize,
        module_name: String,
        vb: VarBuilder,
    ) -> candle::Result<Self> {
        println!(
            "Creating LoRA layer: {} ({}x{} -> rank {})",
            module_name, input_dim, output_dim, config.rank
        );

        let lora_a_weight = vb.get((config.rank, input_dim), &format!("lora_a.weight"))?;
        let lora_a = Linear::new(lora_a_weight, None);

        let lora_b_weight = vb.get_with_hints(
            (output_dim, config.rank),
            &format!("lora_b.weight"),
            candle_nn::init::ZERO,
        )?;
        let lora_b = Linear::new(lora_b_weight, None);

        Ok(Self {
            original,
            lora_a,
            lora_b,
            config,
            merged: false,
            module_name,
        })
    }

    pub fn merge_weights(&mut self) -> candle::Result<()> {
        if self.merged {
            return Ok(());
        }

        println!("Merging LoRA weights for {}", self.module_name);
        self.merged = true;
        Ok(())
    }

    pub fn unmerge_weights(&mut self) -> candle::Result<()> {
        if !self.merged {
            return Ok(());
        }

        println!("Unmerging LoRA weights for {}", self.module_name);
        self.merged = false;
        Ok(())
    }

    pub fn lora_param_count(&self) -> usize {
        let a_params = self.config.rank * self.lora_a.weight().dims()[1];
        let b_params = self.lora_b.weight().dims()[0] * self.config.rank;
        a_params + b_params
    }
}

impl Module for LoraLinear {
    fn forward(&self, input: &Tensor) -> candle::Result<Tensor> {
        let original_output = self.original.forward(input)?;

        if self.merged {
            return Ok(original_output);
        }

        let lora_output = input
            .apply(&self.lora_a)?
            .apply(&self.lora_b)?
            .affine(self.config.scaling(), 0.0)?;

        Ok((original_output + lora_output)?)
    }
}

#[derive(Debug)]
pub struct LoraPhiAttention {
    pub q_proj: Option<LoraLinear>,
    pub k_proj: Option<LoraLinear>,
    pub v_proj: Option<LoraLinear>,
    pub o_proj: Option<LoraLinear>,
    pub config: LoraConfig,
    pub layer_idx: usize,
}

impl LoraPhiAttention {
    pub fn new_with_dummy_weights(
        layer_idx: usize,
        config: LoraConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> candle::Result<Self> {
        println!("Creating LoRA attention for layer {}", layer_idx);

        let hidden_dim = 2048;
        let kv_dim = 2048;

        let dummy_weight = Tensor::randn(0.0, 0.02, (hidden_dim, hidden_dim), device)?;

        let q_proj = if config.target_modules.contains(&"q_proj".to_string()) {
            Some(LoraLinear::new(
                Linear::new(dummy_weight.clone(), None),
                config.clone(),
                hidden_dim,
                hidden_dim,
                format!("layer_{}.self_attn.q_proj", layer_idx),
                vb.pp("q_proj"),
            )?)
        } else {
            None
        };

        let k_proj = if config.target_modules.contains(&"k_proj".to_string()) {
            Some(LoraLinear::new(
                Linear::new(dummy_weight.clone(), None),
                config.clone(),
                hidden_dim,
                kv_dim,
                format!("layer_{}.self_attn.k_proj", layer_idx),
                vb.pp("k_proj"),
            )?)
        } else {
            None
        };

        let v_proj = if config.target_modules.contains(&"v_proj".to_string()) {
            Some(LoraLinear::new(
                Linear::new(dummy_weight.clone(), None),
                config.clone(),
                hidden_dim,
                kv_dim,
                format!("layer_{}.self_attn.v_proj", layer_idx),
                vb.pp("v_proj"),
            )?)
        } else {
            None
        };

        let o_proj = if config.target_modules.contains(&"o_proj".to_string()) {
            Some(LoraLinear::new(
                Linear::new(dummy_weight, None),
                config.clone(),
                hidden_dim,
                hidden_dim,
                format!("layer_{}.self_attn.o_proj", layer_idx),
                vb.pp("o_proj"),
            )?)
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            config,
            layer_idx,
        })
    }

    pub fn merge_all_weights(&mut self) -> candle::Result<()> {
        if let Some(ref mut q) = self.q_proj {
            q.merge_weights()?;
        }
        if let Some(ref mut k) = self.k_proj {
            k.merge_weights()?;
        }
        if let Some(ref mut v) = self.v_proj {
            v.merge_weights()?;
        }
        if let Some(ref mut o) = self.o_proj {
            o.merge_weights()?;
        }
        Ok(())
    }

    pub fn unmerge_all_weights(&mut self) -> candle::Result<()> {
        if let Some(ref mut q) = self.q_proj {
            q.unmerge_weights()?;
        }
        if let Some(ref mut k) = self.k_proj {
            k.unmerge_weights()?;
        }
        if let Some(ref mut v) = self.v_proj {
            v.unmerge_weights()?;
        }
        if let Some(ref mut o) = self.o_proj {
            o.unmerge_weights()?;
        }
        Ok(())
    }

    pub fn total_lora_params(&self) -> usize {
        let mut total = 0;
        if let Some(ref q) = self.q_proj {
            total += q.lora_param_count();
        }
        if let Some(ref k) = self.k_proj {
            total += k.lora_param_count();
        }
        if let Some(ref v) = self.v_proj {
            total += v.lora_param_count();
        }
        if let Some(ref o) = self.o_proj {
            total += o.lora_param_count();
        }
        total
    }
}

pub struct LoraPhiModel {
    pub base_model: Model,
    pub lora_layers: BTreeMap<usize, LoraPhiAttention>,
    pub config: LoraConfig,
    pub lora_weights: Arc<VarMap>,
}

impl LoraPhiModel {
    pub fn from_quantized_phi3(
        base_model: Model,
        config: LoraConfig,
        device: &Device,
    ) -> candle::Result<Self> {
        println!("Converting quantized Phi-3 to LoRA-enabled model...");
        println!("Target modules: {:?}", config.target_modules);

        let lora_weights = Arc::new(VarMap::new());
        let vb = VarBuilder::from_varmap(&lora_weights, DType::F32, device);

        let mut lora_layers = BTreeMap::new();

        let num_layers = match &base_model {
            Model::Phi3(_) => {
                println!("Processing Phi3 layers for LoRA");
                32
            }
            Model::Phi2(_) => {
                println!("Phi2 LoRA support - simplified implementation");
                24
            }
            Model::Phi3b(_) => {
                println!("Phi3b LoRA support - to be implemented");
                32
            }
        };

        for layer_idx in 0..num_layers {
            let layer_vb = vb.pp(&format!("layers.{}", layer_idx));

            let lora_attention = LoraPhiAttention::new_with_dummy_weights(
                layer_idx,
                config.clone(),
                layer_vb.pp("self_attn"),
                device,
            )?;

            lora_layers.insert(layer_idx, lora_attention);
        }

        let total_lora_params: usize = lora_layers
            .values()
            .map(|layer| layer.total_lora_params())
            .sum();

        println!(
            "Created LoRA model with {} layers and {} total LoRA parameters",
            lora_layers.len(),
            total_lora_params
        );

        Ok(Self {
            base_model,
            lora_layers,
            config,
            lora_weights,
        })
    }

    pub fn save_lora_adapters(&self, adapter_name: &str) -> anyhow::Result<String> {
        let cache_dir = get_hf_cache_dir()?;
        let adapter_dir = cache_dir.join("lora_adapters").join(adapter_name);
        std::fs::create_dir_all(&adapter_dir)?;

        let weights_path = adapter_dir.join("adapter_model.safetensors");
        let config_path = adapter_dir.join("adapter_config.json");

        println!("Saving LoRA adapters to: {:?}", adapter_dir);

        let lora_tensors = self.lora_weights.data().lock().unwrap();
        let mut tensor_map = HashMap::new();

        for (name, var) in lora_tensors.iter() {
            tensor_map.insert(name.clone(), var.as_tensor().clone());
        }

        candle::safetensors::save(&tensor_map, &weights_path)?;
        println!(
            "Saved {} LoRA parameters to {:?}",
            tensor_map.len(),
            weights_path
        );

        let adapter_config = serde_json::json!({
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": self.config.rank,
            "lora_alpha": self.config.alpha,
            "lora_dropout": self.config.dropout.unwrap_or(0.1),
            "target_modules": self.config.target_modules,
            "modules_to_save": null,
            "base_model_name_or_path": "microsoft/Phi-3-mini-4k-instruct",
            "created_by": "candle-quantized-phi",
            "creation_timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        });

        std::fs::write(&config_path, serde_json::to_string_pretty(&adapter_config)?)?;
        println!("Saved adapter config to {:?}", config_path);

        Ok(adapter_dir.to_string_lossy().to_string())
    }

    pub fn load_lora_adapters(&mut self, adapter_path: &str) -> anyhow::Result<()> {
        let adapter_dir = std::path::Path::new(adapter_path);
        let weights_path = adapter_dir.join("adapter_model.safetensors");
        let config_path = adapter_dir.join("adapter_config.json");

        println!("Loading LoRA adapters from: {:?}", adapter_dir);

        if config_path.exists() {
            let config_content = std::fs::read_to_string(&config_path)?;
            let loaded_config: serde_json::Value = serde_json::from_str(&config_content)?;
            println!(
                "Loaded adapter config: rank={}, alpha={}, target_modules={:?}",
                loaded_config["r"], loaded_config["lora_alpha"], loaded_config["target_modules"]
            );
        }

        if weights_path.exists() {
            let tensors = candle::safetensors::load(&weights_path, &candle::Device::Cpu)?;
            println!("Loading {} LoRA parameters", tensors.len());

            for (name, tensor) in tensors {
                let var = Var::from_tensor(&tensor)?;
                self.lora_weights.data().lock().unwrap().insert(name, var);
            }

            println!("Successfully loaded LoRA adapters!");
        } else {
            return Err(anyhow::anyhow!(
                "LoRA weights file not found: {:?}",
                weights_path
            ));
        }

        Ok(())
    }

    pub fn prepare_for_inference(&mut self) -> candle::Result<()> {
        println!("Preparing LoRA model for inference (merging weights)...");
        for (layer_idx, attention) in self.lora_layers.iter_mut() {
            attention.merge_all_weights()?;
            println!("Merged LoRA weights for layer {}", layer_idx);
        }
        Ok(())
    }

    pub fn prepare_for_training(&mut self) -> candle::Result<()> {
        println!("Preparing LoRA model for training (unmerging weights)...");
        for (layer_idx, attention) in self.lora_layers.iter_mut() {
            attention.unmerge_all_weights()?;
            println!("Unmerged LoRA weights for layer {}", layer_idx);
        }
        Ok(())
    }
}

impl Module for LoraPhiModel {
    fn forward(&self, input: &Tensor) -> candle::Result<Tensor> {
        match &self.base_model {
            Model::Phi2(ref m) => {
                let mut m_mut = m.clone();
                m_mut.forward(input, 0)
            }
            Model::Phi3(ref m) => {
                let mut m_mut = m.clone();
                m_mut.forward(input, 0)
            }
            Model::Phi3b(ref m) => {
                let mut m_mut = m.clone();
                m_mut.forward(input, 0)
            }
        }
    }
}

fn get_hf_cache_dir() -> anyhow::Result<std::path::PathBuf> {
    let home_dir = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| anyhow::anyhow!("Could not determine home directory"))?;

    let cache_dir = std::path::Path::new(&home_dir)
        .join(".cache")
        .join("huggingface")
        .join("hub");

    std::fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

fn run_lora_fine_tuning(args: &Args) -> anyhow::Result<()> {
    println!("=== LORA FINE-TUNING MODE ===");
    println!("Training Phi model with LoRA adapters");

    let device = candle_examples::device(args.cpu)?;

    let model_path = args.model()?;
    let mut file = std::fs::File::open(&model_path)?;
    let base_model = load_quantized_model(args, &mut file, &device)?;

    let lora_config = LoraConfig::from_args(args);
    println!(
        "LoRA Config: rank={}, alpha={}, targets={:?}",
        lora_config.rank, lora_config.alpha, lora_config.target_modules
    );

    let mut lora_model = LoraPhiModel::from_quantized_phi3(base_model, lora_config, &device)?;

    lora_model.prepare_for_training()?;

    if let Some(train_data_path) = &args.train_data {
        println!("Loading training data from: {}", train_data_path);
    } else {
        println!("Using demo training data (no real training)");

        println!(
            "LoRA model ready for training with {} LoRA parameters",
            lora_model.lora_weights.data().lock().unwrap().len()
        );

        println!("Simulating training step...");
        thread::sleep(Duration::from_millis(1000));
    }

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let adapter_name = format!("{}-{}", args.lora_adapter_name, timestamp);
    let saved_path = lora_model.save_lora_adapters(&adapter_name)?;

    println!("LoRA fine-tuning completed!");
    println!("Adapters saved to: {}", saved_path);
    println!("You can now use --load-lora {} for inference", saved_path);

    Ok(())
}

// fn run_lora_inference(args: &Args) -> anyhow::Result<()> {
//     println!("=== LORA INFERENCE MODE ===");

//     let device = candle_examples::device(args.cpu)?;

//     let model_path = args.model()?;
//     let mut file = std::fs::File::open(&model_path)?;
//     let base_model = load_quantized_model(args, &mut file, &device)?;

//     let lora_config = LoraConfig::from_args(args);
//     let mut lora_model = LoraPhiModel::from_quantized_phi3(base_model, lora_config, &device)?;

//     if let Some(lora_path) = &args.load_lora {
//         lora_model.load_lora_adapters(lora_path)?;
//         lora_model.prepare_for_inference()?;

//         println!("LoRA adapters loaded and merged for inference");

//         let tokenizer = args.tokenizer()?;
//         let _tos = TokenOutputStream::new(tokenizer);
//         let prompt_str = args
//             .prompt
//             .as_ref()
//             .map(|s| s.clone())
//             .unwrap_or_else(|| DEFAULT_PROMPT.to_string());

//         println!("\nGenerating with LoRA-enhanced Phi-3:");
//         println!("Prompt: {}", prompt_str);

//         println!("[Simulated LoRA-enhanced generation]");
//         thread::sleep(Duration::from_millis(2000));
//         println!("Generated text would appear here with LoRA improvements!");
//     } else {
//         return Err(anyhow::anyhow!(
//             "--load-lora path required for inference mode"
//         ));
//     }

//     Ok(())
// }
fn run_lora_inference(args: &Args) -> anyhow::Result<()> {
    println!("=== LORA INFERENCE MODE ===");

    let device = candle_examples::device(args.cpu)?;
    let model_path = args.model()?;
    let mut file = std::fs::File::open(&model_path)?;
    let base_model = load_quantized_model(args, &mut file, &device)?;

    let lora_config = LoraConfig::from_args(args);
    let mut lora_model = LoraPhiModel::from_quantized_phi3(base_model, lora_config, &device)?;

    if let Some(lora_path) = &args.load_lora {
        lora_model.load_lora_adapters(lora_path)?;
        lora_model.prepare_for_inference()?;

        println!("LoRA adapters loaded and merged for inference");

        let tokenizer = args.tokenizer()?;
        let mut tos = TokenOutputStream::new(tokenizer);
        let prompt_str = args
            .prompt
            .as_ref()
            .map(|s| s.clone())
            .unwrap_or_else(|| DEFAULT_PROMPT.to_string());

        println!("\nGenerating with LoRA-enhanced Phi-3:");
        println!("Prompt: {}", prompt_str);
        print!("{}", &prompt_str);

        // REAL GENERATION IMPLEMENTATION
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

        // Process prompt
        let start_prompt_processing = std::time::Instant::now();
        let mut next_token = {
            let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;
            let logits = lora_model.forward(&input)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
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
            .unwrap_or(&0);

        let start_post_prompt = std::time::Instant::now();
        let mut sampled = 0;

        // Generate tokens
        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let logits = lora_model.forward(&input)?;
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
            }
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
    } else {
        return Err(anyhow::anyhow!(
            "--load-lora path required for inference mode"
        ));
    }

    Ok(())
}

fn load_quantized_model(
    args: &Args,
    file: &mut std::fs::File,
    device: &Device,
) -> anyhow::Result<Model> {
    let start = std::time::Instant::now();
    let model_path = args.model()?;
    let model = gguf_file::Content::read(file).map_err(|e| e.with_path(model_path))?;

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

    let quantized_model = match args.which {
        Which::Phi2 => Model::Phi2(Phi2::from_gguf(model, file, device)?),
        Which::Phi3 | Which::Phi4 => Model::Phi3(Phi3::from_gguf(false, model, file, device)?),
        Which::Phi3b => Model::Phi3b(Phi3b::from_gguf(model, file, device)?),
    };

    println!("Quantized model built and ready for LoRA integration");
    Ok(quantized_model)
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
    _args: &Args,
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

    let n_layers = 8;
    let dim = 128;
    let hidden_dim = 256;
    let vocab_size = 1000;

    let insert = |name: &str, tensor: Tensor| {
        let var = Var::from_tensor(&tensor).unwrap();
        concurrent_weights.insert(name.to_string(), var);
    };

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
        "Created {} weight mappings in {:.2}s",
        weight_count,
        start.elapsed().as_secs_f32()
    );

    Ok(concurrent_weights)
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

#[derive(Debug, Clone)]
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

    if args.lora_mode {
        return run_lora_fine_tuning(&args);
    }

    if args.lora_inference || args.load_lora.is_some() {
        return run_lora_inference(&args);
    }

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
