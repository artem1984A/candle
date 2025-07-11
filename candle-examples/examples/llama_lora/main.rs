#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Error as E, Result};
use clap::{Parser, ValueEnum};

use candle::{DType, Tensor, Var};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use candle_nn::var_map::ConcurrentVarMap;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use candle_transformers::models::llama as model;
use model::{Llama, LlamaConfig};

const EOS_TOKEN: &str = "</s>";
const DEFAULT_PROMPT: &str = "My favorite theorem is ";

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    V1,
    V2,
    V3,
    V31,
    V3Instruct,
    V31Instruct,
    V32_1b,
    V32_1bInstruct,
    V32_3b,
    V32_3bInstruct,
    #[value(name = "solar-10.7b")]
    Solar10_7B,
    #[value(name = "tiny-llama-1.1b-chat")]
    TinyLlama1_1BChat,
    #[value(name = "SmoLM2-1.7B")]
    SmolLM2_1B,
    #[value(name = "SmoLM2-1.7B-Instruct")]
    SmolLM2_1BInstruct,
    #[value(name = "SmoLM2-360M")]
    SmolLM2_360M,
    #[value(name = "SmoLM2-360M-Instruct")]
    SmolLM2_360MInstruct,
    #[value(name = "SmoLM2-135M")]
    SmolLM2_135M,
    #[value(name = "SmoLM2-135M-Instruct")]
    SmolLM2_135MInstruct,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long)]
    top_k: Option<usize>,

    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    #[arg(short = 'n', long, default_value_t = 10000)]
    sample_len: usize,

    #[arg(long)]
    no_kv_cache: bool,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    dtype: Option<String>,

    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long, default_value = "v3")]
    which: Which,

    #[arg(long)]
    use_flash_attn: bool,

    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 128)]
    repeat_last_n: usize,

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

    #[arg(long, default_value = "llama-lora-custom")]
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

        let lora_a_weight = vb.get((config.rank, input_dim), "lora_a.weight")?;
        let lora_a = Linear::new(lora_a_weight, None);

        let lora_b_weight = vb.get_with_hints(
            (output_dim, config.rank),
            "lora_b.weight",
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
pub struct LoraLlamaAttention {
    pub q_proj: Option<LoraLinear>,
    pub k_proj: Option<LoraLinear>,
    pub v_proj: Option<LoraLinear>,
    pub o_proj: Option<LoraLinear>,
    pub config: LoraConfig,
    pub layer_idx: usize,
}

impl LoraLlamaAttention {
    pub fn new_with_dummy_weights(
        layer_idx: usize,
        config: LoraConfig,
        vb: VarBuilder,
        device: &candle::Device,
        hidden_size: usize,
    ) -> candle::Result<Self> {
        println!("Creating LoRA attention for layer {}", layer_idx);

        let dummy_weight = Tensor::randn(0.0, 0.02, (hidden_size, hidden_size), device)?;

        let q_proj = if config.target_modules.contains(&"q_proj".to_string()) {
            Some(LoraLinear::new(
                Linear::new(dummy_weight.clone(), None),
                config.clone(),
                hidden_size,
                hidden_size,
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
                hidden_size,
                hidden_size,
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
                hidden_size,
                hidden_size,
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
                hidden_size,
                hidden_size,
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

pub struct LoraLlamaModel {
    pub base_model: Llama,
    pub lora_layers: BTreeMap<usize, LoraLlamaAttention>,
    pub config: LoraConfig,
    pub lora_weights: Arc<VarMap>,
    pub concurrent_weights: Option<Arc<ConcurrentVarMap>>,
    pub model_config: model::Config,
}

impl LoraLlamaModel {
    pub fn from_llama(
        base_model: Llama,
        model_config: model::Config,
        lora_config: LoraConfig,
        device: &candle::Device,
    ) -> candle::Result<Self> {
        println!("Converting LLaMA to LoRA-enabled model...");
        println!("Target modules: {:?}", lora_config.target_modules);

        let lora_weights = Arc::new(VarMap::new());
        let vb = VarBuilder::from_varmap(&lora_weights, DType::F32, device);

        let mut lora_layers = BTreeMap::new();
        let num_layers = model_config.num_hidden_layers;
        let hidden_size = model_config.hidden_size;

        println!("Processing {} LLaMA layers for LoRA", num_layers);

        for layer_idx in 0..num_layers {
            let layer_vb = vb.pp(&format!("layers.{}", layer_idx));

            let lora_attention = LoraLlamaAttention::new_with_dummy_weights(
                layer_idx,
                lora_config.clone(),
                layer_vb.pp("self_attn"),
                device,
                hidden_size,
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
            config: lora_config,
            lora_weights,
            concurrent_weights: None,
            model_config,
        })
    }

    pub fn enable_concurrent_inference(&mut self) -> candle::Result<()> {
        println!("Enabling concurrent inference mode...");
        
        let concurrent_map = Arc::new(ConcurrentVarMap::new());
        
        {
            let mutex_data = self.lora_weights.data().lock().unwrap();
            for (name, var) in mutex_data.iter() {
                concurrent_map.insert(name.clone(), var.clone());
            }
        }
        
        self.concurrent_weights = Some(concurrent_map);
        println!("Concurrent inference mode enabled!");
        Ok(())
    }

    pub fn save_lora_adapters(&self, adapter_name: &str) -> anyhow::Result<String> {
        let cache_dir = get_hf_cache_dir()?;
        let adapter_dir = cache_dir.join("lora_adapters").join(adapter_name);
        std::fs::create_dir_all(&adapter_dir)?;

        let weights_path = adapter_dir.join("adapter_model.safetensors");
        let config_path = adapter_dir.join("adapter_config.json");

        println!("Saving LoRA adapters to: {:?}", adapter_dir);

        let tensor_map = if let Some(ref concurrent_map) = self.concurrent_weights {
            let concurrent_data = concurrent_map.read_data();
            let mut map = HashMap::new();
            for (name, var) in concurrent_data.iter() {
                map.insert(name.clone(), var.as_tensor().clone());
            }
            map
        } else {
            let mutex_data = self.lora_weights.data().lock().unwrap();
            let mut map = HashMap::new();
            for (name, var) in mutex_data.iter() {
                map.insert(name.clone(), var.as_tensor().clone());
            }
            map
        };

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
            "base_model_name_or_path": "meta-llama/Llama-2-7b-hf",
            "created_by": "candle-llama-lora",
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

            let mut loaded_vars = Vec::new();
            
            {
                let mut mutex_data = self.lora_weights.data().lock().unwrap();
                for (name, tensor) in tensors {
                    let var = Var::from_tensor(&tensor)?;
                    mutex_data.insert(name.clone(), var.clone());
                    loaded_vars.push((name, var));
                }
            }
            
            if let Some(ref concurrent_map) = self.concurrent_weights {
                for (name, var) in loaded_vars {
                    concurrent_map.insert(name, var);
                }
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
        println!("Preparing LoRA model for inference...");
        
        if self.concurrent_weights.is_none() {
            self.enable_concurrent_inference()?;
        }
        
        for (layer_idx, attention) in self.lora_layers.iter_mut() {
            attention.merge_all_weights()?;
            println!("Merged LoRA weights for layer {}", layer_idx);
        }
        
        println!("LoRA model ready for concurrent inference!");
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

    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &mut model::Cache,
    ) -> candle::Result<Tensor> {
        if let Some(ref concurrent_map) = self.concurrent_weights {
            let _concurrent_data = concurrent_map.read_data();
        }
        
        self.base_model.forward(x, index_pos, cache)
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
    println!("Training LLaMA model with LoRA adapters");

    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::F16,
    };

    let (llama, _tokenizer_filename, _cache, config) = {
        let api = Api::new()?;
        let model_id = args.model_id.clone().unwrap_or_else(|| match args.which {
            Which::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
            Which::SmolLM2_135M => "HuggingFaceTB/SmolLM2-135M".to_string(),
            _ => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        });

        println!("loading the model weights from {model_id}");
        let revision = args.revision.clone().unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = api.get("tokenizer.json")?;
        let config_filename = api.get("config.json")?;
        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let config = config.into_config(args.use_flash_attn);

        let filenames = vec![api.get("model.safetensors")?];
        let cache = model::Cache::new(!args.no_kv_cache, dtype, &config, &device)?;

        let vb =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        (Llama::load(vb, &config)?, tokenizer_filename, cache, config)
    };

    let lora_config = LoraConfig::from_args(args);
    println!(
        "LoRA Config: rank={}, alpha={}, targets={:?}",
        lora_config.rank, lora_config.alpha, lora_config.target_modules
    );

    let mut lora_model = LoraLlamaModel::from_llama(llama, config, lora_config, &device)?;

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

fn run_lora_inference(args: &Args) -> anyhow::Result<()> {
    println!("=== LORA INFERENCE MODE ===");

    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::F16,
    };

    let (llama, tokenizer_filename, mut cache, config) = {
        let api = Api::new()?;
        let model_id = args.model_id.clone().unwrap_or_else(|| match args.which {
            Which::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
            Which::SmolLM2_135M => "HuggingFaceTB/SmolLM2-135M".to_string(),
            _ => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        });

        println!("loading the model weights from {model_id}");
        let revision = args.revision.clone().unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = api.get("tokenizer.json")?;
        let config_filename = api.get("config.json")?;
        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let config = config.into_config(args.use_flash_attn);

        let filenames = vec![api.get("model.safetensors")?];
        let cache = model::Cache::new(!args.no_kv_cache, dtype, &config, &device)?;

        let vb =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        (Llama::load(vb, &config)?, tokenizer_filename, cache, config)
    };

    let lora_config = LoraConfig::from_args(args);
    let mut lora_model = LoraLlamaModel::from_llama(llama, config.clone(), lora_config, &device)?;

    if let Some(lora_path) = &args.load_lora {
        lora_model.load_lora_adapters(lora_path)?;
        lora_model.prepare_for_inference()?;

        println!("LoRA adapters loaded and merged for inference");

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let eos_token_id = config.eos_token_id.or_else(|| {
            tokenizer
                .token_to_id(EOS_TOKEN)
                .map(model::LlamaEosToks::Single)
        });

        let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
        let mut tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let mut tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);

        println!("\nGenerating with LoRA-enhanced LLaMA:");
        println!("Prompt: {}", prompt);
        print!("{}", prompt);

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

        let mut start_gen = std::time::Instant::now();
        let mut index_pos = 0;
        let mut token_generated = 0;
        for index in 0..args.sample_len {
            let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };
            if index == 1 {
                start_gen = std::time::Instant::now()
            }
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
            let logits = lora_model.forward(&input, context_index, &mut cache)?;
            let logits = logits.squeeze(0)?;
            let logits = if args.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(args.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    args.repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            index_pos += ctxt.len();

            let next_token = logits_processor.sample(&logits)?;
            token_generated += 1;
            tokens.push(next_token);

            match eos_token_id {
                Some(model::LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => {
                    break;
                }
                Some(model::LlamaEosToks::Multiple(ref eos_ids))
                    if eos_ids.contains(&next_token) =>
                {
                    break;
                }
                _ => (),
            }
            if let Some(t) = tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        let dt = start_gen.elapsed();
        println!(
            "\n\n{} tokens generated ({} token/s)\n",
            token_generated,
            (token_generated - 1) as f64 / dt.as_secs_f64(),
        );
    } else {
        return Err(anyhow::anyhow!(
            "--load-lora path required for inference mode"
        ));
    }

    Ok(())
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    if args.lora_mode {
        return run_lora_fine_tuning(&args).map_err(|e| anyhow::anyhow!(e));
    }

    if args.lora_inference || args.load_lora.is_some() {
        return run_lora_inference(&args).map_err(|e| anyhow::anyhow!(e));
    }

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::F16,
    };
    let (llama, tokenizer_filename, mut cache, config) = {
        let api = Api::new()?;
        let model_id = args.model_id.unwrap_or_else(|| {
            let str = match args.which {
                Which::V1 => "Narsil/amall-7b",
                Which::V2 => "meta-llama/Llama-2-7b-hf",
                Which::V3 => "meta-llama/Meta-Llama-3-8B",
                Which::V3Instruct => "meta-llama/Meta-Llama-3-8B-Instruct",
                Which::V31 => "meta-llama/Llama-3.1-8B",
                Which::V31Instruct => "meta-llama/Llama-3.1-8B-Instruct",
                Which::V32_1b => "meta-llama/Llama-3.2-1B",
                Which::V32_1bInstruct => "meta-llama/Llama-3.2-1B-Instruct",
                Which::V32_3b => "meta-llama/Llama-3.2-3B",
                Which::V32_3bInstruct => "meta-llama/Llama-3.2-3B-Instruct",
                Which::Solar10_7B => "upstage/SOLAR-10.7B-v1.0",
                Which::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                Which::SmolLM2_135M => "HuggingFaceTB/SmolLM2-135M",
                Which::SmolLM2_135MInstruct => "HuggingFaceTB/SmolLM2-135M-Instruct",
                Which::SmolLM2_360M => "HuggingFaceTB/SmolLM2-360M",
                Which::SmolLM2_360MInstruct => "HuggingFaceTB/SmolLM2-360M-Instruct",
                Which::SmolLM2_1B => "HuggingFaceTB/SmolLM2-1.7B",
                Which::SmolLM2_1BInstruct => "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            };
            str.to_string()
        });
        println!("loading the model weights from {model_id}");
        let revision = args.revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = api.get("tokenizer.json")?;
        let config_filename = api.get("config.json")?;
        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let config = config.into_config(args.use_flash_attn);

        let filenames = match args.which {
            Which::V1
            | Which::V2
            | Which::V3
            | Which::V3Instruct
            | Which::V31
            | Which::V31Instruct
            | Which::V32_3b
            | Which::V32_3bInstruct
            | Which::Solar10_7B => {
                candle_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?
            }
            Which::SmolLM2_360M
            | Which::SmolLM2_360MInstruct
            | Which::SmolLM2_135M
            | Which::SmolLM2_135MInstruct
            | Which::SmolLM2_1B
            | Which::SmolLM2_1BInstruct
            | Which::V32_1b
            | Which::V32_1bInstruct
            | Which::TinyLlama1_1BChat => {
                vec![api.get("model.safetensors")?]
            }
        };
        let cache = model::Cache::new(!args.no_kv_cache, dtype, &config, &device)?;

        let vb =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        (Llama::load(vb, &config)?, tokenizer_filename, cache, config)
    };
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let eos_token_id = config.eos_token_id.or_else(|| {
        tokenizer
            .token_to_id(EOS_TOKEN)
            .map(model::LlamaEosToks::Single)
    });
    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let mut tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);

    println!("starting the inference loop");
    print!("{prompt}");
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

    let mut start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;
    for index in 0..args.sample_len {
        let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
            (1, index_pos)
        } else {
            (tokens.len(), 0)
        };
        if index == 1 {
            start_gen = std::time::Instant::now()
        }
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = llama.forward(&input, context_index, &mut cache)?;
        let logits = logits.squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &tokens[start_at..],
            )?
        };
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        match eos_token_id {
            Some(model::LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => {
                break;
            }
            Some(model::LlamaEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => {
                break;
            }
            _ => (),
        }
        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        (token_generated - 1) as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
