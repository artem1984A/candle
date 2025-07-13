#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Error as E, Result};
use candle::{DType, Tensor, Var};
use candle_nn::var_map::ConcurrentVarMap;
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama as model;
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use model::{Llama, LlamaConfig};
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

const EOS_TOKEN: &str = "</s>";
const DEFAULT_PROMPT: &str = "My favorite theorem is ";

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    #[value(name = "v1")]
    V1,
    #[value(name = "v2")]
    V2,
    #[value(name = "v3")]
    V3,
    #[value(name = "v3-instruct")]
    V3Instruct,
    #[value(name = "v3.1")]
    V31,
    #[value(name = "v3.1-instruct")]
    V31Instruct,
    #[value(name = "v3.2-1b")]
    V32_1b,
    #[value(name = "v3.2-1b-instruct")]
    V32_1bInstruct,
    #[value(name = "v3.2-3b")]
    V32_3b,
    #[value(name = "v3.2-3b-instruct")]
    V32_3bInstruct,
    #[value(name = "solar-10.7b")]
    Solar10_7B,
    #[value(name = "tiny-llama-1.1b-chat")]
    TinyLlama1_1BChat,
    #[value(name = "SmoLM2-135M")]
    SmolLM2_135M,
    #[value(name = "SmoLM2-135M-Instruct")]
    SmolLM2_135MInstruct,
    #[value(name = "SmoLM2-360M")]
    SmolLM2_360M,
    #[value(name = "SmoLM2-360M-Instruct")]
    SmolLM2_360MInstruct,
    #[value(name = "SmoLM2-1B")]
    SmolLM2_1B,
    #[value(name = "SmoLM2-1B-Instruct")]
    SmolLM2_1BInstruct,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    cpu: bool,
    #[arg(long)]
    tracing: bool,
    #[arg(long)]
    lora_mode: bool,
    #[arg(long)]
    lora_inference: bool,
    #[arg(long, default_value = "16")]
    lora_rank: usize,
    #[arg(long, default_value = "32.0")]
    lora_alpha: f64,
    #[arg(long, default_value = "q_proj,v_proj,k_proj,o_proj")]
    lora_target_modules: String,
    #[arg(long)]
    load_lora: Option<String>,
    #[arg(long, default_value = "my-lora-adapter")]
    lora_adapter_name: String,
    #[arg(long)]
    train_data: Option<String>,
    #[arg(long)]
    use_flash_attn: bool,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    #[arg(long, default_value_t = 9999)]
    sample_len: usize,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    top_p: Option<f64>,
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
    #[arg(long)]
    no_kv_cache: bool,
    #[arg(long)]
    dtype: Option<String>,
    #[arg(long)]
    model_id: Option<String>,
    #[arg(long)]
    revision: Option<String>,
    #[arg(long, default_value = "tiny-llama-1.1b-chat")]
    which: Which,
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,
    #[arg(long, default_value_t = 128)]
    repeat_last_n: usize,
    #[arg(long)]
    compare_mode: bool,
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
        in_features: usize,
        out_features: usize,
        module_name: String,
        vb: VarBuilder,
    ) -> candle::Result<Self> {
        println!(
            "Creating LoRA layer: {} ({}x{} -> rank {})",
            module_name, out_features, in_features, config.rank
        );
        let lora_a_weight = vb.get_with_hints(
            (config.rank, in_features),
            &format!("{}.lora_a.weight", module_name),
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;
        let lora_a = Linear::new(lora_a_weight, None);
        let lora_b_weight = vb.get_with_hints(
            (out_features, config.rank),
            &format!("{}.lora_b.weight", module_name),
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
            println!("Warning: Weights for {} already merged", self.module_name);
            return Ok(());
        }
        println!("Merging LoRA weights for {}", self.module_name);
        let lora_a_weight = self.lora_a.weight();
        let lora_b_weight = self.lora_b.weight();

        // Ensure dtype consistency
        let original_dtype = self.original.weight().dtype();
        let lora_a_weight = if lora_a_weight.dtype() != original_dtype {
            lora_a_weight.to_dtype(original_dtype)?
        } else {
            lora_a_weight.clone()
        };
        let lora_b_weight = if lora_b_weight.dtype() != original_dtype {
            lora_b_weight.to_dtype(original_dtype)?
        } else {
            lora_b_weight.clone()
        };

        let delta = lora_b_weight.matmul(&lora_a_weight)?;
        let scaled_delta = delta.affine(self.config.scaling(), 0.0)?;
        let original_weight = self.original.weight();
        let merged_weight = (original_weight + scaled_delta)?;
        self.original = Linear::new(merged_weight, self.original.bias().cloned());
        self.merged = true;
        Ok(())
    }

    pub fn unmerge_weights(&mut self) -> candle::Result<()> {
        if !self.merged {
            println!("Warning: Weights for {} not merged", self.module_name);
            return Ok(());
        }
        println!("Unmerging LoRA weights for {}", self.module_name);
        let lora_a_weight = self.lora_a.weight();
        let lora_b_weight = self.lora_b.weight();

        // Ensure dtype consistency
        let original_dtype = self.original.weight().dtype();
        let lora_a_weight = if lora_a_weight.dtype() != original_dtype {
            lora_a_weight.to_dtype(original_dtype)?
        } else {
            lora_a_weight.clone()
        };
        let lora_b_weight = if lora_b_weight.dtype() != original_dtype {
            lora_b_weight.to_dtype(original_dtype)?
        } else {
            lora_b_weight.clone()
        };

        let delta = lora_b_weight.matmul(&lora_a_weight)?;
        let scaled_delta = delta.affine(self.config.scaling(), 0.0)?;
        let current_weight = self.original.weight();
        let original_weight = (current_weight - scaled_delta)?;
        self.original = Linear::new(original_weight, self.original.bias().cloned());
        self.merged = false;
        Ok(())
    }

    pub fn lora_param_count(&self) -> usize {
        let a_params = self.lora_a.weight().elem_count();
        let b_params = self.lora_b.weight().elem_count();
        a_params + b_params
    }

    pub fn debug_weight_change(&self) -> candle::Result<()> {
        let original_norm = self.original.weight().sqr()?.sum_all()?.sqrt()?;

        // Convert to f64 first, then to f32
        let original_norm_f32 = match original_norm.dtype() {
            DType::F64 => original_norm.to_scalar::<f64>()? as f32,
            DType::F32 => original_norm.to_scalar::<f32>()?,
            DType::F16 => original_norm.to_dtype(DType::F32)?.to_scalar::<f32>()?,
            DType::BF16 => original_norm.to_dtype(DType::F32)?.to_scalar::<f32>()?,
            _ => return Err(candle::Error::Msg("Unsupported dtype for norm".to_string())),
        };

        println!(
            "Weight norm for {}: {}",
            self.module_name, original_norm_f32
        );

        // Calculate the LoRA contribution
        let delta = self.lora_b.weight().matmul(&self.lora_a.weight())?;
        let delta_norm = delta.sqr()?.sum_all()?.sqrt()?;

        // Convert to f64 first, then to f32
        let delta_norm_f32 = match delta_norm.dtype() {
            DType::F64 => delta_norm.to_scalar::<f64>()? as f32,
            DType::F32 => delta_norm.to_scalar::<f32>()?,
            DType::F16 => delta_norm.to_dtype(DType::F32)?.to_scalar::<f32>()?,
            DType::BF16 => delta_norm.to_dtype(DType::F32)?.to_scalar::<f32>()?,
            _ => return Err(candle::Error::Msg("Unsupported dtype for norm".to_string())),
        };

        println!(
            "LoRA delta norm: {} ({}% of original)",
            delta_norm_f32,
            (delta_norm_f32 / original_norm_f32) * 100.0
        );
        Ok(())
    }
    pub fn weight(&self) -> &Tensor {
        self.original.weight()
    }
}

impl Module for LoraLinear {
    fn forward(&self, input: &Tensor) -> candle::Result<Tensor> {
        let original_output = self.original.forward(input)?;
        if self.merged {
            return Ok(original_output);
        }

        // Ensure LoRA computation uses the same dtype as input
        let input_dtype = input.dtype();
        let lora_output = if self.lora_a.weight().dtype() != input_dtype {
            let lora_a_weight = self.lora_a.weight().to_dtype(input_dtype)?;
            let lora_b_weight = self.lora_b.weight().to_dtype(input_dtype)?;
            let lora_a = Linear::new(lora_a_weight, None);
            let lora_b = Linear::new(lora_b_weight, None);
            input
                .apply(&lora_a)?
                .apply(&lora_b)?
                .affine(self.config.scaling(), 0.0)?
        } else {
            input
                .apply(&self.lora_a)?
                .apply(&self.lora_b)?
                .affine(self.config.scaling(), 0.0)?
        };

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

    pub fn is_merged(&self) -> bool {
        let mut all_merged = true;
        if let Some(ref q) = self.q_proj {
            all_merged &= q.merged;
        }
        if let Some(ref k) = self.k_proj {
            all_merged &= k.merged;
        }
        if let Some(ref v) = self.v_proj {
            all_merged &= v.merged;
        }
        if let Some(ref o) = self.o_proj {
            all_merged &= o.merged;
        }
        all_merged
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

    pub fn get_lora_param_count(&self) -> usize {
        let mut count = 0;
        if let Some(ref q) = self.q_proj {
            count += q.lora_param_count();
        }
        if let Some(ref k) = self.k_proj {
            count += k.lora_param_count();
        }
        if let Some(ref v) = self.v_proj {
            count += v.lora_param_count();
        }
        if let Some(ref o) = self.o_proj {
            count += o.lora_param_count();
        }
        count
    }
}

pub struct LoraLlamaModel {
    pub base_model: Llama,
    pub lora_layers: BTreeMap<usize, LoraLlamaAttention>,
    pub config: LoraConfig,
    pub lora_weights: Arc<VarMap>,
    pub concurrent_weights: Option<Arc<ConcurrentVarMap>>,
    pub model_config: model::Config,
    pub device: candle::Device,
}

impl LoraLlamaModel {
    pub fn from_llama(
        base_model: Llama,
        model_config: model::Config,
        config: LoraConfig,
        device: &candle::Device,
    ) -> anyhow::Result<Self> {
        println!("Converting LLaMA to LoRA-enabled model...");
        println!("Target modules: {:?}", config.target_modules);
        let lora_weights = Arc::new(VarMap::new());
        // Get dtype from the base model config or use F32 as default for LoRA weights
        let dtype = DType::F32; // LoRA weights are typically F32 even if base model is F16
        let vb = VarBuilder::from_varmap(&lora_weights, dtype, device);
        let num_layers = model_config.num_hidden_layers;
        let hidden_size = model_config.hidden_size;
        println!("Processing {} LLaMA layers for LoRA", num_layers);
        let mut lora_layers = BTreeMap::new();
        for layer_idx in 0..num_layers {
            let layer_vb = vb.pp(format!("layers.{}", layer_idx));
            let attention_vb = layer_vb.pp("self_attn");
            let lora_attention = LoraLlamaAttention::new_with_dummy_weights(
                layer_idx,
                config.clone(),
                attention_vb,
                device,
                hidden_size,
            )?;
            lora_layers.insert(layer_idx, lora_attention);
        }
        let total_lora_params: usize = lora_layers
            .values()
            .map(|attn| attn.get_lora_param_count())
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
            concurrent_weights: None,
            model_config,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &candle::Device {
        &self.device
    }

    pub fn enable_concurrent_inference(&mut self) -> candle::Result<()> {
        println!("Enabling concurrent inference mode...");
        let concurrent_map = Arc::new(ConcurrentVarMap::new());
        {
            let locked_weights = self.lora_weights.data().lock().unwrap();
            for (name, var) in locked_weights.iter() {
                concurrent_map.insert(name.clone(), var.clone());
            }
        }
        self.concurrent_weights = Some(concurrent_map);
        println!("Concurrent inference mode enabled!");
        Ok(())
    }

    pub fn prepare_for_training(&mut self) -> candle::Result<()> {
        println!("Preparing LoRA model for training...");
        for (_, attention) in self.lora_layers.iter_mut() {
            attention.unmerge_all_weights()?;
        }
        println!(
            "LoRA model ready for training with {} parameters",
            self.lora_weights.data().lock().unwrap().len()
        );
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
        println!("LoRA model ready for inference!");
        println!("All LoRA weights have been merged into the base model weights.");
        Ok(())
    }

    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &mut model::Cache,
    ) -> candle::Result<Tensor> {
        self.base_model.forward(x, index_pos, cache)
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
            let tensors = candle::safetensors::load(&weights_path, &self.device)?;
            println!("Loading {} LoRA parameters", tensors.len());
            for (name, tensor) in tensors.iter() {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() >= 5 && parts[0] == "layers" {
                    if let Ok(layer_idx) = parts[1].parse::<usize>() {
                        if let Some(lora_attn) = self.lora_layers.get_mut(&layer_idx) {
                            let module_name = parts[3]; // q_proj, k_proj, etc.
                            let weight_type = parts[4]; // lora_a, lora_b
                            match (module_name, weight_type) {
                                ("q_proj", "lora_a") => {
                                    if let Some(ref mut q) = lora_attn.q_proj {
                                        q.lora_a = Linear::new(tensor.clone(), None);
                                    }
                                }
                                ("q_proj", "lora_b") => {
                                    if let Some(ref mut q) = lora_attn.q_proj {
                                        q.lora_b = Linear::new(tensor.clone(), None);
                                    }
                                }
                                ("k_proj", "lora_a") => {
                                    if let Some(ref mut k) = lora_attn.k_proj {
                                        k.lora_a = Linear::new(tensor.clone(), None);
                                    }
                                }
                                ("k_proj", "lora_b") => {
                                    if let Some(ref mut k) = lora_attn.k_proj {
                                        k.lora_b = Linear::new(tensor.clone(), None);
                                    }
                                }
                                ("v_proj", "lora_a") => {
                                    if let Some(ref mut v) = lora_attn.v_proj {
                                        v.lora_a = Linear::new(tensor.clone(), None);
                                    }
                                }
                                ("v_proj", "lora_b") => {
                                    if let Some(ref mut v) = lora_attn.v_proj {
                                        v.lora_b = Linear::new(tensor.clone(), None);
                                    }
                                }
                                ("o_proj", "lora_a") => {
                                    if let Some(ref mut o) = lora_attn.o_proj {
                                        o.lora_a = Linear::new(tensor.clone(), None);
                                    }
                                }
                                ("o_proj", "lora_b") => {
                                    if let Some(ref mut o) = lora_attn.o_proj {
                                        o.lora_b = Linear::new(tensor.clone(), None);
                                    }
                                }
                                _ => {
                                    println!(
                                        "Unknown module/weight type: {}/{}",
                                        module_name, weight_type
                                    );
                                }
                            }
                        }
                    }
                }
            }
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
            println!("Note: LoRA weights loaded but not yet merged. Call prepare_for_inference() to merge.");
        } else {
            return Err(anyhow::anyhow!(
                "LoRA weights file not found: {:?}",
                weights_path
            ));
        }
        Ok(())
    }

    pub fn save_lora_adapters(&self, adapter_name: &str) -> anyhow::Result<String> {
        let save_dir = std::path::Path::new(&std::env::var("HOME")?)
            .join(".cache/huggingface/hub/lora_adapters")
            .join(adapter_name);
        std::fs::create_dir_all(&save_dir)?;
        let mut tensors = HashMap::new();
        for (layer_idx, attention) in &self.lora_layers {
            if let Some(ref q) = attention.q_proj {
                tensors.insert(
                    format!(
                        "layers.{}.self_attn.q_proj.lora_a.weight", // Remove "_lora" - change to match inference expectation
                        layer_idx
                    ),
                    q.lora_a.weight().clone(),
                );
                tensors.insert(
                    format!(
                        "layers.{}.self_attn.q_proj.lora_b.weight", // Remove "_lora" - change to match inference expectation
                        layer_idx
                    ),
                    q.lora_b.weight().clone(),
                );
            }
            if let Some(ref k) = attention.k_proj {
                tensors.insert(
                    format!(
                        "layers.{}.self_attn.k_proj.lora_a.weight", // Remove "_lora"
                        layer_idx
                    ),
                    k.lora_a.weight().clone(),
                );
                tensors.insert(
                    format!(
                        "layers.{}.self_attn.k_proj.lora_b.weight", // Remove "_lora"
                        layer_idx
                    ),
                    k.lora_b.weight().clone(),
                );
            }
            if let Some(ref v) = attention.v_proj {
                tensors.insert(
                    format!(
                        "layers.{}.self_attn.v_proj.lora_a.weight", // Remove "_lora"
                        layer_idx
                    ),
                    v.lora_a.weight().clone(),
                );
                tensors.insert(
                    format!(
                        "layers.{}.self_attn.v_proj.lora_b.weight", // Remove "_lora"
                        layer_idx
                    ),
                    v.lora_b.weight().clone(),
                );
            }
            if let Some(ref o) = attention.o_proj {
                tensors.insert(
                    format!(
                        "layers.{}.self_attn.o_proj.lora_a.weight", // Remove "_lora"
                        layer_idx
                    ),
                    o.lora_a.weight().clone(),
                );
                tensors.insert(
                    format!(
                        "layers.{}.self_attn.o_proj.lora_b.weight", // Remove "_lora"
                        layer_idx
                    ),
                    o.lora_b.weight().clone(),
                );
            }
        }

        // Rest of the method stays the same...
        let weights_path = save_dir.join("adapter_model.safetensors");
        candle::safetensors::save(&tensors, &weights_path)?;

        let adapter_config = serde_json::json!({
            "r": self.config.rank,
            "lora_alpha": self.config.alpha,
            "lora_dropout": self.config.dropout.unwrap_or(0.0),
            "target_modules": self.config.target_modules,
            "peft_type": "LORA",
        });
        let config_path = save_dir.join("adapter_config.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&adapter_config)?)?;

        println!("Saved LoRA adapters to: {:?}", save_dir);
        println!("  - Weights: adapter_model.safetensors");
        println!("  - Config: adapter_config.json");
        println!("  - Total parameters: {}", tensors.len());
        Ok(save_dir.to_string_lossy().into_owned())
    }

    pub fn verify_merge_status(&self) {
        println!("\n=== LoRA Merge Status ===");
        for (layer_idx, attention) in self.lora_layers.iter() {
            let mut status = vec![];
            if let Some(ref q) = attention.q_proj {
                status.push(format!(
                    "q_proj: {}",
                    if q.merged { "merged" } else { "not merged" }
                ));
            }
            if let Some(ref k) = attention.k_proj {
                status.push(format!(
                    "k_proj: {}",
                    if k.merged { "merged" } else { "not merged" }
                ));
            }
            if let Some(ref v) = attention.v_proj {
                status.push(format!(
                    "v_proj: {}",
                    if v.merged { "merged" } else { "not merged" }
                ));
            }
            if let Some(ref o) = attention.o_proj {
                status.push(format!(
                    "o_proj: {}",
                    if o.merged { "merged" } else { "not merged" }
                ));
            }
            println!("Layer {}: {}", layer_idx, status.join(", "));
        }
        println!("========================\n");
    }
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

    // Load the base model
    let (mut llama, tokenizer_filename, mut cache, mut config) = {
        let api = Api::new()?;
        let model_id = args.model_id.clone().unwrap_or_else(|| match args.which {
            Which::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
            Which::SmolLM2_135M => "HuggingFaceTB/SmolLM2-135M".to_string(),
            Which::SmolLM2_135MInstruct => "HuggingFaceTB/SmolLM2-135M-Instruct".to_string(),
            Which::SmolLM2_360M => "HuggingFaceTB/SmolLM2-360M".to_string(),
            Which::SmolLM2_360MInstruct => "HuggingFaceTB/SmolLM2-360M-Instruct".to_string(),
            Which::SmolLM2_1B => "HuggingFaceTB/SmolLM2-1.7B".to_string(),
            Which::SmolLM2_1BInstruct => "HuggingFaceTB/SmolLM2-1.7B-Instruct".to_string(),
            _ => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        });
        println!("loading the model weights from {model_id}");
        let revision = args.revision.clone().unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
        let tokenizer_filename = api.get("tokenizer.json")?;
        let config_filename = api.get("config.json")?;
        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;

        // Enable LoRA in config if we're loading LoRA weights
        let mut config = config.into_config(args.use_flash_attn);
        if args.load_lora.is_some() {
            let lora_config = model::LoraConfig {
                rank: args.lora_rank,
                alpha: args.lora_alpha,
                dropout: Some(0.1),
                target_modules: args
                    .lora_target_modules
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect(),
                enabled: true,
            };
            config.lora_config = Some(lora_config);
        }

        let filenames = vec![api.get("model.safetensors")?];
        let cache = model::Cache::new(!args.no_kv_cache, dtype, &config, &device)?;
        let vb =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        (Llama::load(vb, &config)?, tokenizer_filename, cache, config)
    };

    // Load tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_filename.clone()).map_err(E::msg)?;

    // If LoRA path is provided, load and optionally merge LoRA weights
    if let Some(lora_path) = &args.load_lora {
        println!("\n=== LOADING LORA ADAPTERS ===");
        println!("Loading LoRA adapters from: {}", lora_path);

        // Load LoRA weights directly into the model
        let adapter_weights_path =
            std::path::Path::new(lora_path).join("adapter_model.safetensors");

        if !adapter_weights_path.exists() {
            return Err(anyhow::anyhow!(
                "LoRA weights file not found: {:?}",
                adapter_weights_path
            ));
        }

        llama.load_lora_weights(adapter_weights_path.to_str().unwrap())?;
        println!("LoRA weights loaded successfully!");

        // Merge weights for faster inference
        llama.merge_lora_weights()?;
        println!("LoRA weights merged into base model!");

        // Show comparison if in compare mode
        if args.compare_mode {
            println!("\n=== BASELINE vs LoRA COMPARISON ===");
            let test_prompts = vec![
                "My printer won't connect to WiFi",
                "How do I reset my router?",
                "The dishwasher is making noise",
            ];

            for prompt in &test_prompts {
                println!("\nPrompt: {}", prompt);
                let tokens = tokenizer
                    .encode(*prompt, true) // Fix: dereference prompt
                    .map_err(E::msg)?
                    .get_ids()
                    .to_vec();
                let input = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
                let logits = llama.forward(&input, 0, &mut cache.clone())?;
                let logits = logits.squeeze(0)?;
                let next_token = logits.argmax(candle::D::Minus1)?.to_scalar::<u32>()?;

                if let Some(word) = tokenizer.decode(&[next_token], false).ok() {
                    println!("LoRA Enhanced: {} [token: {}]", word, next_token);
                }
            }
        }

        println!("\n=== IMPORTANT ===");
        println!("LoRA weights have been merged into the base model weights.");
        println!("The model will now use the adapted weights for generation.");
        println!("=================\n");
    }

    // Get EOS token
    let eos_token_id = config.eos_token_id.or_else(|| {
        tokenizer
            .token_to_id(EOS_TOKEN)
            .map(model::LlamaEosToks::Single)
    });

    // Full generation with user prompt
    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = tokenizer
        .encode(prompt, true) // This is already correct - prompt is &str
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let mut tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);

    println!(
        "\nGenerating with {}LLaMA:",
        if args.load_lora.is_some() {
            "LoRA-enhanced "
        } else {
            ""
        }
    );
    println!("Prompt: {}", prompt);
    print!("{}", prompt);

    // Setup logits processor
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

    // Generation loop
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

        // Use the model normally - LoRA is now integrated
        let logits = llama.forward(&input, context_index, &mut cache)?;
        let logits = logits.squeeze(0)?;

        // Apply repeat penalty if needed
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

        // Check for EOS
        match eos_token_id {
            Some(model::LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => break,
            Some(model::LlamaEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => {
                break
            }
            _ => (),
        }

        // Print token
        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }

    // Print any remaining tokens
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }

    // Print statistics
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        (token_generated - 1) as f64 / dt.as_secs_f64()
    );

    Ok(())
}

fn run_lora_fine_tuning(args: &Args) -> anyhow::Result<()> {
    println!("=== LORA FINE-TUNING MODE ===");
    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::F16,
    };

    // Load the base model
    let (llama, tokenizer_filename, _cache, config) = {
        let api = Api::new()?;
        let model_id = args.model_id.clone().unwrap_or_else(|| match args.which {
            Which::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
            Which::SmolLM2_135M => "HuggingFaceTB/SmolLM2-135M".to_string(),
            Which::SmolLM2_135MInstruct => "HuggingFaceTB/SmolLM2-135M-Instruct".to_string(),
            Which::SmolLM2_360M => "HuggingFaceTB/SmolLM2-360M".to_string(),
            Which::SmolLM2_360MInstruct => "HuggingFaceTB/SmolLM2-360M-Instruct".to_string(),
            Which::SmolLM2_1B => "HuggingFaceTB/SmolLM2-1.7B".to_string(),
            Which::SmolLM2_1BInstruct => "HuggingFaceTB/SmolLM2-1.7B-Instruct".to_string(),
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

    println!(
        "LoRA Config: rank={}, alpha={}, targets={:?}",
        args.lora_rank, args.lora_alpha, args.lora_target_modules
    );

    let lora_config = LoraConfig::from_args(args);
    let mut lora_model = LoraLlamaModel::from_llama(llama, config, lora_config, &device)?;
    lora_model.prepare_for_training()?;

    if let Some(train_data_path) = &args.train_data {
        println!("Loading training data from: {}", train_data_path);
        // Load training data
        let training_texts = load_training_data(train_data_path)?;
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        println!(
            "Starting training simulation with {} examples",
            training_texts.len()
        );

        // Simulate training (real training would require gradient computation)
        for (idx, text) in training_texts.iter().enumerate() {
            println!("Processing example {}/{}", idx + 1, training_texts.len());
            println!("Example text: {}", &text[..text.len().min(100)]);

            let tokens = tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?
                .get_ids()
                .to_vec();

            println!("  - Tokenized into {} tokens", tokens.len());

            // In real training, we would:
            // 1. Forward pass through the model
            // 2. Compute loss
            // 3. Backward pass to compute gradients
            // 4. Update LoRA weights

            // For now, just simulate the process
            std::thread::sleep(std::time::Duration::from_millis(100));

            if idx % 2 == 0 {
                println!("  - Simulating weight update for batch");
            }
        }

        println!(
            "Training simulation completed on {} examples",
            training_texts.len()
        );
        println!("Note: This is a simulation - actual gradient computation not implemented");
    } else {
        println!("Using demo training data (no real training)");
        println!(
            "LoRA model ready for training with {} LoRA parameters",
            lora_model.lora_weights.data().lock().unwrap().len()
        );
        println!("Simulating training step...");
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }

    // Save the LoRA adapters
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

// Add the load_training_data helper function:
fn load_training_data(path: &str) -> anyhow::Result<Vec<String>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Try to parse as JSON first
    if path.ends_with(".json") {
        let content = std::fs::read_to_string(path)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;

        if let serde_json::Value::Array(arr) = data {
            let texts: Vec<String> = arr
                .into_iter()
                .filter_map(|v| {
                    if let serde_json::Value::Object(obj) = v {
                        obj.get("text")
                            .and_then(|t| t.as_str())
                            .map(|s| s.to_string())
                    } else {
                        None
                    }
                })
                .collect();
            return Ok(texts);
        }
    }

    // Otherwise, read as plain text file (one example per line)
    let texts: Vec<String> = reader
        .lines()
        .filter_map(|line| line.ok())
        .filter(|line| !line.trim().is_empty())
        .collect();

    Ok(texts)
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    fn debug_adapter_file(path: &str) -> anyhow::Result<()> {
        let tensors = candle::safetensors::load(path, &candle::Device::Cpu)?;
        println!("Tensor names in adapter file:");
        for (name, tensor) in tensors.iter() {
            println!("  {}: {:?}", name, tensor.shape());
        }
        Ok(())
    }

    let args = Args::parse();
    if args.lora_mode {
        return run_lora_fine_tuning(&args).map_err(|e| anyhow::anyhow!("{}", e));
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
            Some(model::LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => break,
            Some(model::LlamaEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => {
                break
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
        (token_generated - 1) as f64 / dt.as_secs_f64()
    );
    Ok(())
}
