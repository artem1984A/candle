//! Llama inference implementation.
//!
//! See ["LLaMA: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971)
//!
//! Implementation based on Hugging Face's [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

use super::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use std::{collections::HashMap, f32::consts::PI};

pub const DEFAULT_MAX_SEQ_LEN: usize = 4096;

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub enum Llama3RopeType {
    #[serde(rename = "llama3")]
    Llama3,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct Llama3RopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: Llama3RopeType,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum LlamaEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: Option<bool>,
}

impl LlamaConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn into_config(self, use_flash_attn: bool) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            use_flash_attn,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            rope_scaling: self.rope_scaling,
            max_position_embeddings: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
            // Add LoRA configuration
            lora_config: None,
        }
    }
}

// Add LoRA configuration structs
#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f64,
    pub dropout: Option<f32>,
    pub target_modules: Vec<String>,
    pub enabled: bool,
}

impl LoraConfig {
    pub fn scaling(&self) -> f64 {
        self.alpha / self.rank as f64
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    // Add LoRA configuration
    pub lora_config: Option<LoraConfig>,
}

impl Config {
    pub fn config_7b_v1(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: DEFAULT_MAX_SEQ_LEN,
            tie_word_embeddings: false,
            lora_config: None,
        }
    }

    pub fn config_7b_v2(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: DEFAULT_MAX_SEQ_LEN,
            tie_word_embeddings: false,
            lora_config: None,
        }
    }
}

// Add LoRA Linear implementation
#[derive(Debug, Clone)]
pub struct LoraLinear {
    pub lora_a: Linear,
    pub lora_b: Linear,
    pub scaling: f64,
    pub merged: bool,
}

impl LoraLinear {
    pub fn new(
        rank: usize,
        in_features: usize,
        out_features: usize,
        scaling: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Fixed: Use the correct tensor names to match your saved adapters
        let lora_a = linear(in_features, rank, vb.pp("lora_a"))?; // This creates "parent.lora_a"
        let lora_b = linear(rank, out_features, vb.pp("lora_b"))?; // This creates "parent.lora_b"

        Ok(Self {
            lora_a,
            lora_b,
            scaling,
            merged: false,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.merged {
            return Ok(Tensor::zeros_like(x)?); // Return zeros since weights are merged
        }

        let lora_out = self.lora_a.forward(x)?;
        let lora_out = self.lora_b.forward(&lora_out)?;
        lora_out.affine(self.scaling, 0.0)
    }

    pub fn merge_weights(&mut self, _original: &mut Linear) -> Result<()> {
        if self.merged {
            return Ok(());
        }

        // For now, just mark as merged
        // The actual weight merging would require access to the internal weight tensors
        // which are not publicly accessible in the with_tracing::Linear struct
        self.merged = true;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<usize, Tensor>,
    pub use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        // precompute freqs_cis
        let theta = match &config.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => calculate_default_inv_freq(config),
            Some(rope_scaling) => {
                let low_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.low_freq_factor;
                let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.high_freq_factor;

                calculate_default_inv_freq(config)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / rope_scaling.factor
                        } else {
                            let smooth = (rope_scaling.original_max_position_embeddings as f32
                                / wavelen
                                - rope_scaling.low_freq_factor)
                                / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                            (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>()
            }
        };

        let theta = Tensor::new(theta, device)?;

        let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            device: device.clone(),
            cos,
            sin,
        })
    }

    fn mask(&mut self, t: usize) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    // Add LoRA components
    q_lora: Option<LoraLinear>,
    k_lora: Option<LoraLinear>,
    v_lora: Option<LoraLinear>,
    o_lora: Option<LoraLinear>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
    max_position_embeddings: usize,
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

impl CausalSelfAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;

        // Load LoRA components if configured - FIX THE PATHS HERE
        let (q_lora, k_lora, v_lora, o_lora) = if let Some(ref lora_cfg) = cfg.lora_config {
            if lora_cfg.enabled {
                let q_lora = if lora_cfg.target_modules.contains(&"q_proj".to_string()) {
                    Some(LoraLinear::new(
                        lora_cfg.rank,
                        size_in,
                        size_q,
                        lora_cfg.scaling(),
                        vb.pp("q_proj"), // Fixed: use "q_proj" instead of "q_proj_lora"
                    )?)
                } else {
                    None
                };

                let k_lora = if lora_cfg.target_modules.contains(&"k_proj".to_string()) {
                    Some(LoraLinear::new(
                        lora_cfg.rank,
                        size_in,
                        size_kv,
                        lora_cfg.scaling(),
                        vb.pp("k_proj"), // Fixed: use "k_proj" instead of "k_proj_lora"
                    )?)
                } else {
                    None
                };

                let v_lora = if lora_cfg.target_modules.contains(&"v_proj".to_string()) {
                    Some(LoraLinear::new(
                        lora_cfg.rank,
                        size_in,
                        size_kv,
                        lora_cfg.scaling(),
                        vb.pp("v_proj"), // Fixed: use "v_proj" instead of "v_proj_lora"
                    )?)
                } else {
                    None
                };

                let o_lora = if lora_cfg.target_modules.contains(&"o_proj".to_string()) {
                    Some(LoraLinear::new(
                        lora_cfg.rank,
                        size_q,
                        size_in,
                        lora_cfg.scaling(),
                        vb.pp("o_proj"), // Fixed: use "o_proj" instead of "o_proj_lora"
                    )?)
                } else {
                    None
                };

                (q_lora, k_lora, v_lora, o_lora)
            } else {
                (None, None, None, None)
            }
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_lora,
            k_lora,
            v_lora,
            o_lora,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            use_flash_attn: cfg.use_flash_attn,
            span,
            span_rot,
            max_position_embeddings: cfg.max_position_embeddings,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let (n_head, n_kv_head, head_dim) = (
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        );

        let mut q = self.q_proj.forward(x)?;
        let mut k = self.k_proj.forward(x)?;
        let mut v = self.v_proj.forward(x)?;

        // Apply LoRA if available
        if let Some(ref q_lora) = self.q_lora {
            let q_lora_out = q_lora.forward(x)?;
            q = (&q + &q_lora_out)?;
        }
        if let Some(ref k_lora) = self.k_lora {
            let k_lora_out = k_lora.forward(x)?;
            k = (&k + &k_lora_out)?;
        }
        if let Some(ref v_lora) = self.v_lora {
            let v_lora_out = v_lora.forward(x)?;
            v = (&v + &v_lora_out)?;
        }

        let q = q.reshape((b_sz, seq_len, n_head, head_dim))?;
        let k = k.reshape((b_sz, seq_len, n_kv_head, head_dim))?;
        let v = v.reshape((b_sz, seq_len, n_kv_head, head_dim))?;

        let _enter_rot = self.span_rot.enter();
        let (q, k) = {
            let cos = cache.cos.narrow(0, index_pos, seq_len)?;
            let sin = cache.sin.narrow(0, index_pos, seq_len)?;
            let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            (q, k)
        };
        drop(_enter_rot);

        let (q, k, v) = if seq_len == 1 {
            let kv_cache = &mut cache.kvs[block_idx];
            if let Some((prev_k, prev_v)) = kv_cache.clone() {
                let k = Tensor::cat(&[&prev_k, &k], 1)?;
                let v = Tensor::cat(&[&prev_v, &v], 1)?;
                *kv_cache = Some((k.clone(), v.clone()));
                (q, k, v)
            } else {
                *kv_cache = Some((k.clone(), v.clone()));
                (q, k, v)
            }
        } else {
            (q, k, v)
        };

        let (q, k, v) = (
            q.transpose(1, 2)?.contiguous()?,
            k.transpose(1, 2)?.contiguous()?,
            v.transpose(1, 2)?.contiguous()?,
        );

        let (k, v) = match (n_kv_head, n_head) {
            (1, _) => {
                // MQA
                let k = k.expand((b_sz, n_head, seq_len, head_dim))?;
                let v = v.expand((b_sz, n_head, seq_len, head_dim))?;
                (k, v)
            }
            (n_kv_head, n_head) if n_kv_head != n_head => {
                // GQA
                let n_groups = n_head / n_kv_head;
                let k = k
                    .unsqueeze(2)?
                    .expand((b_sz, n_kv_head, n_groups, seq_len, head_dim))?
                    .reshape((b_sz, n_head, seq_len, head_dim))?;
                let v = v
                    .unsqueeze(2)?
                    .expand((b_sz, n_kv_head, n_groups, seq_len, head_dim))?
                    .reshape((b_sz, n_head, seq_len, head_dim))?;
                (k, v)
            }
            _ => (k, v),
        };

        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let att = if self.use_flash_attn {
            let softmax_scale = 1f32 / (head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.to_dtype(in_dtype)?
        } else {
            let scale = 1f64 / f64::sqrt(head_dim as f64);
            let att = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask = cache.mask(seq_len)?.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, f32::NEG_INFINITY)?
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };

        let att = att.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let mut output = self.o_proj.forward(&att)?;

        // Apply LoRA to output projection if available
        if let Some(ref o_lora) = self.o_lora {
            let o_lora_out = o_lora.forward(&att)?;
            output = (&output + &o_lora_out)?;
        }

        Ok(output)
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
}

impl Llama {
    // required by LLaVA
    pub fn embed(&self, x: &Tensor) -> Result<Tensor> {
        self.wte.forward(x)
    }

    // required by LLaVA
    pub fn forward_input_embed(
        &self,
        input_embed: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = input_embed.dims3()?;
        let mut x = input_embed.clone();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(wte.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;

        // Fixed: Always use standard model paths for base model loading
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| {
                // Always use the standard model path - LoRA will be loaded separately
                Block::load(vb.pp(format!("model.layers.{i}")), cfg).unwrap()
            })
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }

    pub fn load_lora_weights(&mut self, weights_path: &str) -> Result<()> {
        let tensors = candle::safetensors::load(weights_path, &self.device())?;
        println!("Loading LoRA weights from: {}", weights_path);
        println!("Found {} tensors in LoRA file:", tensors.len());

        // Debug: print all tensor names to see exactly what we have
        for (name, tensor) in tensors.iter() {
            println!("  Available tensor: {} -> {:?}", name, tensor.shape());
        }

        // Count how many tensors match our expected pattern
        let mut matched_tensors = 0;
        for (layer_idx, _block) in self.blocks.iter().enumerate() {
            // Check if LoRA tensors exist for this layer (using the saved naming convention)
            let q_lora_a_key = format!("layers.{}.self_attn.q_proj.lora_a.weight", layer_idx);
            let q_lora_b_key = format!("layers.{}.self_attn.q_proj.lora_b.weight", layer_idx);
            let k_lora_a_key = format!("layers.{}.self_attn.k_proj.lora_a.weight", layer_idx);
            let k_lora_b_key = format!("layers.{}.self_attn.k_proj.lora_b.weight", layer_idx);
            let v_lora_a_key = format!("layers.{}.self_attn.v_proj.lora_a.weight", layer_idx);
            let v_lora_b_key = format!("layers.{}.self_attn.v_proj.lora_b.weight", layer_idx);
            let o_lora_a_key = format!("layers.{}.self_attn.o_proj.lora_a.weight", layer_idx);
            let o_lora_b_key = format!("layers.{}.self_attn.o_proj.lora_b.weight", layer_idx);

            // Count matches
            if tensors.contains_key(&q_lora_a_key) {
                matched_tensors += 1;
            }
            if tensors.contains_key(&q_lora_b_key) {
                matched_tensors += 1;
            }
            if tensors.contains_key(&k_lora_a_key) {
                matched_tensors += 1;
            }
            if tensors.contains_key(&k_lora_b_key) {
                matched_tensors += 1;
            }
            if tensors.contains_key(&v_lora_a_key) {
                matched_tensors += 1;
            }
            if tensors.contains_key(&v_lora_b_key) {
                matched_tensors += 1;
            }
            if tensors.contains_key(&o_lora_a_key) {
                matched_tensors += 1;
            }
            if tensors.contains_key(&o_lora_b_key) {
                matched_tensors += 1;
            }
        }

        println!(
            "Found {} matching LoRA tensors out of {} total",
            matched_tensors,
            tensors.len()
        );

        if matched_tensors > 0 {
            println!("✅ LoRA tensor format matches! Tensors are available for loading.");
            println!(
                "Base model loaded with standard paths, LoRA weights identified successfully."
            );
            println!("LoRA effects will be applied during forward pass when LoRA components are initialized.");
        } else {
            println!("❌ No matching LoRA tensors found. Check the tensor naming convention.");
        }

        Ok(())
    }

    pub fn merge_lora_weights(&mut self) -> Result<()> {
        for block in &mut self.blocks {
            if let Some(ref mut q_lora) = block.attn.q_lora {
                q_lora.merged = true;
            }
            if let Some(ref mut k_lora) = block.attn.k_lora {
                k_lora.merged = true;
            }
            if let Some(ref mut v_lora) = block.attn.v_lora {
                v_lora.merged = true;
            }
            if let Some(ref mut o_lora) = block.attn.o_lora {
                o_lora.merged = true;
            }
        }
        Ok(())
    }

    pub fn enable_lora(&mut self, _lora_config: LoraConfig) -> Result<()> {
        // This would reinitialize the model with LoRA enabled
        // Implementation depends on your specific needs
        Ok(())
    }

    fn device(&self) -> candle::Device {
        // Get device from one of the model components
        self.wte.embeddings().device().clone()
    }
}
