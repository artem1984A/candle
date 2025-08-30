//! Minimal TinyLlama inference skeleton using external .q8k (BlockQ8K) weight files
//! produced by tensor-tools/bin/quantize_q8k.rs.
//!
//! Run (example):
//    cargo run --release -p candle-examples --example llama-q8k \
//      --model-safetensors ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/<hash>/model.safetensors \
//      --qdir ./tinyllama-q8k --prompt "Hello"
//!
//! This focuses on integrating the quantized linear layers (q_proj,k_proj,v_proj,o_proj,
//! gate_proj, up_proj, down_proj, lm_head). Embeddings + rms norms are kept in f32.
//!
//! NOTE: This is a pedagogical scaffold – it omits full attention cache handling,
//! rotary embeddings & masking details present in the full model
//! ([candle-transformers/src/models/quantized_llama.rs]).

use anyhow::{bail, Context, Result};
use candle::quantized::k_quants::{matmul, BlockQ8K, GgmlType, QK_K};
use candle::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::ops::silu;
use half::{bf16, f16};
use safetensors::tensor::{Dtype, SafeTensors};

pub mod llama_q8k;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Q8KHeader {
    magic: u32,
    version: u32,
    out: u32,
    k: u32,
    blocks_per_row: u32,
    dtype: u32,
}
const MAGIC_Q8K: u32 = 0x4B51_3838;
const DTYPE_Q8K_TAG: u32 = 0x18;

fn read_q8k(path: &Path) -> Result<(Vec<BlockQ8K>, usize, usize)> {
    let data = fs::read(path)?;
    if data.len() < std::mem::size_of::<Q8KHeader>() {
        bail!("file too small: {}", path.display());
    }
    let hdr = *bytemuck::from_bytes::<Q8KHeader>(&data[..std::mem::size_of::<Q8KHeader>()]);
    if hdr.magic != MAGIC_Q8K {
        bail!("bad magic in {}", path.display());
    }
    let total_blocks = (hdr.out as usize) * (hdr.blocks_per_row as usize);
    let expected =
        std::mem::size_of::<Q8KHeader>() + total_blocks * std::mem::size_of::<BlockQ8K>();
    if data.len() != expected {
        bail!(
            "size mismatch {} (have {}, expect {})",
            path.display(),
            data.len(),
            expected
        );
    }
    let mut blocks = vec![BlockQ8K::zeros(); total_blocks];
    let raw = &data[std::mem::size_of::<Q8KHeader>()..];
    unsafe {
        std::ptr::copy_nonoverlapping(raw.as_ptr(), blocks.as_mut_ptr() as *mut u8, raw.len());
    }
    Ok((blocks, hdr.out as usize, hdr.k as usize))
}

struct QuantLinearQ8K {
    blocks: Vec<BlockQ8K>,
    out: usize,
    k: usize,
}

impl QuantLinearQ8K {
    fn load(qdir: &Path, name: &str) -> Result<Self> {
        let file = qdir.join(format!("{name}.q8k"));
        let (blocks, out, k) = read_q8k(&file)?;
        if k % QK_K != 0 {
            bail!("k % {QK_K} != 0 for {name}");
        }
        Ok(Self { blocks, out, k })
    }

    fn forward_2d(&self, x2d: &Tensor) -> Result<Tensor> {
        let (b, k_in) = x2d.dims2()?;
        if k_in != self.k {
            bail!("k mismatch: got {k_in}, expected {}", self.k);
        }
        let x_f32 = x2d.to_dtype(DType::F32)?.contiguous()?;
        let lhs: Vec<f32> = x_f32.to_vec1()?;
        let mut out_buf = vec![0f32; b * self.out];
        matmul::<BlockQ8K>((b, self.k, self.out), &lhs, &self.blocks, &mut out_buf)?;
        Tensor::from_vec(out_buf, (b, self.out), x2d.device())
    }
}

impl Module for QuantLinearQ8K {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match x.dims().len() {
            2 => self.forward_2d(x),
            3 => {
                // (batch, seq, k) -> flatten, matmul, reshape
                let (b, s, k) = x.dims3()?;
                let flat = x.reshape((b * s, k))?;
                let y = self.forward_2d(&flat)?;
                y.reshape((b, s, self.out))
            }
            _ => bail!("unsupported input rank"),
        }
    }
}

struct Mlp {
    gate: QuantLinearQ8K,
    up: QuantLinearQ8K,
    down: QuantLinearQ8K,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(x)?;
        let u = self.up.forward(x)?;
        let act = silu(&g)?;
        let h = (act * u)?;
        self.down.forward(&h)
    }
}

// Very simplified layer: (No rotary, no kv cache, no attention mask for demo)
struct Layer {
    q_proj: QuantLinearQ8K,
    k_proj: QuantLinearQ8K,
    v_proj: QuantLinearQ8K,
    o_proj: QuantLinearQ8K,
    mlp: Mlp,
    attn_norm_w: Tensor,
    ffn_norm_w: Tensor,
    n_head: usize,
    head_dim: usize,
}

fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let (_b, _s, d) = {
        let dims = x.dims();
        (
            dims[dims.len() - 3],
            dims[dims.len() - 2],
            dims[dims.len() - 1],
        )
    };
    let x2 = (x * x)?;
    let mean = x2.mean_keepdim(&[-1])?;
    let inv = (mean + eps)?.rsqrt()?;
    ((x * inv)? * w)
}

impl Layer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_norm = rms_norm(x, &self.attn_norm_w, 1e-5)?;
        let q = self.q_proj.forward(&x_norm)?;
        let k = self.k_proj.forward(&x_norm)?;
        let v = self.v_proj.forward(&x_norm)?;
        // Naive attention (batch, seq, hidden) shapes:
        let (b, s, _) = q.dims3()?;
        let hd = self.head_dim;
        let nh = self.n_head;
        let q = q.reshape((b, s, nh, hd))?.transpose(1, 2)?; // (b, nh, s, hd)
        let k = k.reshape((b, s, nh, hd))?.transpose(1, 2)?;
        let v = v.reshape((b, s, nh, hd))?.transpose(1, 2)?;
        let kt = k.transpose(2, 3)?; // (b, nh, hd, s)
        let att = (q.matmul(&kt)? / (hd as f64).sqrt())?;
        let att = candle_nn::ops::softmax(att, candle::D::Minus1)?;
        let y = att
            .matmul(&v)? // (b, nh, s, hd)
            .transpose(1, 2)?
            .reshape((b, s, nh * hd))?;
        let y = self.o_proj.forward(&y)?;
        let x = (y + x)?;
        // MLP
        let x_norm2 = rms_norm(&x, &self.ffn_norm_w, 1e-5)?;
        let x_mlp = self.mlp.forward(&x_norm2)?;
        (x + x_mlp)
    }
}

struct TinyLlamaQ8K {
    embed: Tensor,
    embed_weight: Tensor,
    layers: Vec<Layer>,
    norm_w: Tensor,
    lm_head: QuantLinearQ8K,
    n_head: usize,
    head_dim: usize,
}

impl TinyLlamaQ8K {
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        // tokens: (batch, seq)
        let x = self.embed.index_select(&self.embed_weight, 0, tokens)?;
        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        let h = rms_norm(&h, &self.norm_w, 1e-5)?;
        // Take last token for logits (causal)
        let last = h.i((.., -1, ..))?.reshape((h.dims()[0], 1, h.dims()[2]))?;
        self.lm_head.forward(&last)?.squeeze(1)
    }
}

// --- Loading utilities ---

fn load_safetensors(path: &Path) -> Result<SafeTensors> {
    Ok(SafeTensors::deserialize(&fs::read(path)?)?)
}

fn tensor_to_f32(bytes: &[u8], dtype: Dtype) -> Result<Vec<f32>> {
    Ok(match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32())
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32())
            .collect(),
        _ => bail!("unsupported dtype"),
    })
}

// Build the model using:
// - .q8k files for linear layers
// - original safetensors for embedding + norm weights
fn build_model(
    safetensors_path: &Path,
    qdir: &Path,
    device: &Device,
    n_layers: usize,
    n_head: usize,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
) -> Result<TinyLlamaQ8K> {
    let st = load_safetensors(safetensors_path)?;
    // Embedding + final norm
    let embed_view = st
        .tensor("model.embed_tokens.weight")
        .context("missing embedding")?;
    let emb_data = tensor_to_f32(embed_view.data(), embed_view.dtype())?;
    let embed_weight = Tensor::from_vec(emb_data, (vocab_size, hidden_size), device)?;
    let norm_w_view = st.tensor("model.norm.weight")?;
    let norm_w = Tensor::from_vec(
        tensor_to_f32(norm_w_view.data(), norm_w_view.dtype())?,
        norm_w_view.shape(),
        device,
    )?;

    // Layer rms norm weights
    let mut layers = Vec::with_capacity(n_layers);
    let head_dim = hidden_size / n_head;
    for i in 0..n_layers {
        let attn_norm_v = st.tensor(&format!("model.layers.{i}.attention_norm.weight"))?;
        let ffn_norm_v = st.tensor(&format!("model.layers.{i}.ffn_norm.weight"))?;
        let attn_norm_w = Tensor::from_vec(
            tensor_to_f32(attn_norm_v.data(), attn_norm_v.dtype())?,
            attn_norm_v.shape(),
            device,
        )?;
        let ffn_norm_w = Tensor::from_vec(
            tensor_to_f32(ffn_norm_v.data(), ffn_norm_v.dtype())?,
            ffn_norm_v.shape(),
            device,
        )?;
        let layer = Layer {
            q_proj: QuantLinearQ8K::load(
                qdir,
                &format!("model.layers.{i}.self_attn.q_proj.weight"),
            )?,
            k_proj: QuantLinearQ8K::load(
                qdir,
                &format!("model.layers.{i}.self_attn.k_proj.weight"),
            )?,
            v_proj: QuantLinearQ8K::load(
                qdir,
                &format!("model.layers.{i}.self_attn.v_proj.weight"),
            )?,
            o_proj: QuantLinearQ8K::load(
                qdir,
                &format!("model.layers.{i}.self_attn.o_proj.weight"),
            )?,
            mlp: Mlp {
                gate: QuantLinearQ8K::load(
                    qdir,
                    &format!("model.layers.{i}.mlp.gate_proj.weight"),
                )?,
                up: QuantLinearQ8K::load(qdir, &format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down: QuantLinearQ8K::load(
                    qdir,
                    &format!("model.layers.{i}.mlp.down_proj.weight"),
                )?,
            },
            attn_norm_w,
            ffn_norm_w,
            n_head,
            head_dim,
        };
        layers.push(layer);
    }
    let lm_head = QuantLinearQ8K::load(qdir, "lm_head.weight")?;
    Ok(TinyLlamaQ8K {
        embed: Tensor::zeros((1,), DType::F32, device)?, // dummy just to reuse index_select
        embed_weight,
        layers,
        norm_w,
        lm_head,
        n_head,
        head_dim,
    })
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path: PathBuf = args
        .next()
        .context("--model-safetensors <path> missing")?
        .into();
    let qdir: PathBuf = args
        .next()
        .context("--qdir <quantized_dir> missing (pass second arg)")?
        .into();
    let prompt: String = args.next().unwrap_or_else(|| "Hello".to_string());

    let device = Device::Cpu;
    // TinyLlama-1.1B Chat config (adjust if different)
    let hidden_size = 2048;
    let n_head = 32;
    let n_layers = 22;
    let intermediate_size = 5632;
    let vocab_size = 32000;

    println!("Loading model...");
    let model = build_model(
        &model_path,
        &qdir,
        &device,
        n_layers,
        n_head,
        hidden_size,
        intermediate_size,
        vocab_size,
    )?;
    println!("Model built. (Demo forward over token ids length=1)");

    // Minimal tokenization stub (replace with real tokenizer).
    let dummy_token_id = 1u32; // pretend BOS + single token
    let tokens = Tensor::new(&[dummy_token_id], &device)?.reshape((1, 1))?;
    let logits = model.forward(&tokens)?;
    println!("Logits slice: {:?}", logits.i((0, 0..10))?);
    println!("Prompt (not actually processed): {prompt}");
    Ok(())
}
