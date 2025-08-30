//! Standalone Q8_K (BlockQ8K) quantizer for a single `model.safetensors` shard.
//!
//! Usage:
//!   cargo run --release -p tensor-tools --bin quantize_q8k \
//!     /home/artem/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/model.safetensors  /home/artem/candle/tinyllama-q8k
//!
//! Produces one .q8k file per selected 2D linear weight:
//!   out-dir/<tensor-name>.q8k
//!
//! Skips embeddings (`embed_tokens`) and norm parameters (`norm`).

use anyhow::{bail, Context, Result};
use candle::quantized::k_quants::{matmul, BlockQ8K, GgmlType, QK_K};
use candle::Device;
use half::{bf16, f16};
use once_cell::sync::Lazy;
use regex::Regex;
use safetensors::tensor::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::{
    fs,
    io::{BufWriter, Write},
    mem,
    path::{Path, PathBuf},
    time::Instant,
};
static LAYER_PERM_CACHE: OnceLock<Mutex<LayerPermCache>> = OnceLock::new();
static ATTN_PROJ_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight$").unwrap());

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

const MAGIC_Q8K: u32 = 0x4B51_3838; // "KQ88" little-endian
const VERSION: u32 = 1;
const DTYPE_Q8K: u32 = 0x18; // Arbitrary tag documenting Q8K

fn is_target_weight(name: &str) -> bool {
    if !name.ends_with(".weight") {
        return false;
    }
    if name.contains("embed_tokens") {
        return false;
    }
    if name.contains("norm") {
        return false;
    }
    true
}

fn validate_quantization(original: &[f32], blocks: &[BlockQ8K], k: usize) -> Result<f32> {
    let rows = original.len() / k;

    // Create test vectors to validate via matrix multiplication
    // This follows the pattern from candle-core/tests/quantized_tests.rs
    let _device = Device::Cpu;
    let test_input = vec![1.0f32; k]; // Simple test vector

    // Expected output: multiply original weights by test vector
    let mut expected_output = vec![0f32; rows];
    for row in 0..rows {
        let row_data = &original[row * k..(row + 1) * k];
        expected_output[row] = row_data.iter().sum(); // Sum since test_input is all 1s
    }

    // Actual output: multiply quantized weights by test vector
    let mut actual_output = vec![0f32; rows];
    matmul::<BlockQ8K>((1, k, rows), &test_input, blocks, &mut actual_output)
        .map_err(|e| anyhow::anyhow!("matmul failed: {}", e))?;

    // Calculate MSE between expected and actual outputs
    let mut mse = 0f32;
    for (expected, actual) in expected_output.iter().zip(actual_output.iter()) {
        let diff = expected - actual;
        mse += diff * diff;
    }
    mse /= rows as f32;

    Ok(mse)
}

fn validate_quantization_direct(original: &[f32], blocks: &[BlockQ8K], k: usize) -> Result<f32> {
    // For now, use the same approach as matmul validation but with a different test vector
    // This ensures we get meaningful results while keeping dual validation
    let rows = original.len() / k;

    // Use a different test pattern to validate reconstruction quality
    let mut test_input = vec![0f32; k];
    for i in 0..k {
        test_input[i] = (i as f32 + 1.0) / k as f32; // Gradient from 0 to 1
    }

    // Expected output: multiply original weights by test vector
    let mut expected_output = vec![0f32; rows];
    for row in 0..rows {
        let row_data = &original[row * k..(row + 1) * k];
        expected_output[row] = row_data
            .iter()
            .zip(test_input.iter())
            .map(|(a, b)| a * b)
            .sum();
    }

    // Actual output: multiply quantized weights by test vector
    let mut actual_output = vec![0f32; rows];
    matmul::<BlockQ8K>((1, k, rows), &test_input, blocks, &mut actual_output)
        .map_err(|e| anyhow::anyhow!("direct validation matmul failed: {}", e))?;

    // Calculate MSE between expected and actual outputs
    let mut mse = 0f32;
    for (expected, actual) in expected_output.iter().zip(actual_output.iter()) {
        let diff = expected - actual;
        mse += diff * diff;
    }
    mse /= rows as f32;

    Ok(mse)
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
        other => bail!("unsupported dtype {other:?}"),
    })
}

fn quantize_rows_q8k(rows: usize, k: usize, data: &[f32]) -> Result<Vec<BlockQ8K>> {
    if k % QK_K != 0 {
        bail!("inner dim {k} not multiple of {QK_K}");
    }
    let blocks_per_row = k / QK_K;
    let mut blocks = vec![BlockQ8K::zeros(); rows * blocks_per_row];
    for r in 0..rows {
        let row = &data[r * k..(r + 1) * k];
        let dst = &mut blocks[r * blocks_per_row..(r + 1) * blocks_per_row];
        BlockQ8K::from_float(row, dst)?;
    }
    Ok(blocks)
}

fn write_q8k(path: &Path, rows: usize, k: usize, blocks: &[BlockQ8K]) -> Result<()> {
    let header = Q8KHeader {
        magic: MAGIC_Q8K,
        version: VERSION,
        out: rows as u32,
        k: k as u32,
        blocks_per_row: (k / QK_K) as u32,
        dtype: DTYPE_Q8K,
    };
    let mut w = BufWriter::new(fs::File::create(path)?);
    w.write_all(bytemuck::bytes_of(&header))?;
    let raw = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            blocks.len() * mem::size_of::<BlockQ8K>(),
        )
    };
    w.write_all(raw)?;
    w.flush()?;
    Ok(())
}

fn column_l2_norms(rows: usize, k: usize, data: &[f32]) -> Vec<f32> {
    let mut sums: Vec<f64> = vec![0.0; k];
    for r in 0..rows {
        let row = &data[r * k..(r + 1) * k];
        for (j, &v) in row.iter().enumerate() {
            let fv = v as f64;
            sums[j] += fv * fv;
        }
    }
    sums.into_iter().map(|s| (s.sqrt()) as f32).collect()
}

fn build_column_permutation(norms: &[f32]) -> Vec<usize> {
    // Sort columns by descending norm to cluster similar magnitudes in contiguous blocks
    let mut idx: Vec<usize> = (0..norms.len()).collect();
    idx.sort_by(|&a, &b| {
        norms[b]
            .partial_cmp(&norms[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

fn apply_column_permutation(rows: usize, k: usize, data: &[f32], perm: &[usize]) -> Vec<f32> {
    let mut out = vec![0f32; rows * k];
    for r in 0..rows {
        let src = &data[r * k..(r + 1) * k];
        let dst = &mut out[r * k..(r + 1) * k];
        for j in 0..k {
            dst[j] = src[perm[j]];
        }
    }
    out
}

fn write_perm(path_q8k: &Path, perm: &[usize]) -> Result<()> {
    // Write a sidecar file "<name>.perm" with u32 indices
    let mut p = path_q8k.to_path_buf();
    p.set_extension("perm");
    let mut w = BufWriter::new(fs::File::create(&p)?);
    // Simple binary: magic + k + u32[k]
    const MAGIC_PERM: u32 = 0x4D52_4550; // "PERM"
    w.write_all(&MAGIC_PERM.to_le_bytes())?;
    w.write_all(&(perm.len() as u32).to_le_bytes())?;
    for &u in perm {
        w.write_all(&(u as u32).to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

fn load_perm(path_q8k: &Path) -> Result<Option<Vec<usize>>> {
    // Looks for <same-name>.perm
    let mut p = path_q8k.to_path_buf();
    p.set_extension("perm");
    if !p.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&p)?;
    if bytes.len() < 8 {
        bail!("perm file too small: {}", p.display());
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    const MAGIC_PERM: u32 = 0x4D52_4550;
    if magic != MAGIC_PERM {
        bail!("bad perm magic in {}", p.display());
    }
    let k = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let expect = 8 + 4 * k;
    if bytes.len() != expect {
        bail!(
            "perm size mismatch {} (got {}, expect {})",
            p.display(),
            bytes.len(),
            expect
        );
    }
    let mut perm = Vec::with_capacity(k);
    for i in 0..k {
        let off = 8 + 4 * i;
        let idx = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        perm.push(idx);
    }
    Ok(Some(perm))
}

// Produce permuted copy of activation x (len = k)
fn permute_activation(x: &[f32], perm: &[usize]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (j, &src) in perm.iter().enumerate() {
        out[j] = x[src];
    }
    out
}

#[derive(Debug, Clone)]
struct LayerPermCache {
    // layer_id -> permutation vector
    map: HashMap<u32, Vec<usize>>,
}

impl LayerPermCache {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn get_or_compute(&mut self, layer_id: u32, rows: usize, k: usize, data: &[f32]) -> Vec<usize> {
        if let Some(p) = self.map.get(&layer_id) {
            return p.clone();
        }
        // Compute permutation from this tensor alone (simplest).
        // (Future: accumulate q+k+v norms before deciding.)
        let norms = column_l2_norms(rows, k, data);
        let perm = build_column_permutation(&norms);
        self.map.insert(layer_id, perm.clone());
        perm
    }
}

fn parse_attention_proj(name: &str) -> Option<(u32, &'static str)> {
    if let Some(caps) = ATTN_PROJ_RE.captures(name) {
        let layer_id: u32 = caps[1].parse().ok()?;
        let kind = match &caps[2] {
            "q" => "q",
            "k" => "k",
            "v" => "v",
            "o" => "o",
            _ => return None,
        };
        Some((layer_id, kind))
    } else {
        None
    }
}

// ...existing code...
fn main() -> Result<()> {
    // Parse args
    let mut args = std::env::args().skip(1);
    let in_file: PathBuf = args
        .next()
        .context("Usage: quantize_q8k <input.safetensors> <output_dir>\n\nExample:\n  cargo run --release -p tensor-tools --bin quantize_q8k model.safetensors ./quantized")?
        .into();
    let out_dir: PathBuf = args
        .next()
        .context("Usage: quantize_q8k <input.safetensors> <output_dir>\n\nExample:\n  cargo run --release -p tensor-tools --bin quantize_q8k model.safetensors ./quantized")?
        .into();

    if !in_file.exists() {
        bail!("Input file does not exist: {}", in_file.display());
    }
    fs::create_dir_all(&out_dir)?;

    // Toggle permutation
    let use_permute = std::env::var("CANDLE_Q8K_PERMUTE")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    println!("Input  : {}", in_file.display());
    println!("Output : {}", out_dir.display());
    println!("Permute: {}", if use_permute { "on" } else { "off" });

    let t0 = Instant::now();
    let bytes = fs::read(&in_file)?;
    let st = SafeTensors::deserialize(&bytes)?;
    println!("Tensors: {}", st.len());

    let mut quantized_count = 0usize;
    let mut skipped_count = 0usize;

    for name in st.names() {
        let t = st.tensor(name)?;
        let shape = t.shape();
        if shape.len() != 2 || !is_target_weight(name) {
            skipped_count += 1;
            continue;
        }
        let (rows, k) = (shape[0], shape[1]);
        if k % QK_K != 0 {
            println!("skip (k % {QK_K} != 0): {name} [{rows} x {k}]");
            skipped_count += 1;
            continue;
        }

        println!("quantizing {name} ({rows} x {k})");

        // Load weights to f32
        let data_f32 = tensor_to_f32(t.data(), t.dtype())?;

        // Optional permutation
        let (data_for_quant, maybe_perm): (Vec<f32>, Option<Vec<usize>>) = if use_permute {
            if let Some((layer_id, kind)) = parse_attention_proj(name) {
                if kind == "o" {
                    (data_f32, None)
                } else {
                    // Lock cache, get/compute perm for this attention layer, then drop lock
                    let perm = {
                        let mut cache = LAYER_PERM_CACHE
                            .get_or_init(|| Mutex::new(LayerPermCache::new()))
                            .lock()
                            .unwrap();
                        cache.get_or_compute(layer_id, rows, k, &data_f32)
                    }; // lock dropped here
                    let permuted = apply_column_permutation(rows, k, &data_f32, &perm);
                    (permuted, Some(perm))
                }
            } else {
                // Non-attention linear: independent permutation
                let norms = column_l2_norms(rows, k, &data_f32);
                let perm = build_column_permutation(&norms);
                let permuted = apply_column_permutation(rows, k, &data_f32, &perm);
                (permuted, Some(perm))
            }
        } else {
            (data_f32, None)
        };

        // Quantize
        let blocks = quantize_rows_q8k(rows, k, &data_for_quant)?;

        // Validate using the same matrix we quantized
        let mse_matmul = validate_quantization(&data_for_quant, &blocks, k)?;
        let mse_direct = validate_quantization_direct(&data_for_quant, &blocks, k)?;
        println!("  MSE (matmul): {:.6e}", mse_matmul);
        println!("  MSE (direct): {:.6e}", mse_direct);
        let diff = (mse_matmul - mse_direct).abs();
        if diff > 1e-6 {
            println!("    [INFO] Validation methods differ by {:.8e}", diff);
        }
        if mse_matmul > 1e-2 || mse_direct > 1e-2 {
            println!("    [WARN] High MSE detected - quantization may be lossy");
        }

        // Write artifacts
        let out_path = out_dir.join(format!("{name}.q8k"));
        write_q8k(&out_path, rows, k, &blocks)?;
        if let Some(perm) = &maybe_perm {
            write_perm(&out_path, perm)?;
        }

        quantized_count += 1;
    }

    println!(
        "Done in {:.2}s. Quantized: {quantized_count}, skipped: {skipped_count}",
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
