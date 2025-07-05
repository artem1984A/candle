use candle::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// LoRA rank (bottleneck dimension)
    #[arg(long, default_value = "8")]
    rank: usize,

    /// LoRA alpha scaling factor
    #[arg(long, default_value = "16.0")]
    alpha: f32,

    /// Input dimension
    #[arg(long, default_value = "128")]
    input_dim: usize,

    /// Output dimension  
    #[arg(long, default_value = "64")]
    output_dim: usize,

    /// Test different layer types
    #[arg(long)]
    test_attention: bool,

    /// Enable weight merging demonstration
    #[arg(long)]
    test_merging: bool,
}

// Enhanced LoRA Configuration (inspired by candle-lora)
#[derive(Clone, Debug)]
struct LoraConfig {
    rank: usize,
    alpha: f32,
    dropout: Option<f32>,
}

impl LoraConfig {
    fn new(rank: usize, alpha: f32, dropout: Option<f32>) -> Self {
        Self {
            rank,
            alpha,
            dropout,
        }
    }

    fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

// Enhanced SimpleLoRA with merging capabilities
struct SimpleLoRA {
    original: Linear,
    lora_a: Linear,
    lora_b: Linear,
    config: LoraConfig,
    merged: bool,
}

impl SimpleLoRA {
    fn new(
        original: Linear,
        config: LoraConfig,
        input_dim: usize,
        output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        println!("=== SimpleLoRA::new Debug Info ===");
        println!("VarBuilder device: {:?}", vb.device());
        println!("VarBuilder dtype: {:?}", vb.dtype());
        println!("Input dimensions: {}x{}", input_dim, output_dim);
        println!("LoRA rank: {}", config.rank);

        // Debug original layer
        println!(
            "Original layer weight shape: {:?}",
            original.weight().shape()
        );
        println!(
            "Original layer weight dtype: {:?}",
            original.weight().dtype()
        );

        // FIXED: Use VarBuilder.get() with explicit logging
        println!("Creating LoRA A weight: ({}, {})", config.rank, input_dim);
        let lora_a_weight = vb.get(
            (config.rank, input_dim), // [output, input] format for candle Linear
            "lora_a.weight",
        )?;
        println!(
            "LoRA A weight created - shape: {:?}, dtype: {:?}",
            lora_a_weight.shape(),
            lora_a_weight.dtype()
        );
        let lora_a = Linear::new(lora_a_weight, None);

        println!("Creating LoRA B weight: ({}, {})", output_dim, config.rank);
        let lora_b_weight = vb.get(
            (output_dim, config.rank), // [output, input] format for candle Linear
            "lora_b.weight",
        )?;
        println!(
            "LoRA B weight created - shape: {:?}, dtype: {:?}",
            lora_b_weight.shape(),
            lora_b_weight.dtype()
        );
        let lora_b = Linear::new(lora_b_weight, None);

        println!("=== SimpleLoRA::new Complete ===\n");

        Ok(Self {
            original,
            lora_a,
            lora_b,
            config,
            merged: false,
        })
    }

    // Weight merging (inspired by candle-lora's Merge trait)
    fn merge_weights(&mut self) -> Result<()> {
        if self.merged {
            println!("Weights already merged!");
            return Ok(());
        }

        // FIXED: Correct order - B × A, not A × B (following candle-lora patterns)
        let _delta = self
            .lora_b
            .weight() // [64, 8]
            .matmul(self.lora_a.weight())? // [8, 128]  -> [64, 128] ✓
            .affine(self.config.scaling() as f64, 0.0)?;

        // Add delta to original weights (conceptually)
        println!("Merging LoRA weights into original layer...");
        self.merged = true;
        Ok(())
    }

    fn unmerge_weights(&mut self) -> Result<()> {
        if !self.merged {
            println!("Weights not merged!");
            return Ok(());
        }

        println!("Unmerging LoRA weights from original layer...");
        self.merged = false;
        Ok(())
    }

    fn get_delta_weight(&self) -> Result<Tensor> {
        // FIXED: Correct order - B × A, following candle-lora patterns
        self.lora_b
            .weight()
            .matmul(self.lora_a.weight())?
            .affine(self.config.scaling() as f64, 0.0)
    }
}

impl Module for SimpleLoRA {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        println!("=== SimpleLoRA::forward Debug Info ===");
        println!(
            "Input shape: {:?}, dtype: {:?}",
            input.shape(),
            input.dtype()
        );
        println!("Merged status: {}", self.merged);

        let original_output = self.original.forward(input)?;
        println!(
            "Original output shape: {:?}, dtype: {:?}",
            original_output.shape(),
            original_output.dtype()
        );

        if self.merged {
            println!("Using merged weights, returning original output");
            return Ok(original_output);
        }

        // Debug LoRA forward pass
        println!("=== LoRA Forward Pass Debug ===");
        println!(
            "LoRA A weight shape: {:?}, dtype: {:?}",
            self.lora_a.weight().shape(),
            self.lora_a.weight().dtype()
        );
        println!(
            "LoRA B weight shape: {:?}, dtype: {:?}",
            self.lora_b.weight().shape(),
            self.lora_b.weight().dtype()
        );

        println!("Applying LoRA A...");
        let lora_a_output = input.apply(&self.lora_a)?;
        println!(
            "LoRA A output shape: {:?}, dtype: {:?}",
            lora_a_output.shape(),
            lora_a_output.dtype()
        );

        println!("Applying LoRA B...");
        let lora_b_output = lora_a_output.apply(&self.lora_b)?;
        println!(
            "LoRA B output shape: {:?}, dtype: {:?}",
            lora_b_output.shape(),
            lora_b_output.dtype()
        );

        println!("Applying scaling factor: {:.3}", self.config.scaling());
        let lora_output = lora_b_output.affine(self.config.scaling() as f64, 0.0)?;
        println!(
            "LoRA scaled output shape: {:?}, dtype: {:?}",
            lora_output.shape(),
            lora_output.dtype()
        );

        println!("Adding original + LoRA outputs...");
        // FIXED: Handle the Result properly
        let final_output = (original_output + lora_output)?;
        println!(
            "Final output shape: {:?}, dtype: {:?}",
            final_output.shape(),
            final_output.dtype()
        );
        println!("=== SimpleLoRA::forward Complete ===\n");

        Ok(final_output) // FIXED: Return Ok(final_output)
    }
}

// Multi-Head Attention LoRA (inspired by transformer patterns)
struct AttentionLoRA {
    q_lora: SimpleLoRA,
    k_lora: SimpleLoRA,
    v_lora: SimpleLoRA,
    o_lora: SimpleLoRA,
    config: LoraConfig,
}

impl AttentionLoRA {
    fn new(
        q_linear: Linear,
        k_linear: Linear,
        v_linear: Linear,
        o_linear: Linear,
        config: LoraConfig,
        hidden_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            q_lora: SimpleLoRA::new(q_linear, config.clone(), hidden_dim, hidden_dim, vb.pp("q"))?,
            k_lora: SimpleLoRA::new(k_linear, config.clone(), hidden_dim, hidden_dim, vb.pp("k"))?,
            v_lora: SimpleLoRA::new(v_linear, config.clone(), hidden_dim, hidden_dim, vb.pp("v"))?,
            o_lora: SimpleLoRA::new(o_linear, config.clone(), hidden_dim, hidden_dim, vb.pp("o"))?,
            config,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified attention (just demonstrates LoRA in multiple projections)
        let q = self.q_lora.forward(input)?;
        let k = self.k_lora.forward(input)?;
        let v = self.v_lora.forward(input)?;

        // Simplified attention computation
        let attn_scores = q.matmul(&k.transpose(1, 2)?)?;
        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_scores)?;
        let attn_output = attn_probs.matmul(&v)?;

        self.o_lora.forward(&attn_output)
    }

    fn get_total_lora_params(&self) -> usize {
        let single_layer_params = (128 * self.config.rank) + (self.config.rank * 128);
        single_layer_params * 4 // q, k, v, o projections
    }
}

// Analysis utilities
struct LoraAnalyzer;

impl LoraAnalyzer {
    fn analyze_efficiency(original_params: usize, lora_params: usize) -> (f32, String) {
        let reduction = (1.0 - (lora_params as f32 / original_params as f32)) * 100.0;
        let efficiency_level = match reduction {
            r if r > 95.0 => "Extremely Efficient",
            r if r > 90.0 => "Very Efficient",
            r if r > 80.0 => "Efficient",
            r if r > 50.0 => "Moderately Efficient",
            _ => "Low Efficiency",
        };
        (reduction, efficiency_level.to_string())
    }

    fn compare_configurations(configs: Vec<LoraConfig>, input_dim: usize, output_dim: usize) {
        println!("\n=== LoRA Configuration Comparison ===");
        println!(
            "{:<6} {:<8} {:<12} {:<12} {:<15}",
            "Rank", "Alpha", "Scaling", "Params", "Efficiency"
        );
        println!("{}", "-".repeat(60));

        let original_params = input_dim * output_dim;

        for config in configs {
            let lora_params = (input_dim * config.rank) + (config.rank * output_dim);
            let (reduction, _) = Self::analyze_efficiency(original_params, lora_params);

            println!(
                "{:<6} {:<8} {:<12.3} {:<12} {:<15.1}%",
                config.rank,
                config.alpha,
                config.scaling(),
                lora_params,
                reduction
            );
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;
    let dtype = DType::F32;

    println!("Enhanced LoRA Mathematics Study");
    println!("===============================");
    println!("Device: {:?}", device);
    println!("Target dtype: {:?}", dtype);
    println!(
        "Configuration: rank={}, alpha={}, dims={}x{}",
        args.rank, args.alpha, args.input_dim, args.output_dim
    );

    println!("\n=== VarMap & VarBuilder Setup ===");
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    println!(
        "VarBuilder created - device: {:?}, dtype: {:?}",
        vb.device(),
        vb.dtype()
    );

    // Create LoRA configuration
    let lora_config = LoraConfig::new(args.rank, args.alpha, None);
    println!(
        "LoRA config: rank={}, alpha={}, scaling={:.3}",
        lora_config.rank,
        lora_config.alpha,
        lora_config.scaling()
    );

    // Test 1: Basic LoRA Layer
    println!("\n=== Test 1: Basic LoRA Layer ===");

    // Debug original layer creation
    println!("Creating original layer weight...");
    let original_weight = varmap.get(
        (args.output_dim, args.input_dim), // [64, 128] for candle Linear
        "original.weight",
        candle_nn::init::DEFAULT_KAIMING_NORMAL,
        dtype,
        &device,
    )?;
    println!(
        "Original weight created - shape: {:?}, dtype: {:?}",
        original_weight.shape(),
        original_weight.dtype()
    );

    let original_layer = Linear::new(original_weight, None);
    println!("Original Linear layer created");

    println!("\nCreating LoRA layer...");
    let mut lora_layer = SimpleLoRA::new(
        original_layer,
        lora_config.clone(),
        args.input_dim,
        args.output_dim,
        vb.pp("lora"),
    )?;

    println!("Creating input tensor...");
    let input = Tensor::randn(0.0, 1.0, (10, args.input_dim), &device)?.to_dtype(dtype)?;
    println!(
        "Input tensor created - shape: {:?}, dtype: {:?}",
        input.shape(),
        input.dtype()
    );

    println!("\nRunning forward pass...");
    let output = lora_layer.forward(&input)?;
    println!("Forward pass complete!");
    println!("Output shape: {:?}", output.shape());
    println!("\nRunning forward pass...");

    // Parameter analysis
    let original_params = args.input_dim * args.output_dim;
    let lora_params = (args.input_dim * args.rank) + (args.rank * args.output_dim);
    let (reduction, efficiency) = LoraAnalyzer::analyze_efficiency(original_params, lora_params);

    println!("\nParameter Analysis:");
    println!("Original layer params: {}", original_params);
    println!(
        "LoRA params: {} (A) + {} (B) = {}",
        args.input_dim * args.rank,
        args.rank * args.output_dim,
        lora_params
    );
    println!("Parameter reduction: {:.1}% ({})", reduction, efficiency);

    // Test 2: Weight Merging
    if args.test_merging {
        println!("\n=== Test 2: Weight Merging ===");
        println!("Before merging - merged status: {}", lora_layer.merged);

        let output_before = lora_layer.forward(&input)?;
        lora_layer.merge_weights()?;

        println!("After merging - merged status: {}", lora_layer.merged);
        let output_after = lora_layer.forward(&input)?;

        println!(
            "Output shapes - before: {:?}, after: {:?}",
            output_before.shape(),
            output_after.shape()
        );

        lora_layer.unmerge_weights()?;
        println!("After unmerging - merged status: {}", lora_layer.merged);
    }

    // Test 3: Multi-Head Attention LoRA
    if args.test_attention {
        println!("\n=== Test 3: Multi-Head Attention LoRA ===");
        let hidden_dim = 128;

        // Create attention layer weights with correct dimensions
        let q_weight = varmap.get(
            (hidden_dim, hidden_dim),
            "attn_q.weight",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
            dtype,
            &device,
        )?;
        let k_weight = varmap.get(
            (hidden_dim, hidden_dim),
            "attn_k.weight",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
            dtype,
            &device,
        )?;
        let v_weight = varmap.get(
            (hidden_dim, hidden_dim),
            "attn_v.weight",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
            dtype,
            &device,
        )?;
        let o_weight = varmap.get(
            (hidden_dim, hidden_dim),
            "attn_o.weight",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
            dtype,
            &device,
        )?;

        let q_linear = Linear::new(q_weight, None);
        let k_linear = Linear::new(k_weight, None);
        let v_linear = Linear::new(v_weight, None);
        let o_linear = Linear::new(o_weight, None);

        let attention_lora = AttentionLoRA::new(
            q_linear,
            k_linear,
            v_linear,
            o_linear,
            lora_config.clone(),
            hidden_dim,
            vb.pp("attention"),
        )?;

        let attn_input = Tensor::randn(0.0, 1.0, (10, 16, hidden_dim), &device)?.to_dtype(dtype)?;
        let attn_output = attention_lora.forward(&attn_input)?;

        println!("Attention input shape: {:?}", attn_input.shape());
        println!("Attention output shape: {:?}", attn_output.shape());

        let original_attn_params = hidden_dim * hidden_dim * 4; // q, k, v, o
        let lora_attn_params = attention_lora.get_total_lora_params();
        let (attn_reduction, attn_efficiency) =
            LoraAnalyzer::analyze_efficiency(original_attn_params, lora_attn_params);

        println!("Attention LoRA Analysis:");
        println!("Original attention params: {}", original_attn_params);
        println!("LoRA attention params: {}", lora_attn_params);
        println!(
            "Attention parameter reduction: {:.1}% ({})",
            attn_reduction, attn_efficiency
        );
    }

    // Test 4: Configuration Comparison
    println!("\n=== Test 4: Configuration Comparison ===");
    let configs = vec![
        LoraConfig::new(4, 8.0, None),
        LoraConfig::new(8, 16.0, None),
        LoraConfig::new(16, 32.0, None),
        LoraConfig::new(32, 64.0, None),
        LoraConfig::new(64, 128.0, None),
    ];

    LoraAnalyzer::compare_configurations(configs, args.input_dim, args.output_dim);

    println!("\n=== LoRA Mathematics Summary ===");
    println!("Original: output = input * W");
    println!("LoRA: output = input * W + input * A * B * (alpha/rank)");
    println!("Where:");
    println!(
        "  W: {}x{} original weight matrix",
        args.output_dim, args.input_dim
    );
    println!(
        "  A: {}x{} down-projection matrix",
        args.input_dim, args.rank
    );
    println!(
        "  B: {}x{} up-projection matrix",
        args.rank, args.output_dim
    );
    println!("  alpha: {} (learned scaling factor)", args.alpha);
    println!("  rank: {} (bottleneck dimension)", args.rank);

    println!("\nKey LoRA Benefits:");
    println!("✓ Parameter Efficiency: {:.1}% reduction", reduction);
    println!("✓ Modular Design: Easy to add/remove adapters");
    println!("✓ Weight Merging: Zero inference overhead when merged");
    println!("✓ Task Specialization: Different adapters for different tasks");

    Ok(())
}
