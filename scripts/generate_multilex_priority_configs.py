#!/usr/bin/env python3
"""
Generate config files for high-priority summ_multilex experiments
These are the 20 critical experiments needed to complete the plots.
"""

import os
from pathlib import Path

# Create output directory
config_dir = Path("scripts/configs/multilex_priority")
config_dir.mkdir(parents=True, exist_ok=True)

# Define the high-priority experiments
priority_experiments = []

# 1. Baseline, INT4, INT8 for both DeepSeek models (16k only)
for model in ["DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Llama-8B"]:
    for technique, quant_type in [("baseline", None), ("INT4", "4bit"), ("INT8", "8bit")]:
        priority_experiments.append({
            "technique": technique,
            "model": model,
            "context": "16k",
            "task": "summ_multilex",
            "quant_type": quant_type,
            "cache_config": None
        })

# 2. SnapKV - w256_c2048_k7_maxpool and w2048_c8192_k7_maxpool for all 4 models (16k)
snapkv_models = [
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-7B"
]
for model in snapkv_models:
    for cache_config in ["w256_c2048_k7_maxpool", "w2048_c8192_k7_maxpool"]:
        priority_experiments.append({
            "technique": "snapkv",
            "model": model,
            "context": "16k",
            "task": "summ_multilex",
            "quant_type": None,
            "cache_config": cache_config
        })

# 3. PyramidKV - w256_c2048_k7_avgpool and w2048_c8192_k7_avgpool for 3 models (16k)
# DeepSeek-R1-Distill-Llama-8B w2048 already has data, so we need 5 experiments
pyramidkv_configs = [
    ("DeepSeek-R1-Distill-Qwen-7B", "w256_c2048_k7_avgpool"),
    ("DeepSeek-R1-Distill-Qwen-7B", "w2048_c8192_k7_avgpool"),
    ("Qwen2.5-7B-Instruct", "w256_c2048_k7_avgpool"),
    ("Qwen2.5-7B-Instruct", "w2048_c8192_k7_avgpool"),
    ("Llama-3.1-8B-Instruct", "w256_c2048_k7_avgpool"),
    ("Llama-3.1-8B-Instruct", "w2048_c8192_k7_avgpool"),
]
for model, cache_config in pyramidkv_configs:
    priority_experiments.append({
        "technique": "pyramidkv",
        "model": model,
        "context": "16k",
        "task": "summ_multilex",
        "quant_type": None,
        "cache_config": cache_config
    })

print(f"Generating {len(priority_experiments)} high-priority multilex config files...")

# Template for config file
def generate_config_content(exp):
    """Generate the config file content in the format expected by submit_job.sh"""
    technique = exp["technique"]
    model = exp["model"]
    context = exp["context"]
    task = exp["task"]
    quant_type = exp["quant_type"]
    cache_config = exp["cache_config"]

    # Determine quantization value for QUANTIZE array
    if technique == "baseline":
        quantize_val = "16"  # Default baseline is 16-bit
    elif technique == "INT4":
        quantize_val = "4"
        technique = "baseline"  # INT4/INT8 are treated as baseline with different quantization
    elif technique == "INT8":
        quantize_val = "8"
        technique = "baseline"
    else:
        quantize_val = "16"  # Non-quantized techniques use 16-bit

    # Generate job name
    model_short = model.replace("-", "_")
    job_name = f"{technique}_{model_short}_{task}_{context}"
    if cache_config:
        job_name += f"_{cache_config}"

    # Base config in array format expected by submit_job.sh
    lines = [
        "# Inference technique evaluation config",
        f"# Technique: {technique}, Model: {model}, Task: {task}, Context: {context}",
        f'declare -a BASE_CONFIGS=("{task}")',
        f'declare -a CONTEXT_LENGTHS=("{context}")',
        f'declare -a MODELS=("{model}")',
        f'declare -a QUANTIZE=("{quantize_val}")',
        f'EXP_TYPE="{technique}"',
        'BENCHMARK="helmet"',
        'USE_REASONING_CONFIG="false"',
        'SEED=42',
        "",
    ]

    # Add KV cache parameters for snapkv and pyramidkv
    if cache_config:
        if technique == "snapkv":
            # w256_c2048_k7_maxpool -> window=256, cache=2048, kernel=7, pooling=maxpool
            parts = cache_config.split('_')
            window = parts[0].replace('w', '')
            cache_size = parts[1].replace('c', '')
            kernel = parts[2].replace('k', '')
            pooling = parts[3]

            lines.extend([
                f'KV_TYPE="snapkv"',
                f'WINDOW_SIZE="{window}"',
                f'MAX_CAPACITY_PROMPT="{cache_size}"',
                f'KERNEL_SIZE="{kernel}"',
                f'POOLING="{pooling}"',
                "",
            ])

        elif technique == "pyramidkv":
            # w256_c2048_k7_avgpool -> window=256, cache=2048, kernel=7, pooling=avgpool
            parts = cache_config.split('_')
            window = parts[0].replace('w', '')
            cache_size = parts[1].replace('c', '')
            kernel = parts[2].replace('k', '')
            pooling = parts[3]

            lines.extend([
                f'KV_TYPE="pyramidkv"',
                f'WINDOW_SIZE="{window}"',
                f'MAX_CAPACITY_PROMPT="{cache_size}"',
                f'KERNEL_SIZE="{kernel}"',
                f'POOLING="{pooling}"',
                "",
            ])

    # SLURM Configuration - 1 hour for summ_multilex
    lines.extend([
        "",
        "# SLURM Configuration",
        'JOB_TIME="01:00:00"',
        f'JOB_NAME="{job_name}"',
        "",
        "# Export variables",
        "export BASE_CONFIGS",
        "export CONTEXT_LENGTHS",
        "export MODELS",
        "export QUANTIZE",
        "export EXP_TYPE",
        "export BENCHMARK",
        "export USE_REASONING_CONFIG",
        "export SEED",
    ])

    # Export KV cache parameters if they exist
    if cache_config:
        lines.extend([
            "export KV_TYPE",
            "export WINDOW_SIZE",
            "export MAX_CAPACITY_PROMPT",
            "export KERNEL_SIZE",
            "export POOLING",
        ])

    return "\n".join(lines) + "\n"

# Generate all config files
generated_count = 0
for exp in priority_experiments:
    technique = exp["technique"]
    model = exp["model"].replace("/", "_")
    context = exp["context"]
    cache_config = exp["cache_config"] if exp["cache_config"] else "default"

    # Create filename
    filename = f"{technique}_{model}_{context}_{cache_config}.sh"
    filepath = config_dir / filename

    # Write config file
    content = generate_config_content(exp)
    with open(filepath, 'w') as f:
        f.write(content)

    # Make executable
    os.chmod(filepath, 0o755)

    generated_count += 1
    print(f"  Generated: {filename}")

print(f"\n✅ Generated {generated_count} config files in {config_dir}")
print(f"\nNext steps:")
print(f"  1. Review configs: ls -la {config_dir}/")
print(f"  2. Submit jobs: ./scripts/submit_multilex_priority.sh")
