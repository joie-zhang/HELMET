#!/usr/bin/env python3
"""
Filter priority jobs based on:
1. Currently queued jobs (exclude)
2. Priority: summ_multilex first, then StreamingLLM 8k
"""

import os
import re
from pathlib import Path

# Currently queued job patterns from squeue output
QUEUED_JOBS = [
    # Extract patterns from job names
    "snapkv_longproc_snapkv_k7_w256_c2048_maxpool_Qwen2_5_7B_Instruct_travel_planning",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_Yarn_Qwen3_8B_thinking_pseudo_to_code",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_Yarn_Qwen3_8B_pseudo_to_code",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_Yarn_Qwen3_8B_nothinking_travel_planning",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_Yarn_Qwen3_8B_nothinking_pseudo_to_code",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_Yarn_Qwen3_8B_nothinking_html_to_tsv",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_Yarn_Qwen3_8B_html_to_tsv",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_Qwen3_8B_travel_planning",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_Qwen3_8B_thinking_travel_planning",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_Qwen3_8B_thinking_html_to_tsv",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_Qwen3_8B_nothinking_travel_planning",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_Qwen3_8B_nothinking_html_to_tsv",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_Qwen3_8B_html_to_tsv",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_Qwen2_5_7B_Instruct_html_to_tsv",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_Llama_3_1_8B_Instruct_travel_planning",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_Llama_3_1_8B_Instruct_html_to_tsv",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_DeepSeek_R1_Distill_Qwen_7B_travel_planning",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_DeepSeek_R1_Distill_Qwen_7B_html_to_tsv",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_DeepSeek_R1_Distill_Llama_8B_travel_planning",
    "pyramidkv_longproc_pyramidkv_k7_w2048_c8192_avgpool_Yarn_Qwen3_8B_travel_planning",
    "pyramidkv_longproc_pyramidkv_k7_w256_c2048_avgpool_DeepSeek_R1_Distill_Llama_8B_html_to_tsv",
    "pyramidkv_longproc_pyramidkv_k7_w2048_c8192_avgpool_Yarn_Qwen3_8B_thinking_travel_planning",
    "pyramidkv_longproc_pyramidkv_k7_w2048_c8192_avgpool_Yarn_Qwen3_8B_nothinking_travel_planning",
    "pyramidkv_longproc_pyramidkv_k7_w2048_c8192_avgpool_Qwen3_8B_travel_planning",
    "pyramidkv_longproc_pyramidkv_k7_w2048_c8192_avgpool_Qwen3_8B_thinking_travel_planning",
    "pyramidkv_longproc_pyramidkv_k7_w2048_c8192_avgpool_Qwen3_8B_nothinking_travel_planning",
    "pyramidkv_longproc_pyramidkv_k7_w2048_c8192_avgpool_Llama_3_1_8B_Instruct_travel_planning",
    "pyramidkv_longproc_pyramidkv_k7_w2048_c8192_avgpool_Qwen2_5_7B_Instruct_html_to_tsv",
    "pyramidkv_longproc_pyramidkv_k7_w2048_c8192_avgpool_DeepSeek_R1_Distill_Qwen_7B_travel_planning",
    "pyramidkv_longproc_pyramidkv_k7_w2048_c8192_avgpool_DeepSeek_R1_Distill_Llama_8B_travel_planning",
    "baseline_longproc_int8_quant_Yarn_Qwen3_8B_thinking_travel_planning",
    "baseline_longproc_int8_quant_Yarn_Qwen3_8B_travel_planning",
    "baseline_longproc_int8_quant_Yarn_Qwen3_8B_thinking_html_to_tsv",
    "baseline_longproc_int8_quant_Yarn_Qwen3_8B_nothinking_travel_planning",
    "baseline_longproc_int8_quant_Yarn_Qwen3_8B_html_to_tsv",
    "baseline_longproc_int8_quant_Yarn_Qwen3_8B_nothinking_html_to_tsv",
    "baseline_longproc_int8_quant_Qwen3_8B_thinking_travel_planning",
    "baseline_longproc_int8_quant_Qwen3_8B_travel_planning",
    "baseline_longproc_int8_quant_Qwen3_8B_thinking_html_to_tsv",
    "baseline_longproc_int8_quant_Qwen3_8B_nothinking_travel_planning",
    "baseline_longproc_int8_quant_Qwen3_8B_nothinking_html_to_tsv",
    "baseline_longproc_int8_quant_Qwen3_8B_html_to_tsv",
    "baseline_longproc_int8_quant_DeepSeek_R1_Distill_Qwen_7B_travel_planning",
    "baseline_longproc_int8_quant_DeepSeek_R1_Distill_Qwen_7B_html_to_tsv",
    "baseline_longproc_int8_quant_DeepSeek_R1_Distill_Llama_8B_travel_planning",
    "baseline_longproc_int4_quant_Yarn_Qwen3_8B_thinking_html_to_tsv",
    "baseline_longproc_int4_quant_Yarn_Qwen3_8B_nothinking_html_to_tsv",
    "baseline_longproc_int4_quant_Yarn_Qwen3_8B_html_to_tsv",
    "baseline_longproc_baseline_Yarn_Qwen3_8B_thinking_html_to_tsv",
    "baseline_longproc_baseline_Yarn_Qwen3_8B_nothinking_html_to_tsv",
    "baseline_longproc_baseline_Yarn_Qwen3_8B_html_to_tsv",
]

def normalize_job_name(job_name):
    """
    Normalize job names to match between squeue output and config files
    - Remove prefixes like 'baseline_longproc_' or 'pyramidkv_longproc_'
    - Standardize model names
    """
    # Remove common prefixes
    job_name = re.sub(r'^(baseline|pyramidkv|snapkv|streamingllm)_(longproc|helmet)_', '', job_name)

    # Standardize model name variations
    job_name = job_name.replace('Yarn_Qwen3_8B', 'Yarn-Qwen3-8B')
    job_name = job_name.replace('Qwen3_8B', 'Qwen3-8B')
    job_name = job_name.replace('Qwen2_5_7B_Instruct', 'Qwen2.5-7B-Instruct')
    job_name = job_name.replace('Llama_3_1_8B_Instruct', 'Llama-3.1-8B-Instruct')
    job_name = job_name.replace('DeepSeek_R1_Distill_Qwen_7B', 'DeepSeek-R1-Distill-Qwen-7B')
    job_name = job_name.replace('DeepSeek_R1_Distill_Llama_8B', 'DeepSeek-R1-Distill-Llama-8B')

    return job_name

def is_job_queued(config_file):
    """Check if this config corresponds to a currently queued job"""
    config_name = os.path.basename(config_file).replace('.sh', '').replace('_config', '')
    normalized_config = normalize_job_name(config_name)

    for queued in QUEUED_JOBS:
        normalized_queued = normalize_job_name(queued)
        # Check if the normalized queued job pattern is in the config name
        if normalized_queued in normalized_config or normalized_config in normalized_queued:
            return True
    return False

def should_exclude_thinking_variant(config_file):
    """
    Exclude -thinking variants for Qwen3-8B and Yarn-Qwen3-8B
    since Qwen3-8B-thinking = Qwen3-8B and Yarn-Qwen3-8B-thinking = Yarn-Qwen3-8B
    """
    config_name = os.path.basename(config_file)

    # Exclude if it contains thinking variant of Qwen3-8B or Yarn-Qwen3-8B
    if 'Qwen3_8B_thinking' in config_name or 'Yarn_Qwen3_8B_thinking' in config_name:
        return True

    return False

def get_priority_jobs():
    """Get prioritized list of jobs to submit"""
    config_dir = Path("/scratch/gpfs/DANQIC/jz4391/HELMET/scripts/configs/missing_experiments_sweep")

    # Priority 1: summ_multilex
    summ_multilex_files = sorted(config_dir.glob("*summ_multilex*.sh"))

    # Priority 2: StreamingLLM 8k
    streamingllm_8k_files = sorted(config_dir.glob("*streamingllm*8k*.sh"))

    # Filter out:
    # 1. Queued jobs
    # 2. -thinking variants for Qwen3-8B and Yarn-Qwen3-8B
    priority1 = [f for f in summ_multilex_files
                 if not is_job_queued(str(f)) and not should_exclude_thinking_variant(str(f))]
    priority2 = [f for f in streamingllm_8k_files
                 if not is_job_queued(str(f)) and not should_exclude_thinking_variant(str(f))]

    return priority1, priority2

def main():
    priority1, priority2 = get_priority_jobs()

    print("="*80)
    print("PRIORITY JOB ANALYSIS")
    print("="*80)
    print()

    print(f"Priority 1: summ_multilex experiments")
    print(f"  Total configs: 130")
    print(f"  After filtering:")
    print(f"    - Excluded queued jobs")
    print(f"    - Excluded Qwen3-8B-thinking variants (equivalent to Qwen3-8B)")
    print(f"    - Excluded Yarn-Qwen3-8B-thinking variants (equivalent to Yarn-Qwen3-8B)")
    print(f"  Final count: {len(priority1)}")
    print()

    print(f"Priority 2: StreamingLLM 8k experiments")
    print(f"  Total configs: 60")
    print(f"  After filtering:")
    print(f"    - Excluded queued jobs")
    print(f"    - Excluded Qwen3-8B-thinking variants (equivalent to Qwen3-8B)")
    print(f"    - Excluded Yarn-Qwen3-8B-thinking variants (equivalent to Yarn-Qwen3-8B)")
    print(f"  Final count: {len(priority2)}")
    print()

    # Show sample of priority 1 jobs
    print("="*80)
    print("SAMPLE PRIORITY 1 JOBS (summ_multilex) - First 20:")
    print("="*80)
    for f in priority1[:20]:
        print(f"  {f.name}")
    if len(priority1) > 20:
        print(f"  ... and {len(priority1) - 20} more")
    print()

    # Show sample of priority 2 jobs
    print("="*80)
    print("SAMPLE PRIORITY 2 JOBS (StreamingLLM 8k) - First 20:")
    print("="*80)
    for f in priority2[:20]:
        print(f"  {f.name}")
    if len(priority2) > 20:
        print(f"  ... and {len(priority2) - 20} more")
    print()

    # Write filtered lists to files
    with open('/scratch/gpfs/DANQIC/jz4391/HELMET/priority1_summ_multilex.txt', 'w') as f:
        for config in priority1:
            f.write(f"{config}\n")

    with open('/scratch/gpfs/DANQIC/jz4391/HELMET/priority2_streamingllm_8k.txt', 'w') as f:
        for config in priority2:
            f.write(f"{config}\n")

    print("="*80)
    print("Output files created:")
    print("  - priority1_summ_multilex.txt")
    print("  - priority2_streamingllm_8k.txt")
    print("="*80)

if __name__ == "__main__":
    main()
