# High-Priority summ_multilex Experiments - Submission Guide

## Overview

This guide helps you submit the **20 critical experiments** needed to complete the summ_multilex data in your task delta plots.

## What's Missing?

Based on analysis of `results/helmet_results/helmet_performance.csv`, 108 experiments are missing summ_multilex data. However, only **20 critical experiments** are needed to complete the main plotting configurations:

### Critical Missing Experiments (20 total):

1. **Baseline/INT4/INT8 (6 experiments)** - DeepSeek models, 16k context
   - DeepSeek-R1-Distill-Qwen-7B: baseline, INT4, INT8
   - DeepSeek-R1-Distill-Llama-8B: baseline, INT4, INT8

2. **SnapKV (8 experiments)** - All 4 models, 16k context, 2 cache configs
   - All 4 models × 2 configs (w256_c2048_k7_maxpool, w2048_c8192_k7_maxpool)

3. **PyramidKV (6 experiments)** - 3 models, 16k context, 2 cache configs
   - DeepSeek-R1-Distill-Qwen-7B: w256 and w2048 configs
   - Qwen2.5-7B-Instruct: w256 and w2048 configs
   - Llama-3.1-8B-Instruct: w256 and w2048 configs

## Setup (Already Done!)

The scripts have been generated and are ready to use:

```bash
# Config files generated in:
scripts/configs/multilex_priority/

# Submission script:
scripts/submit_multilex_priority.sh
```

**Important:** All config files now include:
- ✅ Correct array-based format (compatible with `submit_job.sh`)
- ✅ `JOB_TIME="01:00:00"` (1 hour runtime for each job)
- ✅ Proper variable exports and KV cache parameters
- ✅ INT4/INT8 handled as baseline with `QUANTIZE=("4")` and `QUANTIZE=("8")`

## How to Submit Jobs

### Option 1: Submit All 20 Jobs at Once

```bash
./scripts/submit_multilex_priority.sh
```

### Option 2: Test with One Job First (Recommended)

```bash
# Submit 1 test job (1 hour time limit)
./scripts/submit_multilex_priority.sh --test

# Monitor the test job
squeue -u $USER

# If successful, submit all remaining jobs
./scripts/submit_multilex_priority.sh
```

### Option 3: Submit by Technique (Phased Approach)

```bash
# Submit baseline/INT4/INT8 first (6 jobs - fastest)
./scripts/submit_multilex_priority.sh --baseline

# Then submit SnapKV (8 jobs)
./scripts/submit_multilex_priority.sh --snapkv

# Finally submit PyramidKV (6 jobs)
./scripts/submit_multilex_priority.sh --pyramidkv

# Or combine multiple techniques:
./scripts/submit_multilex_priority.sh --baseline --snapkv
```

### Option 4: Dry Run (Preview Without Submitting)

```bash
# See what would be submitted without actually submitting
./scripts/submit_multilex_priority.sh --dry-run

# Dry run with filters
./scripts/submit_multilex_priority.sh --dry-run --baseline
```

## After Job Completion

Once jobs finish, follow these steps:

### 1. Run GPT-4 Evaluation

The summ_multilex task requires GPT-4 evaluation to compute F1 scores:

```bash
python scripts/eval_gpt4_summ.py
```

This will create `-gpt4eval_o.json` files with GPT-4 F1 scores.

### 2. Re-collect Results

The updated `collect_results_new.py` script now properly reads the `-gpt4eval_o.json` files:

```bash
python scripts/collect_results_new.py
```

This will populate the summ_multilex column in:
- `results/helmet_results/helmet_performance.csv`

### 3. Regenerate Plots

```bash
# Generate both task delta plots with complete data
python scripts/plot_task_deltas_averaged_configs.py
python scripts/plot_task_deltas_separate_configs.py
```

The multi_lexsum bars should now be fully populated!

## Monitoring Jobs

```bash
# Check all your jobs
squeue -u $USER

# Check specific job IDs (displayed after submission)
squeue -j <job_id1>,<job_id2>,...

# Cancel all submitted jobs (if needed)
scancel <job_id1> <job_id2> ...
```

## Expected Output Locations

Results will be saved to:
```
/scratch/gpfs/DANQIC/jz4391/HELMET/output/
├── baseline/16k/DeepSeek-R1-Distill-Qwen-7B/
├── baseline/16k/DeepSeek-R1-Distill-Llama-8B/
├── snapkv/16k/Llama-3.1-8B-Instruct/w256_c2048_k7_maxpool/
├── snapkv/16k/Llama-3.1-8B-Instruct/w2048_c8192_k7_maxpool/
├── pyramidkv/16k/Qwen2.5-7B-Instruct/w256_c2048_k7_avgpool/
└── ... (and more)
```

## Troubleshooting

### If a job fails:

1. Check the log file in `joblog/`
2. Review the config file in `scripts/configs/multilex_priority/`
3. Resubmit just that specific technique:
   ```bash
   # Resubmit just baseline jobs
   ./scripts/submit_multilex_priority.sh --baseline
   ```

### If collect_results_new.py doesn't find summ_multilex data:

1. Verify `-gpt4eval_o.json` files exist in the output directories
2. Run `eval_gpt4_summ.py` to generate them if missing
3. Check that the files contain `averaged_metrics["gpt-4-f1"]` field

## Summary

**Quick Start (Recommended):**
```bash
# 1. Test with one job
./scripts/submit_multilex_priority.sh --test

# 2. If test succeeds, submit all
./scripts/submit_multilex_priority.sh

# 3. After jobs complete, run evaluation
python scripts/eval_gpt4_summ.py

# 4. Re-collect results
python scripts/collect_results_new.py

# 5. Regenerate plots
python scripts/plot_task_deltas_averaged_configs.py
python scripts/plot_task_deltas_separate_configs.py
```

That's it! These 20 experiments will give you complete summ_multilex data for your main plotting configurations.
