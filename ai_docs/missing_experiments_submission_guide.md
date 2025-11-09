# Missing Experiments Submission Guide

This guide documents how to generate and submit the missing experiments for the HELMET project, including all filtering options and strategies for managing large job submissions.

## Overview

The missing experiments system fills in gaps from the helmet_performance.csv results, covering:
- **StreamingLLM corrections** (cache size 4092+4 instead of 3968+128)
- **StreamingLLM new suite** (2044+4 cache size)
- **PyramidKV & SnapKV** missing 32k runs
- **Quantization** (INT4, INT8, baseline) for missing models/contexts
- **LongProc** complete suite (0.5k, 2k, 8k)
- **Qwen3-8B models** with both thinking modes (enable_thinking=True/False)

## Quick Start

```bash
# 1. Generate all config files
./scripts/generate_missing_experiments.sh

# 2. Preview what will be submitted (16k HELMET + 0.5k LongProc)
./scripts/submit_missing_experiments.sh --dry-run --helmet-16k --longproc-0.5k

# 3. Test with one job
./scripts/submit_missing_experiments.sh --test

# 4. Submit shorter jobs first (to avoid overwhelming SLURM)
./scripts/submit_missing_experiments.sh --helmet-16k --longproc-0.5k

# 5. Later, submit longer jobs
./scripts/submit_missing_experiments.sh --helmet-32k --longproc-2k --longproc-8k
```

## Scripts

### 1. `generate_missing_experiments.sh`

Generates config files for all missing experiments.

**Location:** `scripts/generate_missing_experiments.sh`

**What it generates:**
- StreamingLLM cache size corrections (4092+4)
- StreamingLLM new suite (2044+4)
- PyramidKV 32k runs (w256_c2048 and w2048_c8192)
- SnapKV 32k runs (w256_c2048 and w2048_c8192)
- Quantization missing runs (baseline, INT4, INT8)
- Complete LongProc suite (all models × all tasks × 0.5k/2k/8k)
- Qwen3-8B & Yarn-Qwen3-8B with BOTH thinking modes

**Output:** `scripts/configs/missing_experiments_sweep/`

**Job time allocations:**
- HELMET 32k: 18 hours
- HELMET 16k: 6 hours
- LongProc 8k: 18 hours
- LongProc 2k: 12 hours
- LongProc 0.5k: 6 hours

**Special handling:**
- All configs use `USE_REASONING_CONFIG=false` (standard configs, NOT configs_reasoning)
- Qwen3 models use `ENABLE_THINKING` parameter to control thinking mode:
  - `ENABLE_THINKING=True` → Verbose chain-of-thought reasoning
  - `ENABLE_THINKING=False` → No chain-of-thought
- Config filenames include `_thinking` or `_nothinking` suffix for Qwen3 models

### 2. `submit_missing_experiments.sh`

Submits jobs with flexible filtering options.

**Location:** `scripts/submit_missing_experiments.sh`

**Usage:**
```bash
./scripts/submit_missing_experiments.sh [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--test` or `-t` | Submit only 1 job for testing (automatically selects a Qwen3 thinking mode config with 1 hour time limit) |
| `--dry-run` or `-d` | Show what would be submitted without actually submitting |

**Additive Filters (combine multiple to select specific subsets):**

| Option | Description |
|--------|-------------|
| `--helmet-16k` | Include 16k HELMET jobs |
| `--helmet-32k` | Include 32k HELMET jobs |
| `--longproc-0.5k` | Include 0.5k LongProc jobs |
| `--longproc-2k` | Include 2k LongProc jobs |
| `--longproc-8k` | Include 8k LongProc jobs |

**Important:** Filters are **additive** - you can combine multiple flags to select exactly what you want. If no filters are specified, all jobs are submitted.

**Examples:**
```bash
# Submit only 16k HELMET jobs
./scripts/submit_missing_experiments.sh --helmet-16k

# Submit 16k HELMET + 0.5k LongProc
./scripts/submit_missing_experiments.sh --helmet-16k --longproc-0.5k

# Submit all HELMET (16k + 32k)
./scripts/submit_missing_experiments.sh --helmet-16k --helmet-32k

# Submit short LongProc jobs only
./scripts/submit_missing_experiments.sh --longproc-0.5k --longproc-2k
```

**Common Combinations:**

| Command | What Gets Submitted | Typical Use Case |
|---------|---------------------|------------------|
| No flags | Everything | Final comprehensive submission |
| `--helmet-16k` | 16k HELMET only | Test HELMET before 32k |
| `--helmet-32k` | 32k HELMET only | Submit long HELMET jobs separately |
| `--helmet-16k --helmet-32k` | All HELMET | Focus on HELMET benchmarks |
| `--longproc-0.5k` | 0.5k LongProc only | Quick LongProc tasks |
| `--longproc-0.5k --longproc-2k --longproc-8k` | All LongProc | Focus on LongProc benchmarks |
| `--helmet-16k --longproc-0.5k` | Short jobs only | Avoid overwhelming SLURM |
| `--helmet-32k --longproc-8k` | Long jobs only | Submit after short jobs complete |

**Features:**
- Groups jobs by category in summary
- Provides `squeue` and `scancel` commands for monitoring
- Shows reminders about remaining jobs when using filters
- Tracks job IDs for all submitted jobs

## Recommended Workflow

### Strategy 1: Phased Submission by Job Duration (Recommended)

Avoid overwhelming SLURM by submitting shorter jobs first:

```bash
# Phase 1: Short jobs (16k HELMET + 0.5k LongProc)
./scripts/submit_missing_experiments.sh --helmet-16k --longproc-0.5k

# Monitor progress
squeue -u $USER

# Phase 2: Medium duration jobs (2k LongProc)
./scripts/submit_missing_experiments.sh --longproc-2k

# Phase 3: Longer jobs (32k HELMET + 8k LongProc)
./scripts/submit_missing_experiments.sh --helmet-32k --longproc-8k
```

### Strategy 2: Separate HELMET and LongProc

```bash
# Submit all HELMET jobs (16k and 32k)
./scripts/submit_missing_experiments.sh --helmet-16k --helmet-32k

# Later, submit all LongProc jobs
./scripts/submit_missing_experiments.sh --longproc-0.5k --longproc-2k --longproc-8k
```

### Strategy 3: Gradual Submission

Start small, scale up:

```bash
# Start with fastest jobs
./scripts/submit_missing_experiments.sh --helmet-16k

# Add more as capacity allows
./scripts/submit_missing_experiments.sh --longproc-0.5k

# Eventually submit the longer jobs
./scripts/submit_missing_experiments.sh --helmet-32k
./scripts/submit_missing_experiments.sh --longproc-8k
```

## Testing Before Full Submission

Always test with a single job before submitting hundreds:

```bash
# Test mode automatically selects a Qwen3 nothinking config
# Time is overridden to 1 hour for quick testing
./scripts/submit_missing_experiments.sh --test

# Monitor the test job
squeue -u $USER
watch -n 10 squeue -u $USER

# Check logs
tail -f joblog/[job-name]-[job-id]_0.out

# If successful, proceed with full submission
./scripts/submit_missing_experiments.sh --exclude-32k
```

**Test mode features:**
- Automatically selects a Qwen3-8B or Yarn-Qwen3-8B config with `enable_thinking=True` (thinking mode)
- Overrides job time to 1 hour (regardless of config's original time)
- Tests the new `ENABLE_THINKING` parameter functionality
- Confirms that thinking mode uses regular `configs/` folder (NOT `configs_reasoning/`)

## Understanding Qwen3 Thinking Modes

### What is `enable_thinking`?

For Qwen3-8B and Yarn-Qwen3-8B models, `enable_thinking` controls whether the model uses verbose chain-of-thought reasoning during inference.

**Implementation:**
- Set in `tokenizer.apply_chat_template()` via the `enable_thinking` parameter
- When `True`: Model outputs thinking process (enclosed in `<think>` tags) before final answer
- When `False`: Model outputs answer directly without thinking process

**Existing vs New Runs:**
- **Existing runs** (from helmet_performance.csv): Used `enable_thinking=True` (thinking mode)
- **New configs**: Generate BOTH modes for complete comparison

**Config naming:**
- `*_thinking_*`: `ENABLE_THINKING=True`
- `*_nothinking_*`: `ENABLE_THINKING=False`

**Important distinction:**
- `USE_REASONING_CONFIG` → Controls which config folder (configs vs configs_reasoning)
- `ENABLE_THINKING` → Controls Qwen3's internal CoT mode (thinking vs nothinking)
- ALL new configs use `USE_REASONING_CONFIG=false` regardless of thinking mode

## Missing Experiments Breakdown

### 1. StreamingLLM Cache Size Corrections (4092+4)

**Why:** Existing runs used 3968+128, need to re-run with correct 4092+4

**Models:**
- Qwen2.5-7B-Instruct
- Llama-3.1-8B-Instruct

**Contexts:** 16k, 32k

**Tasks:** All HELMET tasks

### 2. StreamingLLM New Suite (2044+4)

**Why:** New cache size configuration to test

**Models:** All 6 models

**Contexts:** 16k, 32k

**Tasks:** All HELMET tasks

### 3. StreamingLLM Missing 32k

**Models:**
- DeepSeek-R1-Distill-Qwen-7B
- DeepSeek-R1-Distill-Llama-8B
- Qwen3-8B
- Yarn-Qwen3-8B

**Tasks:** All HELMET tasks

### 4. PyramidKV & SnapKV Missing 32k

**Configs:**
- w256_c2048_k7 (avgpool for PyramidKV, maxpool for SnapKV)
- w2048_c8192_k7 (avgpool for PyramidKV, maxpool for SnapKV)

**Models:** All 6 models

**Tasks:** All HELMET tasks

### 5. Quantization Missing Runs

**16k:**
- Qwen3-8B: INT4, INT8

**32k (all three: baseline, INT4, INT8):**
- DeepSeek-R1-Distill-Qwen-7B
- DeepSeek-R1-Distill-Llama-8B
- Qwen3-8B
- Yarn-Qwen3-8B

### 6. LongProc Complete Suite

**Models:** All 6 models

**Contexts:** 0.5k, 2k, 8k

**Tasks:**
- html_to_tsv
- pseudo_to_code
- travel_planning

**Techniques:** Baseline, INT4, INT8, StreamingLLM (both cache sizes), PyramidKV (both configs), SnapKV (both configs)

### 7. Qwen3 Thinking Modes

**Models:**
- Qwen3-8B
- Yarn-Qwen3-8B

**Modes:** Both `enable_thinking=True` and `enable_thinking=False`

**Coverage:** ALL tasks, ALL contexts (16k, 32k, 0.5k, 2k, 8k), ALL techniques

## HELMET Tasks

Full list of HELMET tasks covered:
- niah (Needle in a Haystack)
- cite (Citation accuracy)
- recall_jsonkv (JSON key-value recall)
- rag_hotpotqa (RAG with HotpotQA)
- rag_nq (RAG with Natural Questions)
- rerank (Reranking)
- icl (In-context Learning - Banking77 & Clinic150)
- summ_multilex (Summarization - MultiLexSum)

## LongProc Tasks

- html_to_tsv (HTML parsing to TSV)
- pseudo_to_code (Pseudocode to code generation)
- travel_planning (Travel itinerary planning)

## Monitoring Jobs

```bash
# Check all your jobs
squeue -u $USER

# Check specific job IDs (from submission output)
squeue -j 12345,12346,12347

# Watch job queue in real-time
watch -n 10 squeue -u $USER

# Check job logs
ls joblog/
tail -f joblog/[job-name]-[job-id]_0.out
tail -f joblog/[job-name]-[job-id]_0.err

# Cancel all submitted jobs (use job IDs from submission output)
scancel 12345 12346 12347

# Or cancel by job name pattern
scancel -n baseline_helmet_*
```

## Results Location

Results are saved to:
```
/scratch/gpfs/DANQIC/jz4391/HELMET/output/{technique}/{context_length}/{model}/
```

Examples:
- `.../output/baseline/16k/Qwen2.5-7B-Instruct/`
- `.../output/streamingllm/32k/Llama-3.1-8B-Instruct/`
- `.../output/pyramidkv/2k/Yarn-Qwen3-8B/`

## Aggregating Results

After jobs complete:

```bash
# Aggregate all results into CSV
python scripts/aggregate_results.py

# Check updated results
cat results/helmet_results/helmet_performance.csv
```

## Troubleshooting

### Job fails immediately

Check error log:
```bash
tail joblog/[job-name]-[job-id]_0.err
```

Common issues:
- Missing environment variables → Check config exports
- Model not found → Check model paths in config
- GPU out of memory → May need to adjust batch size or use quantization

### Test job hangs

```bash
# Check if it's running
squeue -u $USER

# Check output
tail -f joblog/[job-name]-[job-id]_0.out

# If truly stuck, cancel and investigate
scancel [job-id]
```

### Too many jobs in queue

```bash
# Cancel all jobs
scancel -u $USER

# Resubmit with more aggressive filtering
./scripts/submit_missing_experiments.sh --only-16k
```

### Qwen3 thinking mode not working

Verify config has `ENABLE_THINKING` set:
```bash
grep ENABLE_THINKING scripts/configs/missing_experiments_sweep/baseline_Qwen3_8B_nothinking_niah_16k_config.sh
```

Should see:
```bash
ENABLE_THINKING="False"  # or "True"
export ENABLE_THINKING
```

## Implementation Details

### Infrastructure Changes Made

1. **run_job.sh** - Added `ENABLE_THINKING_PARAM` support
2. **submit_job.sh** - Added `ENABLE_THINKING` to SLURM exports + optional time override parameter
3. **generate_missing_experiments.sh** - New comprehensive config generator
4. **submit_missing_experiments.sh** - New flexible submission script with filtering

**submit_job.sh enhancement:**
```bash
# Normal usage
./scripts/submit_job.sh config.sh

# With time override (used in test mode)
./scripts/submit_job.sh config.sh 01:00:00
```

### Key Parameters

| Parameter | Purpose | Values |
|-----------|---------|--------|
| `USE_REASONING_CONFIG` | Selects config folder | `true` (configs_reasoning) / `false` (configs) |
| `ENABLE_THINKING` | Qwen3 thinking mode | `True` (verbose CoT) / `False` (direct answer) |
| `EXP_TYPE` | Technique type | `baseline`, `streamingllm`, `pyramidkv`, `snapkv`, etc. |
| `BENCHMARK` | Benchmark type | `helmet`, `longproc` |
| `QUANTIZE` | Quantization bits | `4`, `8`, `16` |

### Config File Structure

Example config file (`baseline_Qwen3_8B_nothinking_niah_16k_config.sh`):
```bash
declare -a BASE_CONFIGS=("niah")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("Qwen3-8B")
declare -a QUANTIZE=("16")
EXP_TYPE="baseline"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
ENABLE_THINKING="False"
SEED=42

# SLURM Configuration
JOB_TIME="06:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_baseline_Qwen3_8B_nothinking_niah_eval"

# Export variables
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export ENABLE_THINKING
export SEED
```

## Summary

This system provides a comprehensive solution for filling in missing experimental results while maintaining manageable SLURM job loads through flexible filtering. The key innovation is the ability to control Qwen3's thinking mode separately from the reasoning config folders, enabling complete exploration of both thinking and non-thinking modes across all tasks and techniques.
