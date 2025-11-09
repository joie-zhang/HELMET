# DuoAttention R1-Distill-Llama-8B 32K Submission Guide

## Summary

This document describes the missing DuoAttention experiments and how to submit them.

### Current DuoAttention Coverage

Based on `results/helmet_results/helmet_performance.csv`:

✅ **Completed:**
- Llama-3.1-8B-Instruct @ 16K context
- Llama-3.1-8B-Instruct @ 32K context
- DeepSeek-R1-Distill-Llama-8B @ 16K context

❌ **Missing:**
- **DeepSeek-R1-Distill-Llama-8B @ 32K context** ← This is what we need!

## Missing Experiments Details

The following 32K tasks are missing for DeepSeek-R1-Distill-Llama-8B:

1. `rag_hotpotqa_32k.yaml`
2. `rag_nq_32k.yaml`
3. `rerank_32k.yaml`
4. `recall_jsonkv_32k.yaml`
5. `cite_32k.yaml`
6. `niah_32k.yaml`
7. `icl_32k.yaml`
8. `summ_multilex_32k.yaml`

**Total: 8 jobs**

## Submission Instructions

### Prerequisites

The submission script has been created at:
```bash
/scratch/gpfs/DANQIC/jz4391/HELMET/scripts/submit_duoattn_r1_32k.sh
```

### DuoAttention Configuration

The script uses the following settings (same as previous DuoAttention runs):

- **Attention Pattern**: Llama-3.1-8B-Instruct pattern (compatible with R1-Distill-Llama)
  - Path: `/scratch/gpfs/ab4197/p-longhead/duo-attention/attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10/full_attention_heads.tsv`
- **Sparsity**: 0.5 (50% retrieval head sparsity)
- **Chunk Prefilling**: 32768 tokens
- **Sink Size**: 128 tokens (default)
- **Sliding Window**: 1024 tokens (default)
- **Seed**: 42

### How to Submit

1. **Navigate to HELMET directory:**
   ```bash
   cd /scratch/gpfs/DANQIC/jz4391/HELMET
   ```

2. **Submit the jobs:**
   ```bash
   bash scripts/submit_duoattn_r1_32k.sh
   ```

3. **Monitor job status:**
   ```bash
   # Check SLURM queue
   squeue -u $USER

   # Check job logs (as they run)
   ls -ltr joie_joblog/helmet-duoattn-32k-DeepSeek-R1-Distill-Llama-8B-*
   ```

### Expected Runtime

- Each job should take approximately 2-4 hours (12-hour time limit allocated)
- Total wall time for all 8 jobs (running in parallel): ~2-4 hours

### Output Locations

Results will be saved to:
```
output/duoattn/32k/DeepSeek-R1-Distill-Llama-8B/_sp0.5_pf32768_tg/
```

### Verification

After jobs complete, verify results with:

```bash
# Check for completion markers
ls output/duoattn/32k/DeepSeek-R1-Distill-Llama-8B/_sp0.5_pf32768_tg/.*.completed

# Check results in CSV
grep "duoattn,32k,DeepSeek-R1-Distill-Llama-8B" results/helmet_results/helmet_performance.csv
```

## Technical Notes

### Why Use Llama-3.1 Attention Pattern?

DeepSeek-R1-Distill-Llama-8B is distilled from DeepSeek-R1 into Llama-3.1-8B architecture, so it shares the same attention pattern characteristics as Llama-3.1-8B-Instruct.

### Job Configuration

- **GPU**: 1x A100 80GB (`--constraint=gpu80`)
- **Memory**: 50GB
- **CPUs**: 8
- **Time Limit**: 12 hours
- **Conda Environment**: `duo` (includes DuoAttention implementation)

### Script Features

- ✅ Checks for already-completed jobs (skips if `.completed` file exists)
- ✅ Uses same parameters as successful Llama-3.1 32K runs
- ✅ Proper error handling and logging
- ✅ Email notifications on failure/timeout

## Troubleshooting

### If jobs fail:

1. **Check job logs:**
   ```bash
   tail -100 joie_joblog/helmet-duoattn-32k-DeepSeek-R1-Distill-Llama-8B-*err
   ```

2. **Common issues:**
   - Out of memory: R1-Distill models have thinking tokens that may increase memory usage
   - Timeout: 32K context can be slow; may need to increase time limit
   - Conda environment: Ensure `duo` environment is activated with DuoAttention installed

3. **Resubmit failed jobs:**
   - The script automatically skips completed jobs, so you can simply re-run:
     ```bash
     bash scripts/submit_duoattn_r1_32k.sh
     ```

## After Completion

Once all jobs finish:

1. **Collect results:**
   ```bash
   python scripts/collect_results_new.py
   ```

2. **Regenerate plots:**
   ```bash
   python scripts/plot_averages_comparison.py
   python scripts/plot_quadrant_comparison.py
   ```

3. **Verify DuoAttention appears in plots** for both:
   - All models view (R1-Distill-Llama should show DuoAttention data point)
   - Model-averaged view (averaged across all 4 models including R1-Distill-Llama)

---

**Created**: 2025-10-26
**Script**: `scripts/submit_duoattn_r1_32k.sh`
**Status**: Ready to submit
