# R1 Distill Qwen 7B Performance Investigation

## Issue Summary
DeepSeek-R1-Distill-Qwen-7B is performing extremely poorly across all HELMET tasks, showing dramatic underperformance compared to other models even in baseline conditions and with token eviction techniques.

## Investigation Findings

### 1. Severe Hallucination and Response Irrelevance
The R1 Distill Qwen 7B model generates completely irrelevant responses to questions:

**Example 1:**
- **Question**: "Who is opening for Shania Twain in Ottawa?"
- **R1 Distill Qwen 7B**: "The record for the longest live musical career is held by Celine Dion..."
- **Qwen2.5-7B**: "Shania Twain's 'Now Tour' opened with Bastian Baker..."

**Example 2:**
- **Question**: "When does the new army uniform come out?"
- **R1 Distill Qwen 7B**: "The record for the longest field goal in the NFL is held by..."
- **Qwen2.5-7B**: "The most recent information indicates that the new Army uniform, based on the OCP..."

### 2. Performance Comparison
Quantitative performance metrics from helmet_performance.csv:

**CITE Task (str_em metric):**
- Qwen2.5-7B-Instruct (baseline): 36.58%
- **R1 Distill Qwen 7B (baseline): 13.98%**

**Citation Precision/Recall:**
- Qwen2.5-7B-Instruct: 44.53% rec / 38.56% prec
- **R1 Distill Qwen 7B: 0.0% rec / 0.0% prec**

**NIAH Task:**
- R1 Distill Qwen 7B (baseline): 36.0%
- R1 Distill Qwen 7B (pyramidkv configurations): Range from 5.0% to 21.0%

### 3. Missing Results Across Tasks
The CSV shows extensive missing data (empty cells) for R1 Distill Qwen 7B across:
- recall_jsonkv
- rag_hotpotqa
- rag_nq
- Multiple PyramidKV configurations

### 4. Configuration Issues Identified
Multiple job logs showed configuration errors:
- Invalid quantization parameters (using "int4" instead of "4")
- Jobs failing due to setup issues
- **Critical finding**: Chat template mismatch

### 5. Root Cause Analysis

**Primary Issue: Chat Template Handling**
- R1 Distill Qwen 7B config: `"use_chat_template": false`
- Successful Qwen2.5-7B config: `"use_chat_template": true`

**Secondary Issues:**
- Thinking token processing: `"enable_thinking": true` may not be properly handled
- Model architecture compatibility with evaluation pipeline
- Possible tokenizer issues with special tokens

## Model Configuration Analysis
From `/scratch/gpfs/DANQIC/models/DeepSeek-R1-Distill-Qwen-7B/config.json`:
```json
{
  "architectures": ["Qwen2ForCausalLM"],
  "model_type": "qwen2",
  "vocab_size": 152064,
  "max_position_embeddings": 131072,
  "sliding_window": 4096,
  "use_sliding_window": false
}
```

## Sample Output Quality Comparison

**F1 Scores for Same Questions:**
- R1 Distill Qwen 7B: 0.28, 0.05, 0.14 (very low)
- Qwen2.5-7B: 0.30, 0.19, 0.24 (reasonable)

## Recommendations

### Immediate Fix
1. **Enable chat template**: Set `use_chat_template: true` for R1 Distill Qwen 7B
2. **Re-run key experiments**: Focus on SnapKV and PyramidKV for critical tasks

### Additional Debugging Steps
1. Verify model integrity and loading
2. Test standalone inference outside HELMET framework
3. Debug thinking token processing and stripping
4. Check for R1-specific evaluation parameters

### Priority Re-runs
- Tasks: niah, cite, rag_hotpotqa, rerank
- Techniques: SnapKV, PyramidKV
- Model: DeepSeek-R1-Distill-Qwen-7B only

## Conclusion
The performance issues appear to be systematic integration problems rather than model capability issues. The chat template mismatch is likely the primary cause of the nonsensical responses and poor performance across all tasks.

---
*Investigation completed: 2025-01-31*
*Files examined: 254 experiment outputs, configuration files, job logs*