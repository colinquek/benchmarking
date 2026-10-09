# Benchmark Specification: Autoregressive Generation vs. 1-Token Constrained Decoding

## Overview & Goal
This specification outlines a benchmark suite to compare two inference strategies on a self-hosted **Qwen** model cluster (e.g., Qwen2.5-14B / 72B):
1. **Full Autoregressive Generation:** Standard multi-step text output generation (`max_tokens=128+`).
2. **1-Token Constrained Decoding (Decision Engine Pattern):** A single forward pass ($T=1$) with logit masking (`max_tokens=1`, choice-constrained decoding) to evaluate softmax probabilities directly over an allowed candidate set (e.g., `["YES", "NO"]` or `["billing", "technical", "sales", "general"]`).

The objective is to measure latency gains, structural reliability, and confidence calibration between the two approaches.

---

## Architectural Comparison

| Metric / Feature | Full Autoregressive Generation | 1-Token Constrained Decoding |
| :--- | :--- | :--- |
| **Execution Loop** | Multi-step autoregressive generation ($T$ tokens). | Single forward pass ($T=1$ token). |
| **Logit Masking** | None or loose JSON grammar masking. | Hard bitmask forcing disallowed token logits to $-\infty$. |
| **Latency Profile** | Dependent on output string length ($\sim 1000\text{--}2000\text{ ms}$). | Fixed forward-pass time ($\sim 10\text{--}50\text{ ms}$). |
| **Format Determinism** | Probabilistic (prone to extra text/markdown without parsers). | **100% Structurally Guaranteed** (physically restricted to choice pool). |
| **Semantic Determinism** | Probabilistic (subject to GPU CUDA variances & temperature). | Probabilistic (subject to GPU CUDA variances & temperature). |

---

## Implementation Code Snippets

### 1. Approach A: Full Autoregressive Generation (Baseline)

```python
import time
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-14B-Instruct")

def run_autoregressive_generation(prompts: list[str]):
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128
    )

    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    results = []
    for out in outputs:
        results.append({
            "text": out.outputs[0].text.strip(),
            "latency_ms": elapsed_ms / len(prompts)
        })
    return results
```

---

### 2. Approach B: Single-Pass Logit Softmax & Confidence Score (PyTorch / Transformers)

Extracts candidate token probabilities on pass 1 without running an autoregressive generation loop.

```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

def evaluate_decision_single_pass(prompt: str, candidate_labels: list[str]):
    start_time = time.perf_counter()
    
    # 1. Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # 2. Single forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]

    # 3. Extract target candidate token IDs
    candidate_ids = [
        tokenizer.encode(label, add_special_tokens=False)[0] 
        for label in candidate_labels
    ]
    
    # 4. Filter logits & compute Softmax probabilities
    candidate_logits = torch.tensor([next_token_logits[cid] for cid in candidate_ids])
    probs = torch.softmax(candidate_logits, dim=-1)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    scores = {label: prob.item() for label, prob in zip(candidate_labels, probs)}
    top_decision = candidate_labels[torch.argmax(probs).item()]
    
    return {
        "decision": top_decision,
        "confidence": scores[top_decision],
        "probabilities": scores,
        "latency_ms": elapsed_ms
    }
```

---

### 3. Approach B: Constrained Decoding via vLLM Engine

Uses vLLM's guided decoding engine to restrict sampling to fixed candidate tokens at step 1.

```python
import time
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

def run_vllm_constrained_decision(llm: LLM, prompts: list[str], choices: list[str]):
    guided = GuidedDecodingParams(choice=choices)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        guided_decoding=guided
    )

    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    results = []
    for out in outputs:
        results.append({
            "text": out.outputs[0].text.strip(),
            "latency_ms": elapsed_ms / len(prompts)
        })
    return results
```

---

### 4. Verification Step: Self-Consistency ($N$-Sampling Agreement)

Performs $N$ decision evaluations at $T > 0$ to calculate decision agreement scores as a secondary confidence metric.

```python
from collections import Counter
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

def evaluate_self_consistency(llm: LLM, prompt: str, choices: list[str], n_samples: int = 5):
    guided = GuidedDecodingParams(choice=choices)
    sampling_params = SamplingParams(
        temperature=0.7, 
        max_tokens=1, 
        n=n_samples, 
        guided_decoding=guided
    )

    outputs = llm.generate([prompt], sampling_params)
    samples = [out.text.strip() for out in outputs[0].outputs]
    
    counts = Counter(samples)
    most_common, top_count = counts.most_common(1)[0]
    agreement_score = top_count / n_samples

    return {
        "consensus_decision": most_common,
        "agreement_score": agreement_score,
        "vote_distribution": dict(counts)
    }
```

---

## Instructions for OpenCode TUI
1. Implement a unified benchmark script (`benchmark.py`) loading a dataset of test prompts.
2. Run both **Full Autoregressive Generation** and **1-Token Constrained Decoding** across all test prompts.
3. Compute and render a summary report measuring:
   - **p50 and p99 Latency (ms)**
   - **Throughput (requests/sec)**
   - **Formatting Error Rate (%)**
   - **Confidence Score Correlation**