# Autoregressive vs 1-Token Constrained Decoding Benchmark

Implements `opencode_benchmark_plan.md`: benchmarks four inference strategies
against an OpenAI-compatible vLLM endpoint over public labeled datasets and
renders tabulated results (latency, throughput, format-error rate, accuracy,
confidence calibration).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration (environment variables)

| Variable         | Meaning                                   | Example                                        |
|------------------|-------------------------------------------|------------------------------------------------|
| `OPENAI_BASE_URL`| Base URL of the OpenAI-compatible endpoint | `https://yourllm-url/v1`                      |
| `OPENAI_API_KEY` | Bearer token                               | `sk-.yourllm-token`                            |
| `BENCH_MODEL`    | Model id served by the endpoint            | `yourllm-model`                               |
| `HF_TOKEN`       | Hugging face token                         | `hg-xxxx`                                     |
| `HF_TOKEN`       | Hugging face token                         | `hg-xxxx`                                     |
| `HF_ENDPOINT`    | Hugging default endpoint url               | `https://huggingface.co)`                     |
| `SSL_CERT_FILE`  | CA Cert location                           | `/etc/ssl/certs/yourcert.crt`                 |


## Run

```bash
# Preview the exact report layout with mock data - no network, no key, no downloads
python3 benchmark.py --dry-run --limit 20

# Real run (small smoke)
OPENAI_BASE_URL=... OPENAI_API_KEY=... BENCH_MODEL=yourllm-model \
  python3 benchmark.py --limit 25 --concurrency 4

# Full run
OPENAI_BASE_URL=... OPENAI_API_KEY=... BENCH_MODEL=yourllm-model \
  python3 benchmark.py --limit 150 --concurrency 8
```

Useful flags: `--dataset {sst2,agnews,both}`, `--strategies autoreg,singlepass,constrained,selfconsistency`,
`--n-samples` (self-consistency votes), `--top-logprobs`, `--thinking` (keep Qwen
reasoning mode on - default is disabled), `--outdir`.

## Strategies benchmarked

1. `autoreg` - full autoregressive generation (`max_tokens=128`, temp 0), regex-parsed.
2. `singlepass` - 1 token, temp 0, `logprobs` + `top_logprobs`; candidate softmax
   reconstructed from the logprob window (API stand-in for the plan's HF forward pass).
3. `constrained` - 1-token-style decision via `structured_outputs.choice`
   (`max_tokens` = longest label), guaranteed format.
4. `selfconsistency` - constrained at temp 0.7 with `n` votes; confidence = agreement fraction.

## Datasets

- **SST-2** (`glue/sst2`) - binary YES/NO sentiment (the plan's `["YES","NO"]` case).
- **AG News** (`ag_news`) - 4-way topic routing (the plan's 4-class intent case).

Both are pulled on first use via Hugging Face `datasets` and cached locally.

## Output

- Console + `report.md`: four tabulated tables
  - Table 1: per-strategy summary (p50/p99 ms, req/s, fmt_err %, acc %)
  - Table 2: Comparison A - autoreg vs single-pass (accuracy, latency, agreement)
  - Table 3: Comparison B - confidence vs correctness (Pearson, Spearman, ECE)
  - Table 4: latency-vs-accuracy ladder with relative p50 vs autoreg
  - Table 5: glossary defining every strategy, metric, and column used above
- `results.json`: config (key redacted), per-strategy summaries, per-item records.

## Endpoint notes

- Backend is vLLM 0.30 serving `qwen3.5-397b-fp8`, a reasoning model. The script
  sends `chat_template_kwargs: {"enable_thinking": false}` by default; without it,
  `max_tokens=1` yields a thinking token instead of an answer.
- `structured_outputs.choice` is honored; legacy `guided_choice` is silently
  ignored by this server, so it is not sent.
- `logprobs`/`top_logprobs` work; labels outside the top-K window are floored
  in the softmax reconstruction rather than causing an error.
- Latency is end-to-end HTTPS (RTT + queue + generate). The plan's 10-50 ms
  single-pass figure is only reachable with a colocated GPU; over the network
  expect a ~150-250 ms floor.
