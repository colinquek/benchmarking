#!/usr/bin/env python3
"""Benchmark: full autoregressive generation vs 1-token constrained decoding.

Compares four inference strategies over public, labeled text-classification
datasets against an OpenAI-compatible vLLM endpoint, then renders tabulated
results: latency (p50/p99), throughput, formatting error rate, accuracy,
cross-strategy decision agreement, and confidence-vs-correctness correlation.

Configuration comes from environment variables: OPENAI_BASE_URL, OPENAI_API_KEY,
BENCH_MODEL. Use --dry-run to preview the report with deterministic mock
responses (no network, no key, no dataset download).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ALL_STRATEGIES = ("autoreg", "singlepass", "constrained", "selfconsistency")


@dataclass
class Config:
    base_url: str = ""
    api_key: str = ""
    model: str = ""                         # required: BENCH_MODEL or --model
    dataset: str = "both"                 # sst2 | agnews | both
    limit: int = 150                      # samples per dataset
    concurrency: int = 8
    top_logprobs: int = 20
    n_samples: int = 5                    # self-consistency votes
    max_tokens_autoreg: int = 128
    thinking: bool = False                # Qwen reasoning mode
    strategies: tuple[str, ...] = ALL_STRATEGIES
    outdir: str = "."
    warmup: int = 2
    timeout: float = 120.0
    dry_run: bool = False
    cache_bust: bool = True               # defeats server-side prompt caching

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            base_url=os.environ.get("OPENAI_BASE_URL", "").rstrip("/"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=os.environ.get("BENCH_MODEL", ""),
        )


# --------------------------------------------------------------------------- #
# Datasets (public, labeled)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Item:
    key: str
    prompt: str
    choices: tuple[str, ...]
    gold: str


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    choices: tuple[str, ...]
    max_answer_tokens: int
    loader: Callable[[int], list[Item]]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_sst2(limit: int) -> list[Item]:
    """SST-2 sentiment -> binary YES/NO decision, the spec's flagship case."""
    from datasets import load_dataset

    rows = load_dataset("nyu-mll/glue", "sst2", split="validation")
    items: list[Item] = []
    for row in rows:
        if len(items) >= limit:
            break
        prompt = (
            "Read the review and decide whether the sentiment is positive.\n"
            "Answer with exactly one word: YES or NO.\n\n"
            f"Review: {row['sentence']}\n\nAnswer:"
        )
        items.append(
            Item(key=f"sst2-{row['idx']}", prompt=prompt,
                 choices=("YES", "NO"), gold="YES" if row["label"] == 1 else "NO")
        )
    return items


def load_agnews(limit: int) -> list[Item]:
    """AG News -> 4-way intent routing, the spec's billing/technical/sales case."""
    from datasets import load_dataset

    labels = ("world", "sports", "business", "science")
    rows = load_dataset("fancyzhx/ag_news", split="test")
    items: list[Item] = []
    for i, row in enumerate(rows):
        if len(items) >= limit:
            break
        prompt = (
            "Route this news snippet to exactly one topic.\n"
            "Choose from: world, sports, business, science.\n\n"
            f"Snippet: {row['text']}\n\nTopic:"
        )
        items.append(
            Item(key=f"agnews-{i}", prompt=prompt, choices=labels, gold=labels[row["label"]])
        )
    return items


DATASETS: dict[str, DatasetSpec] = {
    "sst2": DatasetSpec("sst2", ("YES", "NO"), 2, load_sst2),
    "agnews": DatasetSpec("agnews", ("world", "sports", "business", "science"), 4, load_agnews),
}


# --------------------------------------------------------------------------- #
# Chat completion clients
# --------------------------------------------------------------------------- #

class ChatClient:
    """Async OpenAI-compatible client. Adds enable_thinking=false by default."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._http: Any = None

    async def __aenter__(self) -> "ChatClient":
        import httpx

        self._http = httpx.AsyncClient(
            base_url=self.cfg.base_url,
            timeout=self.cfg.timeout,
            headers={"Authorization": f"Bearer {self.cfg.api_key}"},
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._http is not None:
            await self._http.aclose()

    async def complete(self, prompt: str, params: dict) -> tuple[dict, float]:
        if self.cfg.cache_bust:
            # Some vLLM deployments cache completions keyed on prompt+model only,
            # so a repeat prompt returns the FIRST strategy's response (with its
            # parameters' side effects, e.g. missing logprobs). A unique request
            # id per call forces a cache miss and measures real generation.
            prompt = f"[req {uuid.uuid4().hex[:12]}] {prompt}"
        body = {"model": self.cfg.model,
                "messages": [{"role": "user", "content": prompt}], **params}
        if not self.cfg.thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        started = time.perf_counter()
        resp = await self._http.post("/chat/completions", json=body)
        latency_ms = (time.perf_counter() - started) * 1000
        resp.raise_for_status()
        return resp.json(), latency_ms


class MockClient(ChatClient):
    """Deterministic stand-in for --dry-run: renders the full report offline.

    Gold labels come from the dataset so accuracies are realistic; latency is a
    smooth function of max_tokens/n so latency ordering stays plausible.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._truth: dict[str, tuple[tuple[str, ...], str]] = {}

    def register(self, items: Iterable[Item]) -> None:
        for item in items:
            self._truth[_prompt_id(item.prompt)] = (item.choices, item.gold)

    async def __aenter__(self) -> "MockClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def complete(self, prompt: str, params: dict) -> tuple[dict, float]:
        choices, gold = self._truth.get(_prompt_id(prompt), (("YES", "NO"), "YES"))
        rng = random.Random(_prompt_id(prompt))
        constrained = "structured_outputs" in params
        autoregressive = params.get("max_tokens", 1) > 4
        n = params.get("n", 1)
        latency_ms = 140 + 9.0 * params.get("max_tokens", 1) + 30 * n + rng.uniform(0, 40)
        await asyncio.sleep(latency_ms / 1000.0)

        # Deterministic seeded error injection so the dry-run report is non-degenerate.
        pid = int(_prompt_id(prompt)[:8], 16)
        error_rate = 0.15 if autoregressive else (0.04 if constrained else 0.08)

        outs = []
        for k in range(n):
            flip = random.Random(f"{pid}-{k}").random() < error_rate
            label = rng.choice([c for c in choices if c != gold] or choices) if flip else gold
            prob = rng.uniform(0.05, 0.3) if flip else rng.uniform(0.75, 0.99)
            content = label
            if autoregressive and not flip and rng.random() < 0.25:
                content = f"**{label}**"   # inject format errors into autoreg
            outs.append({
                "message": {"content": content},
                "logprobs": {"content": [{
                    "token": content, "logprob": math.log(prob),
                    # Mass follows the emitted label, so single-pass shows errors
                    # when they are injected and its correlation table is defined.
                    "top_logprobs": _mock_top_logprobs(choices, label, math.log(prob), rng),
                }]},
            })
        return {"choices": outs}, latency_ms


def _mock_top_logprobs(choices, picked, picked_lp, rng) -> list[dict]:
    entries = [{"token": picked, "logprob": picked_lp}]
    for c in choices:
        if c != picked:
            entries.append({"token": c, "logprob": picked_lp - rng.uniform(1.5, 9.0)})
    return entries


def _prompt_id(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Response parsing / candidate softmax
# --------------------------------------------------------------------------- #

def logsumexp(values: list[float]) -> float:
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def softmax(logits: dict[str, float]) -> dict[str, float]:
    if not logits:
        return {}
    m = max(logits.values())
    exps = {k: math.exp(v - m) for k, v in logits.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def match_choice(token: str, choices: tuple[str, ...]) -> str | None:
    for c in choices:
        if normalize(token) == normalize(c):
            return c
    return None


def normalize_decision(text: str | None, choices: tuple[str, ...]) -> str | None:
    """Map raw output onto a choice, tolerating punctuation and unambiguous prefixes.

    Constrained decoding with max_tokens=1 can emit a prefix token such as "Y"
    for "YES"; accepting prefixes keeps that from inflating the error rate.
    """
    if text is None:
        return None
    cleaned = normalize(text).rstrip(".!").strip()
    exact = match_choice(cleaned, choices)
    if exact:
        return exact
    if cleaned:
        hits = [c for c in choices if normalize(c).startswith(cleaned)]
        if len(hits) == 1:
            return hits[0]
    return None


def candidate_probs(choices: tuple[str, ...], top_logprobs: list[dict]) -> dict[str, float]:
    """Fold top_logprobs mass onto candidate labels, then renormalize.

    Labels outside the top-K window get a floored logit so they never win and
    the call degrades gracefully instead of raising.
    """
    if not top_logprobs:
        return {}
    floor = min(e["logprob"] for e in top_logprobs) - 10.0
    logits = {c: floor for c in choices}
    for entry in top_logprobs:
        label = match_choice(entry["token"], choices)
        if label is not None:
            logits[label] = logsumexp([logits[label], entry["logprob"]])
    return softmax(logits)


def first_top_logprobs(resp: dict) -> list[dict]:
    choice = (resp.get("choices") or [{}])[0]
    content = (choice.get("logprobs") or {}).get("content") or []
    return (content[0].get("top_logprobs") or []) if content else []


def first_content(resp: dict, index: int = 0) -> str | None:
    choice = (resp.get("choices") or [{}])[index]
    return (choice.get("message") or {}).get("content")


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

@dataclass
class Record:
    dataset: str
    strategy: str
    key: str
    gold: str
    decision: str | None
    format_ok: bool
    confidence: float | None
    latency_ms: float

    @property
    def correct(self) -> bool | None:
        if self.decision is None:
            return None
        return normalize(self.decision) == normalize(self.gold)


@dataclass
class StrategyRun:
    name: str
    records: list[Record] = field(default_factory=list)
    wall_ms: float = 0.0
    errors: int = 0
    first_error: str = ""


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

def params_autoreg(cfg: Config) -> dict:
    return {"max_tokens": cfg.max_tokens_autoreg, "temperature": 0.0}


def params_singlepass(cfg: Config) -> dict:
    return {"max_tokens": 1, "temperature": 0.0,
            "logprobs": True, "top_logprobs": cfg.top_logprobs}


def params_constrained(cfg: Config, item: Item, spec: DatasetSpec) -> dict:
    return {"max_tokens": spec.max_answer_tokens, "temperature": 0.0,
            "logprobs": True, "top_logprobs": max(cfg.top_logprobs, len(item.choices) + 2),
            "structured_outputs": {"choice": list(item.choices)}}


def params_selfconsistency(cfg: Config, item: Item, spec: DatasetSpec) -> dict:
    params = params_constrained(cfg, item, spec)
    params.update({"temperature": 0.7, "n": cfg.n_samples})
    return params


async def run_simple(
    cfg: Config, client: ChatClient, spec: DatasetSpec, strategy: str,
    items: list[Item], make_params: Callable[[Item], dict],
    parse: Callable[[Item, dict, float], Record], sem: asyncio.Semaphore,
) -> StrategyRun:
    run = StrategyRun(strategy)

    async def one(item: Item) -> None:
        async with sem:
            try:
                resp, ms = await client.complete(item.prompt, make_params(item))
            except Exception as exc:
                run.errors += 1
                run.first_error = run.first_error or f"{type(exc).__name__}: {exc}"
                run.records.append(Record(spec.name, strategy, item.key, item.gold,
                                          None, False, None, 0.0))
                return
        run.records.append(parse(item, resp, ms))

    started = time.perf_counter()
    await asyncio.gather(*(one(item) for item in items))
    run.wall_ms = (time.perf_counter() - started) * 1000
    return run


async def run_autoreg(cfg, client, spec, items, sem) -> StrategyRun:
    def parse(item, resp, ms):
        raw = first_content(resp)
        decision = normalize_decision(raw, item.choices)
        # Format error = output is not exactly the bare label (markdown, prose, ...).
        format_ok = raw is not None and normalize(raw) == normalize(decision or "\x00")
        return Record(spec.name, "autoreg", item.key, item.gold, decision, format_ok, None, ms)

    return await run_simple(cfg, client, spec, "autoreg", items,
                            lambda item: params_autoreg(cfg), parse, sem)


async def run_singlepass(cfg, client, spec, items, sem) -> StrategyRun:
    def parse(item, resp, ms):
        probs = candidate_probs(item.choices, first_top_logprobs(resp))
        decision = max(probs, key=probs.get) if probs else None
        return Record(spec.name, "singlepass", item.key, item.gold, decision,
                      decision is not None, probs.get(decision), ms)

    return await run_simple(cfg, client, spec, "singlepass", items,
                            lambda item: params_singlepass(cfg), parse, sem)


async def run_constrained(cfg, client, spec, items, sem) -> StrategyRun:
    def parse(item, resp, ms):
        decision = normalize_decision(first_content(resp), item.choices)
        probs = candidate_probs(item.choices, first_top_logprobs(resp))
        return Record(spec.name, "constrained", item.key, item.gold, decision,
                      decision is not None, probs.get(decision), ms)

    return await run_simple(
        cfg, client, spec, "constrained", items,
        lambda item: params_constrained(cfg, item, spec), parse, sem)


async def run_selfconsistency(cfg, client, spec, items, sem) -> StrategyRun:
    run = StrategyRun("selfconsistency")

    async def one(item: Item) -> None:
        async with sem:
            try:
                resp, ms = await client.complete(
                    item.prompt, params_selfconsistency(cfg, item, spec))
            except Exception as exc:
                run.errors += 1
                run.first_error = run.first_error or f"{type(exc).__name__}: {exc}"
                run.records.append(Record(spec.name, "selfconsistency", item.key,
                                          item.gold, None, False, None, 0.0))
                return
        votes = [normalize_decision(first_content(resp, i), item.choices)
                 for i in range(len(resp.get("choices") or []))]
        votes = [v for v in votes if v]
        decision = max(set(votes), key=votes.count) if votes else None
        agreement = votes.count(decision) / len(votes) if votes else None
        run.records.append(Record(spec.name, "selfconsistency", item.key, item.gold,
                                  decision, decision is not None, agreement, ms))

    started = time.perf_counter()
    await asyncio.gather(*(one(item) for item in items))
    run.wall_ms = (time.perf_counter() - started) * 1000
    return run


STRATEGY_RUNNERS = {
    "autoreg": run_autoreg,
    "singlepass": run_singlepass,
    "constrained": run_constrained,
    "selfconsistency": run_selfconsistency,
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def mean(values: Iterable[float | None]) -> float:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else math.nan


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return math.nan
    xs2, ys2 = (list(t) for t in zip(*pairs))
    n = len(pairs)
    mx, my = sum(xs2) / n, sum(ys2) / n
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = sum((x - mx) ** 2 for x in xs2)
    vy = sum((y - my) ** 2 for y in ys2)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else math.nan


def rank_data(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return math.nan
    a, b = (list(t) for t in zip(*pairs))
    return pearson(rank_data(a), rank_data(b))


def expected_calibration_error(records: list[Record], bins: int = 10) -> float:
    pts = [(r.confidence, 1.0 if r.correct else 0.0)
           for r in records if r.confidence is not None and r.correct is not None]
    if not pts:
        return math.nan
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(c, y) for c, y in pts if lo <= c < hi or (hi == 1.0 and c == 1.0)]
        if bucket:
            ece += (len(bucket) / len(pts)) * abs(mean(c for c, _ in bucket)
                                                  - mean(y for _, y in bucket))
    return ece


# --------------------------------------------------------------------------- #
# Tabulated report
# --------------------------------------------------------------------------- #

def render_table(headers: list[str], rows: list[list[str]], aligns: str = "l") -> str:
    grid = [headers] + rows
    widths = [max(len(str(r[i])) for r in grid) for i in range(len(headers))]

    def render(row):
        cells = []
        for i, cell in enumerate(row):
            text = str(cell)
            width = widths[i]
            cells.append(text.rjust(width) if aligns[i] == "r" else text.ljust(width))
        return " | ".join(cells).rstrip()

    separator = "-+-".join("-" * w for w in widths)
    return "\n".join([render(headers), separator] + [render(r) for r in rows])


def fmt_ms(x: float) -> str:
    return "n/a" if x != x else f"{x:.1f}"


def fmt_pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}"


def fmt_num(x: float, nd: int = 3) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


@dataclass
class Summary:
    dataset: str
    strategy: str
    n: int
    p50: float
    p99: float
    throughput: float
    format_err: float
    accuracy: float
    errors: int
    first_error: str = ""
    records: list[Record] = field(default_factory=list)


def summarize(run: StrategyRun, spec: DatasetSpec) -> Summary:
    lat = [r.latency_ms for r in run.records if r.latency_ms > 0]
    graded = [1.0 if r.correct else 0.0 for r in run.records if r.correct is not None]
    return Summary(
        dataset=spec.name, strategy=run.name, n=len(run.records),
        p50=percentile(lat, 50), p99=percentile(lat, 99),
        throughput=(len(run.records) / (run.wall_ms / 1000.0)) if run.wall_ms else math.nan,
        format_err=mean([0.0 if r.format_ok else 1.0 for r in run.records]),
        accuracy=mean(graded) if graded else math.nan,
        errors=run.errors, records=run.records, first_error=run.first_error,
    )


def find(summaries: list[Summary], dataset: str, strategy: str) -> Summary | None:
    for s in summaries:
        if s.dataset == dataset and s.strategy == strategy:
            return s
    return None


def decision_agreement(a: Summary, b: Summary) -> float:
    """Fraction of prompts where both strategies pick the same label."""
    by_key = {r.key: r for r in b.records}
    same = total = 0
    for ra in a.records:
        rb = by_key.get(ra.key)
        if ra.decision is None or rb is None or rb.decision is None:
            continue
        total += 1
        same += normalize(ra.decision) == normalize(rb.decision)
    return same / total if total else math.nan


def build_report(cfg: Config, summaries: list[Summary]) -> str:
    datasets = sorted({s.dataset for s in summaries})
    lines: list[str] = []

    lines.append("# Benchmark Report: Autoregressive vs 1-Token Constrained Decoding")
    lines.append("")
    lines.append(f"model            : {cfg.model or '(unset)'}")
    lines.append(f"endpoint         : {cfg.base_url or '(dry-run, mock responses)'}")
    lines.append(f"thinking mode    : {'on' if cfg.thinking else 'off (chat_template_kwargs.enable_thinking=false)'}")
    lines.append(f"samples / dataset: {cfg.limit}    concurrency: {cfg.concurrency}    "
                 f"top_logprobs: {cfg.top_logprobs}    n_samples: {cfg.n_samples}")
    lines.append(f"strategies       : {', '.join(cfg.strategies)}")
    lines.append(f"cache busting    : {'on (unique [req ...] id per call)' if cfg.cache_bust else 'OFF'}")
    if cfg.dry_run:
        lines.append("")
        lines.append("**DRY-RUN: numbers below come from the built-in mock client, not a real model.**")
    lines.append("")
    lines.append("Latency is end-to-end over HTTPS (RTT + queue + generate), so it is not the")
    lines.append("colocated-GPU single-forward-pass figure quoted in the plan.")
    lines.append("")

    # ---- Table 1: headline per-strategy metrics
    lines.append("## Table 1 - Per-strategy summary")
    lines.append("")
    headers = ["dataset", "strategy", "n", "p50 ms", "p99 ms", "req/s", "fmt_err %", "acc %", "errs"]
    rows = [[s.dataset, s.strategy, s.n, fmt_ms(s.p50), fmt_ms(s.p99), fmt_num(s.throughput, 1),
             fmt_pct(s.format_err), fmt_pct(s.accuracy), s.errors] for s in summaries]
    lines.append(render_table(headers, rows, "llrrrrrrr"))
    lines.append("")

    # ---- Table 2: Comparison A - autoreg vs single-pass
    lines.append("## Table 2 - Comparison A: full autoregression vs single-pass decision")
    lines.append("")
    headers = ["dataset", "metric", "autoreg", "singlepass", "delta (auto - single)"]
    rows = []
    for ds in datasets:
        a, b = find(summaries, ds, "autoreg"), find(summaries, ds, "singlepass")
        if not a or not b:
            continue
        agree = decision_agreement(a, b)
        rows.append([ds, "accuracy %", fmt_pct(a.accuracy), fmt_pct(b.accuracy),
                     _delta_pct(a.accuracy, b.accuracy)])
        rows.append([ds, "p50 latency ms", fmt_ms(a.p50), fmt_ms(b.p50),
                     _delta(a.p50, b.p50)])
        rows.append([ds, "p99 latency ms", fmt_ms(a.p99), fmt_ms(b.p99),
                     _delta(a.p99, b.p99)])
        rows.append([ds, "req/s", fmt_num(a.throughput, 1), fmt_num(b.throughput, 1),
                     _delta(a.throughput, b.throughput)])
        rows.append([ds, "fmt_err %", fmt_pct(a.format_err), fmt_pct(b.format_err),
                     _delta_pct(a.format_err, b.format_err)])
        rows.append([ds, "decision agreement %", fmt_pct(agree), fmt_pct(agree), "-"])
    lines.append(render_table(headers, rows, "llrrr"))
    lines.append("")

    # ---- Table 3: Comparison B - confidence vs correctness
    lines.append("## Table 3 - Comparison B: confidence calibration vs correctness")
    lines.append("")
    headers = ["dataset", "confidence source", "n", "pearson r", "spearman rho",
               "ECE", "mean conf", "accuracy %"]
    rows = []
    for ds in datasets:
        for strat, label in (("singlepass", "single-pass softmax"),
                             ("constrained", "constrained logprob"),
                             ("selfconsistency", "self-consistency vote")):
            s = find(summaries, ds, strat)
            if not s:
                continue
            conf = [r.confidence for r in s.records]
            correct = [1.0 if r.correct else 0.0 for r in s.records]
            kept = [(c, y) for c, y in zip(conf, correct) if c is not None]
            if not kept:
                continue
            xs, ys = (list(t) for t in zip(*kept))
            rows.append([ds, label, len(xs), fmt_num(pearson(xs, ys)), fmt_num(spearman(xs, ys)),
                         fmt_num(expected_calibration_error(s.records)),
                         fmt_num(mean(xs)), fmt_pct(statistics.fmean(ys) if ys else math.nan)])
    lines.append(render_table(headers, rows, "llrrrrrr"))
    lines.append("")
    lines.append("Correlation is computed against binary correctness (1 = matches gold). "
                 "ECE = expected calibration error, lower is better. For self-consistency the "
                 "\"confidence\" is the vote-agreement fraction, not a token probability.")
    lines.append("")

    # ---- Table 4: latency/accuracy ladder across all strategies
    lines.append("## Table 4 - Latency vs accuracy ladder")
    lines.append("")
    headers = ["dataset", "strategy", "p50 ms", "p99 ms", "acc %", "fmt_err %", "rel. p50 vs autoreg"]
    rows = []
    for ds in datasets:
        base = find(summaries, ds, "autoreg")
        for strat in cfg.strategies:
            s = find(summaries, ds, strat)
            if not s:
                continue
            rel = (s.p50 / base.p50) if base and base.p50 == base.p50 and s.p50 == s.p50 else math.nan
            rows.append([ds, s.strategy, fmt_ms(s.p50), fmt_ms(s.p99), fmt_pct(s.accuracy),
                         fmt_pct(s.format_err), "1.00x" if rel != rel else f"{rel:.2f}x"])
    lines.append(render_table(headers, rows, "llrrrrr"))
    lines.append("")

    # ---- Table 5: glossary of every term used above
    lines.append("## Table 5 - Glossary of terms")
    lines.append("")
    lines.append("**Strategies**")
    lines.append("")
    strategy_terms = [
        ["autoreg", "Baseline: free-form generation up to max_tokens with temperature 0; "
                    "output parsed by exact match after whitespace/punctuation cleanup."],
        ["singlepass", "One generated token, temperature 0; decision and confidence come from "
                       "softmaxing the candidate labels' logprob mass inside the top_logprobs window."],
        ["constrained", "Server-side choice constraint (structured_outputs.choice): the model can "
                        "only emit one of the labels; confidence = renormalized label probability."],
        ["selfconsistency", "Constrained decoding repeated n times at temperature 0.7; the majority "
                            "vote is the decision and the vote share is the confidence."],
    ]
    lines.append(render_table(["term", "meaning"], strategy_terms, "ll"))
    lines.append("")
    lines.append("**Metrics and columns**")
    lines.append("")
    metric_terms = [
        ["dataset", "Test set: sst2 = binary sentiment (YES/NO); agnews = 4-way news-topic routing."],
        ["n", "Number of prompts evaluated for this dataset/strategy combination."],
        ["p50 ms", "Median end-to-end latency per request: client-side timer around the HTTPS "
                   "call (network RTT + server queue + generation). Not GPU-only time."],
        ["p99 ms", "99th-percentile latency; the worst-case tail most users of a decision "
                   "endpoint would feel. Linear-interpolation percentile."],
        ["req/s", "Throughput = prompts completed / wall-clock seconds for the batch at the "
                  "configured concurrency. Includes self-consistency calls as ONE request."],
        ["fmt_err %", "Percent of outputs that are NOT exactly one bare label (extra prose, "
                      "markdown, casing/punctuation noise). Constrained strategies are 0 by construction."],
        ["acc %", "Accuracy: predicted label equals the dataset gold label. Unparseable output "
                  "counts as incorrect."],
        ["errs", "Transport/HTTP failures for this combination (excluded from latency, counted "
                 "as format errors and incorrect)."],
        ["decision agreement %", "Share of prompts where the two compared strategies pick the same "
                                 "label, regardless of whether that label is correct."],
        ["pearson r", "Linear correlation in [-1,1] between confidence and binary correctness "
                      "(1=correct). ~1 means high confidence almost always means correct."],
        ["spearman rho", "Rank correlation between confidence and correctness; robust to a "
                         "non-linear confidence scale, unlike Pearson."],
        ["ECE", "Expected calibration error over 10 equal-width confidence bins: weighted mean of "
                "per-bin gap between avg confidence and accuracy. 0 = perfectly calibrated; "
                "lower is better."],
        ["mean conf", "Average confidence. For single-pass/constrained this is token probability; "
                      "for self-consistency it is the vote-agreement fraction (e.g. 4/5 = 0.80)."],
        ["delta (auto - single)", "Autoregressive value minus single-pass value. Positive = "
                                  "autoreg is slower/higher on that metric."],
        ["rel. p50 vs autoreg", "p50 of this strategy divided by autoreg p50. 0.15x = ~6.7x faster."],
        ["confidence source", "Which signal filled the correlation table: single-pass softmax "
                              "(unconstrained logits), constrained logprob (mask-renormalized), or "
                              "self-consistency vote share."],
    ]
    lines.append(render_table(["term", "meaning"], metric_terms, "ll"))
    return "\n".join(lines)


def _delta(a: float, b: float) -> str:
    if a != a or b != b:
        return "n/a"
    return f"{a - b:+.1f}"


def _delta_pct(a: float, b: float) -> str:
    if a != a or b != b:
        return "n/a"
    return f"{100 * (a - b):+.1f} pts"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

async def warmup(cfg: Config, client: ChatClient) -> None:
    params = {"max_tokens": 2, "temperature": 0.0}
    for _ in range(cfg.warmup):
        try:
            await client.complete("Reply with the single word OK.", params)
        except Exception:
            pass


def select_datasets(cfg: Config) -> list[DatasetSpec]:
    if cfg.dataset == "both":
        return [DATASETS["sst2"], DATASETS["agnews"]]
    return [DATASETS[cfg.dataset]]


async def main_async(cfg: Config) -> None:
    specs = select_datasets(cfg)
    client: ChatClient = MockClient(cfg) if cfg.dry_run else ChatClient(cfg)

    loaded: dict[str, list[Item]] = {}
    for spec in specs:
        if cfg.dry_run:
            loaded[spec.name] = _mock_items(spec, cfg.limit)
        else:
            print(f"[data] loading {spec.name} ...", flush=True)
            loaded[spec.name] = spec.loader(cfg.limit)
        if cfg.dry_run:
            client.register(loaded[spec.name])
        print(f"[data] {spec.name}: {len(loaded[spec.name])} items", flush=True)

    sem = asyncio.Semaphore(cfg.concurrency)
    summaries: list[Summary] = []
    async with client:
        if not cfg.dry_run:
            await warmup(cfg, client)
        for spec in specs:
            for strat in cfg.strategies:
                run = await STRATEGY_RUNNERS[strat](cfg, client, spec, loaded[spec.name], sem)
                summary = summarize(run, spec)
                summaries.append(summary)
                print(f"[done] {spec.name:7s} {strat:15s} n={summary.n:3d} "
                      f"p50={fmt_ms(summary.p50):>8s}ms acc={fmt_pct(summary.accuracy):>5s}%",
                      flush=True)
                if summary.errors:
                    print(f"[warn] {spec.name}/{strat}: {summary.errors}/{summary.n} requests "
                          f"failed; first error: {summary.first_error}", flush=True)

    report = build_report(cfg, summaries)
    print("\n" + report + "\n")

    os.makedirs(cfg.outdir, exist_ok=True)
    report_path = os.path.join(cfg.outdir, "report.md")
    with open(report_path, "w") as fh:
        fh.write(report + "\n")
    payload = {
        "config": _redact(cfg),
        "summaries": [{k: v for k, v in s.__dict__.items() if k != "records"}
                      for s in summaries],
        "records": {f"{s.dataset}/{s.strategy}": [
            {"key": r.key, "gold": r.gold, "decision": r.decision,
             "format_ok": r.format_ok, "confidence": r.confidence,
             "correct": r.correct, "latency_ms": r.latency_ms}
            for r in s.records] for s in summaries},
    }
    with open(os.path.join(cfg.outdir, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[out] {report_path}, results.json")


def _mock_items(spec: DatasetSpec, limit: int) -> list[Item]:
    """Small deterministic prompt set for --dry-run (no dataset download)."""
    templates = {
        "sst2": ["the film was genuinely wonderful", "a tedious and joyless mess",
                 "brilliant performances carry a thin script", "i would not watch this again"],
        "agnews": ["stock markets rally as inflation cools",
                   "the home team won the championship final",
                   "peace talks resume between the two nations",
                   "researchers announce a new battery chemistry"],
    }
    pool = templates.get(spec.name, ["sample input"])
    items = []
    for i in range(limit):
        # The case tag makes every prompt unique: the mock seeds its error
        # injection from the prompt, so shared prompts would flip in lockstep
        # and make confidence/correctness correlation degenerate.
        text = f"{pool[i % len(pool)]} (case {i})"
        gold = spec.choices[i % len(spec.choices)] if spec.name == "agnews" else ("YES" if i % 2 == 0 else "NO")
        prompt = f"Classify the following text.\n\nInput: {text}\n\nAnswer:"
        items.append(Item(key=f"{spec.name}-mock-{i}", prompt=prompt,
                          choices=spec.choices, gold=gold))
    return items


def _redact(cfg: Config) -> dict:
    data = dict(cfg.__dict__)
    data["api_key"] = "***redacted***"
    return data


def parse_args() -> Config:
    cfg = Config.from_env()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["sst2", "agnews", "both"], default=cfg.dataset)
    parser.add_argument("--limit", type=int, default=cfg.limit, help="samples per dataset")
    parser.add_argument("--concurrency", type=int, default=cfg.concurrency)
    parser.add_argument("--top-logprobs", type=int, default=cfg.top_logprobs)
    parser.add_argument("--n-samples", type=int, default=cfg.n_samples,
                        help="self-consistency vote count")
    parser.add_argument("--max-tokens-autoreg", type=int, default=cfg.max_tokens_autoreg)
    parser.add_argument("--model", default=cfg.model)
    parser.add_argument("--base-url", default=cfg.base_url)
    parser.add_argument("--thinking", action="store_true",
                        help="leave Qwen reasoning mode enabled (default: disabled)")
    parser.add_argument("--strategies", default=",".join(ALL_STRATEGIES),
                        help="comma list from: " + ",".join(ALL_STRATEGIES))
    parser.add_argument("--outdir", default=cfg.outdir)
    parser.add_argument("--warmup", type=int, default=cfg.warmup)
    parser.add_argument("--timeout", type=float, default=cfg.timeout)
    parser.add_argument("--dry-run", action="store_true",
                        help="mock responses: no network, no API key, no dataset download")
    parser.add_argument("--no-cache-bust", action="store_true",
                        help="send prompts verbatim (risky if the server caches completions)")
    args = parser.parse_args()

    cfg.dataset = args.dataset
    cfg.limit = args.limit
    cfg.concurrency = args.concurrency
    cfg.top_logprobs = args.top_logprobs
    cfg.n_samples = args.n_samples
    cfg.max_tokens_autoreg = args.max_tokens_autoreg
    cfg.model = args.model
    cfg.base_url = args.base_url.rstrip("/")
    cfg.thinking = args.thinking
    cfg.strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
    cfg.outdir = args.outdir
    cfg.warmup = args.warmup
    cfg.timeout = args.timeout
    cfg.dry_run = args.dry_run
    cfg.cache_bust = not args.no_cache_bust

    unknown = [s for s in cfg.strategies if s not in STRATEGY_RUNNERS]
    if unknown:
        raise SystemExit(f"Unknown strategies: {unknown}. Valid: {list(STRATEGY_RUNNERS)}")
    if not cfg.dry_run:
        if not cfg.base_url:
            raise SystemExit("Set OPENAI_BASE_URL (or --base-url), or pass --dry-run.")
        if not cfg.api_key:
            raise SystemExit("Set OPENAI_API_KEY, or pass --dry-run.")
        if not cfg.model:
            raise SystemExit("Set BENCH_MODEL (or --model), or pass --dry-run.")
    return cfg


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
