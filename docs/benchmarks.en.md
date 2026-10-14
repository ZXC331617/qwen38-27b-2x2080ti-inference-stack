# Tier-by-Tier Benchmarks (Qwen3.8-27B × 2×RTX 2080 Ti)

[简体中文](benchmarks.md) | **English**

Every measured number on this machine, with methodology and log provenance so each row can be
traced back.

---

## 1. Measurement protocol (read first)

| Item | Value |
|---|---|
| Concurrency | **single stream** (`--max-num-seqs 1`), no concurrency |
| Corpus | a real deepseek-harness session payload: 1 system message + 58 real tool results + 38 real tool definitions |
| Sampling | the harness's real values `temperature=0.7 / top_p=0.8 / top_k=20 / repetition_penalty=1.05` |
| Path | measuring host → local forwarder `127.0.0.1:8788` → model host `:8000` (same path as production) |
| Generation | `gen=256`, one request per tier |
| Config | production final: W4A16 + DFlash2 (probabilistic draft sampling) + fp8 KV + 225 W + Marlin atomic add |

### Metric definitions and reliability

| Column | Definition | Reliability |
|---|---|---|
| `decode t/s` | `completion_tokens / (total_s − ttft_s)` | **high** — derived from this request's own timestamps and token counts |
| `TTFT` | time to first token | **high** — same |
| `prefill t/s` | `prompt_tokens / ttft_s` | high, but affected by prefix cache (see below) |
| `drafts` / `acc_rate` / `mean_len` / `pos0..6` | deltas of vLLM's **global Prometheus counters** (`vllm:spec_decode_num_*_total`) across the request | ⚠️ **medium** — if *any* concurrent traffic exists in the measurement window (e.g. another client using the same model), those counts are mixed in, producing the tell-tale `drafts × mean_len ≫ 256` distortion |

> So `decode t/s`, `TTFT` and `prefill t/s` are treated as the primary metrics here; the
> acceptance columns are only quoted from batches where the window was verified to be free of
> concurrent traffic (flagged below).

### Two prefix-cache regimes

When the same template is replayed across tiers, the shared prefix is served from vLLM's prefix
cache (the `cached` column). `TTFT` then reflects **incremental prefill only**, not a cold start.
Both regimes are reported:

- **cold TTFT** — empty cache, first request;
- **warm-prefix TTFT** — prefix already cached (the real steady state of a multi-turn session).

---

## 2. Main table: tier-by-tier single-stream speed (re-run 2026-10-14, production config)

Log: `logs/bench-final-prod2.log`

| Tier | prompt_tok | Prefix hit | **decode t/s** | TTFT | prefill t/s | Note |
|---:|---:|---:|---:|---:|---:|---|
| 32K | 35,672 | 6,592 | **109.6** | 37.1 s | 783 | near-cold |
| 64K | 65,102 | 62,624 | **104.7** | 2.9 s | 847 | warm prefix |
| 96K | 95,953 | 93,936 | **127.7** | 3.1 s | 655 | only 126 tokens generated this run; small sample |
| 128K | 130,249 | 128,544 | **98.0** | 3.2 s | 526 | warm prefix |
| 160K | 160,703 | 158,208 | **67.7** | 4.9 s | 512 | warm prefix |
| 192K | — | — | — | — | — | rejected with **HTTP 400** (usable prompt ceiling ≈193K) |

> These single-stream measurements ran against a production endpoint that was **also serving
> live traffic** (`max-num-seqs 1`, requests queued serially), so a few rows may read low due to
> interleaving; ±10% between batches of the same tier is normal.

### Cold TTFT (no prefix cache, earlier same-config batch)

Logs: `logs/bench-prob.log` / `logs/bench-225w.log`

| Tier | Cold TTFT | Cold prefill t/s |
|---:|---:|---:|
| 32K | 24.1 s | 1,480 |
| 96K | 84.3 s | 1,139 |
| 160K | 175.1 s | 918 |

On par with the fork's official 2×2080 Ti reference (32K prefill ≈1265 t/s) — no obvious waste
on the prefill side.

---

## 3. Single-variable A/B comparisons (same tiers, same corpus)

### 3.1 Draft sampling `greedy` → `probabilistic` (biggest win)

Logs: `logs/bench-225w.log` (before) / `logs/bench-prob.log`, `logs/bench-prob-t2.log` (after)

| Tier | decode before | decode after | Change | Accept before | Accept after |
|---:|---:|---:|---:|---:|---:|
| 32K | 113.5 | **122.6** | +8% | 0.467 | 0.513 |
| 96K | 89.8 | **136.2** | **+52%** | 0.395 | 0.523 |
| 160K | 63.8 | **77.2** | **+21%** | 0.282 | 0.377 |

### 3.2 `VLLM_MARLIN_USE_ATOMIC_ADD=1`

Logs: `logs/bench-ea-clean.log` (off) / `logs/bench-base-clean.log`, `logs/bench-marlineadd.log` (on)

Comparing decode directly is noisy, so use the accept-rate-independent **step latency**:
`ms/step = (total_s − ttft_s) × 1000 / drafts`.

| Tier | Off | On | Step-latency change | decode change at equal accept |
|---|---:|---:|---:|---:|
| 32K | 37.8 ms/step | **36.8 ms/step** | −2.6% | +2.7% |
| 160K | 46.4 ms/step | **45.0 ms/step** | −3.0% | +3.1% |

### 3.3 Power limit 225 W → 260 W

Logs: `logs/bench-225w.log` / `logs/bench-260w.log`

| Tier | 225 W | 260 W | Change |
|---|---:|---:|---:|
| 32K | 37.5 ms/step | 37.3 ms/step | −0.5% (noise) |
| 160K | 46.5 ms/step | 45.2 ms/step | −2.8% |

Cost: load temps 69/57 °C → 80/70 °C, fans 80%/46%. Production finally locks **225 W**.

### 3.4 Draft model W4A16 → official BF16 lossless

Logs: `logs/bench-base-clean.log` (W4A16 draft) / `logs/bench-draftbf16.log` (BF16 draft)

| Tier | Metric | W4A16 draft | BF16 draft | Change |
|---|---|---:|---:|---:|
| 32K | accept | 4.830 | 4.849 | +0.4% |
| 32K | step latency | 37.8 ms | 39.5 ms | +4.5% |
| 32K | decode | **127.4** | 121.7 | **−4.5%** |
| 160K | accept | 3.493 | 3.556 | +1.8% |
| 160K | step latency | 46.4 ms | 48.2 ms | +3.9% |
| 160K | decode | **75.3** | 73.5 | **−2.4%** |

Accept rate barely moves while the heavier draft costs 4% per step → net loss. **Rolled back.**

### 3.5 Early-baseline full sweep (pre-change reference)

Log: `logs/bench-harness-real.log` (greedy draft + 262K context / non-fp8 KV, the early config)

| Tier | prompt_tok | decode t/s | TTFT | prefill t/s | Accept |
|---:|---:|---:|---:|---:|---:|
| 1K | 8,247 | 84.2 | 1.2 s | 1,400 | 0.291 |
| 8K | 8,247 | 100.8 | 1.1 s | 1,459 | 0.372 |
| 32K | 35,672 | 107.4 | 21.2 s | 1,371 | 0.435 |
| 64K | 65,102 | 103.7 | 48.4 s | 1,209 | 0.445 |
| 96K | 95,953 | 79.8 | 37.6 s | 886 | 0.337 |
| 128K | 130,249 | 81.1 | 62.4 s | 582 | 0.376 |
| 160K | 160,703 | 56.2 | 50.9 s | 632 | 0.236 |
| 192K | 197,218 | 70.3 | 70.7 s | 552 | 0.349 |
| 224K | 229,376 | — | — | — | HTTP 400 |

> In that batch the counters were self-consistent (`drafts × mean_len ≈ 256`), so the accept
> column is trustworthy and usable as the "before" baseline.

---

## 4. Reference comparison

Official DFlash2 on **H200 + official Qwen3.8-27B** (`temperature=1.0 / top_p=0.95 / top_k=20`):
accept length 4.10 (MT-Bench) – 5.46 (GSM8K), single-concurrency throughput **184–236 t/s**.

On this rig, 32K accept length is 4.83 with a **88.7%** position-0 hit rate; the official published
position-0 hit rate is **85.4%**.
→ At short context the draft/target pairing is normal; the gap comes mainly from **hardware
generation** and **long context**.

---

## 5. Log index

| File | Content |
|---|---|
| `bench-final-prod2.log` | main table (re-run, production final config) |
| `bench-final-prod.log` | previous re-run (same config, includes the 8K tier) |
| `bench-prob.log` / `bench-prob-t2.log` | probabilistic draft (32K/160K, 96K) |
| `bench-225w.log` / `bench-260w.log` | power-limit comparison |
| `bench-greedy.log` | fully-greedy comparison (warm prefix) |
| `bench-marlineadd.log` / `bench-ea-clean.log` / `bench-base-clean.log` | atomic-add comparison |
| `bench-draftbf16.log` | BF16 draft comparison |
| `bench-harness-real.log` | early-baseline full sweep (1K–192K) |

Scripts: `bench-harness-real.py` (`BENCH_TEMP` controls temperature), `run-bench.ps1` (launcher).

> Stop `wb-prefix-warm.timer` before reproducing, otherwise the speculative counters are polluted
> by the replayed requests.
