# Optimization Log & Conclusions (Qwen3.8-27B × 2×RTX 2080 Ti)

[简体中文](optimization.md) | **English**

This document records **every** change made on this machine in pursuit of **single-stream
maximum speed**, together with measured numbers and failed attempts.
The only goal is **fastest single stream**; concurrency is not needed at all
(`--max-num-seqs 1`).

Measurement protocol (identical across all runs):

- Corpus = a real deepseek-harness session payload (1 system message + 58 real tool results
  + 38 real tool definitions).
- Sampling = the harness's real values `temperature=0.7 / top_p=0.8 / top_k=20 /
  repetition_penalty=1.05`.
- Path = local forwarder → model host, exactly the production path.
- `gen=256`, one request per tier.

> Per-tier numbers and log files: see [benchmarks.en.md](benchmarks.en.md).

---

## 1. Conclusions first

Ordered by payoff:

| # | Change | Gain | Cost | Kept? |
|---|---|---|---|---|
| 1 | Draft sampling `greedy` → **`probabilistic`** | long-context decode **+21% … +52%** | none | ✅ kept |
| 2 | `VLLM_MARLIN_USE_ATOMIC_ADD=1` | step latency **−2.6% … −3.0%** | output no longer bit-reproducible per seed | ✅ kept |
| 3 | `--max-num-batched-tokens` 16384 → 24576 | expected prefill gain | **vLLM refuses to start** | ❌ rolled back |
| 4 | Draft model → official BF16 lossless | accept rate +0.4% … +1.8% (noise) | 4% slower per step, KV pool −18% | ❌ rolled back |
| 5 | Power limit 225 W → 260 W | −0.5% per step (noise) | temps 69 → 79 °C | ❌ production locked at 225 W |

In one sentence: **on this machine only two things actually matter — the draft sampling
method and the Marlin atomic-add flag. Everything else is noise or a net loss.**

---

## 2. Hardware constraint: SM75 dictates every trade-off

The RTX 2080 Ti is Turing, compute capability **7.5 (SM75)**. Three hard constraints
run through this whole document:

1. **No native bf16 tensor cores** → fp16 compute only (`--dtype half`).
2. **No native FP8** → FP8 weights cannot be inferred directly. Loading NVFP4 / FP8 weights
   fails immediately with `c10::Half != c10::Float8_e4m3fn`. So weights must be **int4
   W4A16** (compressed-tensors). The fp8 KV cache is a *storage* compression emulated by
   vLLM, not the FP8 tensor core feature.
3. **The fused GDN CUDA kernel requires compute capability ≥ 8.0** → automatic fallback to
   Triton:
   ```
   Falling back to the Triton GDN decode path: the fused CUDA kernel requires ... compute capability 8.0+
   ```
   In other words the `VLLM_GDN_DECODE_KERNEL` speed-up path is **already exhausted** on SM75;
   there is no flag left to flip.

The two cards are bridged with **NVLink** (`nvidia-smi topo -m` reports GPU0↔GPU1 = NV2) and
run with `--tensor-parallel-size 2`.

---

## 3. Model architecture: why latency grows slowly but accept rate still drops

The main model `Coder390-HUHUI-INC-A-W4A16` has a **64-layer hybrid attention** stack:

- **48 GDN linear-attention layers** + **16 full-attention layers**
  (`full_attention_interval=4`).
- Only 16 layers read the KV cache → **per-step latency grows very slowly with context**.
- KV cache is fp8: the 6.71 GiB pool for the 16 full-attention layers holds **320,654 tokens**
  (1.42× headroom for a 225K single stream).

This explains a counter-intuitive result: **long-context slowdown is driven by speculative
accept-rate collapse, not by step latency.**

| Tier | Step latency | Tokens per step | decode |
|---:|---:|---:|---:|
| 32K | 37.5 ms | 4.27 | 113.5 t/s |
| 160K | 46.5 ms | 2.98 | 63.8 t/s |

Step latency grew only 24%, but tokens per step fell 30%. So the **target of optimization is
accept rate, not raw compute**.

---

## 4. Optimization 1: draft sampling `greedy` → `probabilistic` (biggest win)

### Problem

The DFlash2 draft proposes tokens with `greedy` by default (argmax only), while the target
side samples at `temperature=0.7`. Temperature flattens the target distribution, so greedy
proposals miss far more often — and it gets worse as context grows.

### Change

One extra field in `--speculative-config`:

```json
{"method":"dflash","num_speculative_tokens":7,
 "model":"/root/models/Qwen3.8-27B-DFlash2-W4A16",
 "attention_backend":"TRITON_ATTN","kv_cache_dtype":"float16",
 "draft_sample_method":"probabilistic"}
```

### Measurements (same tiers, same corpus, `gen=256`)

| Tier | decode before | decode after | Change | Accept before | Accept after |
|---:|---:|---:|---:|---:|---:|
| 32K | 113.47 | **122.58** | +8% | 0.467 | 0.513 |
| 96K | 89.82 | **136.18** | **+52%** | 0.395 | 0.523 |
| 160K | 63.80 | **77.22** | **+21%** | 0.282 | 0.377 |

The gain is largest at long context, because greedy proposals degrade most there.

### Corroboration: the reverse experiment

Making the *target* greedy as well (`BENCH_TEMP=0`) raises 160K decode from 64.2 → 98.7 t/s
(+54%), with accept rate 0.28 → 0.48. **Same draft, only sampling consistency changed, and
more than half the loss comes back** — confirming that "sampling mismatch" is the root cause.

> Production cannot change target sampling (users want `temperature=0.7` diversity), so the
> correct fix is to make the **draft** probabilistic too.

---

## 5. Optimization 2: `VLLM_MARLIN_USE_ATOMIC_ADD=1` (~+3%)

### Origin

vLLM's startup log advertises it:

```
Marlin kernel can achieve better performance for small size_n with experimental use_atomic_add feature.
```

**Single-stream decode is exactly the small-`size_n` case**, so this flag is a natural fit.

### Change

One environment line in the unit, then restart — **no VRAM cost, no context change**:

```
Environment=VLLM_MARLIN_USE_ATOMIC_ADD=1
```

Confirmation: the message above disappears from the startup log once enabled
(`marlin_utils.py` only prints it when the flag is off).

### Measurement method (important)

Comparing decode t/s directly is noisy (accept rate for a tier swings between 0.28 and 0.55).
So use an accept-rate-independent metric:

```
step latency ms/step = (total_s − ttft_s) × 1000 / drafts
```

Also, `wb-prefix-warm.timer` on the host periodically replays prefix requests and corrupts the
speculative counters (visible as "accepted tokens > generated tokens"). It must be stopped
during measurement; only self-consistent counters (`drafts × mean_len ≈ 256`) are counted.

### Measurements (225 W / probabilistic / fixed seed=1234)

| Tier | Off | On | Step latency change | decode change at equal accept |
|---|---:|---:|---:|---:|
| 32K | 37.8 ms/step | **36.8 ms/step** | −2.6% | +2.7% |
| 160K | 46.4 ms/step | **45.0 ms/step** | −3.0% | +3.1% |

Same direction and same magnitude at both tiers → judged a real gain (~+3%).

### Side effect

Atomic add makes float accumulation order non-deterministic, so output is no longer
bit-reproducible under the same seed (measured: same-seed accept length moved from 4.83 to
4.197). Irrelevant for `temperature=0.7` chat workloads.

Rollback: `bash /tmp/revert-marlin-eadd.sh`
Backup: `/etc/systemd/system/vllm-qwen38.service.bak-marlinadd-20261014-185244`

---

## 6. Failure A: `--max-num-batched-tokens 16384 → 24576`

**Result: vLLM refuses to start. Rolled back.**

Startup error:

```
ValueError: To serve at least one request with the model's max seq len (225280),
4.79 GiB KV cache is needed, which is larger than the available KV cache memory (4.66 GiB).
Based on the available memory, the estimated maximum model length is 215888.
```

Measured profile change (per card):

| Item | 16384 (current) | 24576 (trial) | Delta |
|---|---:|---:|---:|
| Weights + non-torch | 11.17 GiB | 11.17 GiB | 0 |
| Peak activation | 3.08 GiB | ≈ 5.0 GiB | **+1.93 GiB** |
| KV pool | 6.71 GiB | 4.78 GiB | **−1.93 GiB** |
| KV token capacity | 320,654 | ≈ 234,000 | −27% |
| Can it serve 225,280 single stream? | yes (1.42× headroom) | **no** | — |

Additional costs:

- Peak activation and KV pool are **zero-sum** (`VLLM_KV_BUDGET_FROM_FREE_MEMORY=1`; KV is
  allocated from free VRAM). Each extra chunk token costs ≈ 0.24 MiB of KV.
- Changing the chunk changes the `torch.compile` dynamic-shape range (1,16384) → (1,24576),
  triggering a full recompile on first start (measured 41 s, cached afterwards).
- Back-solving, the chunk ceiling that still fits 225,280 is ≈ **24,100** — 24576 overshoots by
  ≈0.13 GiB, i.e. zero headroom; any later change would blow it up again.

**If 24576 is really wanted**, `--max-model-len` must drop from 225,280 to ≈212,992
(forwarder guard becomes 212,992 − 24,576 = 188,416, still above the 180,224 compaction
threshold) — i.e. trading ≈12K of context for prefill speed. Not adopted here.

Rollback: `bash /tmp/revert-batchtokens.sh` (executed)
Backup: `/etc/systemd/system/vllm-qwen38.service.bak-bt24576-20261014-181412`

---

## 7. Failure B: swapping in the official BF16 lossless draft (accept rate barely moves)

### First, proof that all public DFlash2 drafts are the same weights

Proved with a sha256 chain (not speculation):

| Source | File | sha256 |
|---|---|---|
| `incoai/Qwen3.8-27B-DFlash2` (official) | model.safetensors 3.849 GB | `67fc76d6…4bab65c` |
| `nerkyor/Qwen3.8-27B-Coder390-…-MTP-DFlash2` → `BF16/DFlash2-FP8` | 2.407 GB | `a7b48bc2…cbfb93`, manifest records `source.bf16_sha256 = 67fc76d6…` |
| `nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-…-DFlash2` → `BF16/DFlash2-FP8` | 2.407 GB | `1f3636a3…fa25d1`, same `source.bf16_sha256 = 67fc76d6…` |

So the so-called "Coder390-specific draft" and "Uncensored-specific draft" are just
**re-quantized copies of the official draft**, with identical underlying weights.
**Swapping drafts cannot buy accuracy.**

### Then, the A/B: BF16 draft vs current W4A16 draft

The official bf16 draft (3.849 GB, sha256 `67fc76d6…` verified) was downloaded and switched in:

| Tier | Metric | W4A16 draft | BF16 draft | Change |
|---|---|---:|---:|---:|
| 32K | accept (mean_len) | 4.830 | 4.849 | **+0.4%** |
| 32K | step latency | 37.8 ms | 39.5 ms | +4.5% |
| 32K | decode | 127.44 t/s | 121.73 t/s | **−4.5%** |
| 160K | accept (mean_len) | 3.493 | 3.556 | **+1.8%** |
| 160K | step latency | 46.4 ms | 48.2 ms | +3.9% |
| 160K | decode | 75.34 t/s | 73.52 t/s | **−2.4%** |

**Accept rate barely moved (noise level), but the heavier draft costs 4% per step — a net
loss.**

VRAM cost: weights + non-torch 11.17 → 12.40 GiB (+1.23 GiB/card), KV pool 6.71 → 5.49 GiB
(320,654 → 252,313 tokens), forcing `max-model-len` down from 225,280 to 196,608.

→ **Rolled back**; draft is again `Qwen3.8-27B-DFlash2-W4A16`, `max-model-len` back to 225,280.

### The real bottleneck: the draft is not from the same source as the target

- The draft was distilled from base **`Qwen/Qwen3.8-27B`**;
- our target is **Coder390-HUHUI-INC-A** (abliterated + heavily fine-tuned);
- the two distributions are systematically offset — that is the root cause of the 0.37
  long-context accept rate.

Reference: the fork's official pairing ("official target + official draft", same source)
reaches 209–220 t/s decode at 32K. Back-solving from our 37 ms step latency, they must be
producing ≈7.7 tokens per step (ceiling 8) — i.e. near-perfect acceptance.
**The gap comes from source pairing, not from quantization precision.**

> The only path that can genuinely raise the accept rate: **distill a DFlash2 draft from the
> current target model itself** (local training, not downloadable).

---

## 8. Failure C: power limit 225 W → 260 W

| Tier | 225 W | 260 W | Change |
|---|---:|---:|---:|
| 32K | 37.5 ms/step | 37.3 ms/step | −0.5% |
| 160K | 46.5 ms/step | 45.2 ms/step | −2.8% |

**Essentially within noise.** Cost: load temps go from 69/57 °C to 80/70 °C, fans 80%/46%.
Production finally **locks 225 W** (thermals not sustainable, and the gain is not real).

---

## 9. Exhausted / not worth retrying

| Item | Status |
|---|---|
| GDN decode on fused CUDA kernel | ❌ unsupported on SM75, falls back to Triton |
| FP8 weights inference | ❌ no FP8 tensor cores on SM75, load fails |
| bf16 weights | ❌ no native bf16 on SM75, fp16 only |
| Swapping drafts for accuracy | ❌ all public drafts share one source |
| Larger prefill chunk | ❌ blows the KV pool, won't start |
| Higher power limit | ❌ gain within noise |
| fp8 KV cache | ✅ in use (320,654 tokens) |
| Prefix caching | ✅ enabled |
| Marlin atomic add | ✅ in use |
| probabilistic draft sampling | ✅ in use (biggest win) |

---

## 10. Reproduction

```bash
# 1) Stop the timer that corrupts speculative counters / steals bandwidth
echo ubuntu | sudo -S systemctl stop wb-prefix-warm.timer

# 2) Run the tier sweep (on the measuring machine; BENCH_TEMP = production temperature 0.7)
BENCH_TEMP=0.7 pwsh -NoProfile -ExecutionPolicy Bypass \
  -File "D:\1\靶场\全层消融\run-bench.ps1" \
  -Tag final-prod -TiersCsv "8192,32768,65536,98304,131072,163840,196608"

# 3) Restore the timer afterwards
echo ubuntu | sudo -S systemctl start wb-prefix-warm.timer
```

Rollback scripts and backup files are listed in the sections above. The production unit
source is at [`../deploy/systemd/vllm-qwen38.service`](../deploy/systemd/vllm-qwen38.service).
