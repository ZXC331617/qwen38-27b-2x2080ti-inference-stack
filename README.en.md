# Qwen3.8-27B on 2x RTX 2080 Ti - Inference Stack

[简体中文](README.md) | **English**

Running an abliterated, 4-bit-quantized **Qwen3.8-27B** on two Turing cards
(RTX 2080 Ti / SM75 - no native bf16, no native FP8) with NVLink + TP=2, at
**225K context and 77-136 tok/s single-stream**. This repo is the complete
deployment plus the measured numbers.

> Contents: ready-to-use systemd units, monitoring stack config, full per-tier
> benchmarks, and a list of optimisation findings including the dead ends.

---

## Docs index

| Document | Contents |
|---|---|
| [README.md](README.md) · [中文](README.md) | overview: hardware, model, speeds, findings, production config |
| [docs/benchmarks.md](docs/benchmarks.md) · [EN](docs/benchmarks.en.md) | per-tier measurements + A/B comparisons (with log provenance) |
| [docs/optimization.md](docs/optimization.md) · [EN](docs/optimization.en.md) | optimisation log, root causes and **failed attempts** |
| [docs/deployment.md](docs/deployment.md) · [EN](docs/deployment.en.md) | nine-step from-scratch deployment + troubleshooting |
| [deploy/one-click-deploy.sh](deploy/one-click-deploy.sh) | **one-click installer** (install / uninstall / verify / status) |
| [deploy/README.md](deploy/README.md) | file-by-file description of the deploy bundle |
| [deploy/systemd/MANIFEST.md](deploy/systemd/MANIFEST.md) | systemd unit manifest (with sha256) |

---

## 1. Hardware

| Item | Spec |
|---|---|
| GPU | 2 x NVIDIA GeForce RTX 2080 Ti, 22,528 MiB each, compute capability **7.5 (SM75)** |
| GPU interconnect | **NVLink NV2** (`nvidia-smi topo -m` reports GPU0<->GPU1 = NV2) |
| Driver / CUDA | 595.99.02 / CUDA 13.0 (V13.0.88) |
| Power limits | 260 W default, 290 W max (production runs locked at 225 W - see below) |
| CPU | 36 logical cores |
| RAM | 31 GiB |
| System disk | 468 GB NVMe (~194 GB free in production) |
| PyTorch | 2.13.0+cu130 |
| Engine | **vLLM 2080 Ti Definitive Edition** (fork, `FORK_RELEASE=0.2.2-post2`, upstream github.com/weicj) |
| One-click deploy | `sudo ./deploy/one-click-deploy.sh --start` |

**Three hard constraints of SM75** (they drive every trade-off below):

1. No native bf16 -> fp16 compute only.
2. No native FP8 -> FP8 weights cannot be run directly (verified: loading fails with
   `c10::Half != c10::Float8_e4m3fn`).
3. The fused CUDA kernel for GDN linear attention requires compute capability >= 8.0,
   so it falls back to the Triton path automatically.

---

## 2. Models

**Target**: `Coder390-HUHUI-INC-A-W4A16`

- Base: an **abliterated** build of Qwen3.8-27B-Coder390 (EfficientThink family)
- Quantization: W4A16, `compressed-tensors` int4 group-128 symmetric, ~18.6 GB
- Architecture: **64 hybrid layers** - 48 GDN linear-attention + 16 full-attention
  (`full_attention_interval=4`)
  - This matters: only 16 layers read the KV cache, so **per-step latency grows very
    slowly with context length**
- Ships a vision tower (`Qwen3_5ForConditionalGeneration`) and an MTP head
  `model_mtp.safetensors` (849 MB)

**Speculative draft**: `Qwen3.8-27B-DFlash2-W4A16`

- DFlash2 = block-diffusion drafter (upstream `incoai/Qwen3.8-27B-DFlash2`, re-quantized
  to W4A16 in this project)
- 5 layers, hidden 5120, **sliding window 2048**, block_size 8, candidate selector
  rank 256 / top_k 16
- Reads hidden states from target layers 5/19/33/47/61

> Verified: **every public DFlash2 draft is the same set of weights.** The
> `DFlash2-FP8` files inside `incoai/Qwen3.8-27B-DFlash2` and the various
> `nerkyor/*-DFlash2` repos all record `source.bf16_sha256 = 67fc76d6...4bab65c` in their
> manifests - they are merely re-quantizations. **Swapping drafts buys no accuracy.**
> See [docs/optimization.en.md](docs/optimization.en.md).

---

## 3. Measured results

Benchmark protocol: **single stream** (`--max-num-seqs 1`), corpus taken from real
deepseek-harness session payloads (1 system message + 58 real tool results + 38 real tool
definitions), sampling parameters taken from the harness itself
(`temperature=0.7 / top_p=0.8 / top_k=20 / repetition_penalty=1.05`), sent through the
same local router the production traffic uses.

### 3.1 Speed by context tier

| Context tier | Decode | Tokens per step | Step latency | TTFT (cold) | Note |
|---:|---:|---:|---:|---:|---|
| 32K | **122.6 tok/s** | 4.59 | 36.6 ms | 24.1 s | |
| 96K | **136.2 tok/s** | 4.66 | - | 84.3 s | only 124 tokens generated that run - small sample |
| 160K | **77.2 tok/s** | 3.64 | 45.0 ms | 175.1 s | |

> Every number is traceable to a log listed in `docs/benchmarks.en.md`.

### 3.2 Before / after (same conditions)

| Change | Tier | Before | After | Delta |
|---|---:|---:|---:|---:|
| Draft sampling `greedy` -> `probabilistic` | 32K | 113.5 | 122.6 | +8% |
| same | 96K | 89.8 | 136.2 | **+52%** |
| same | 160K | 63.8 | 77.2 | **+21%** |
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` | 32K | 37.8 ms/step | 36.8 ms/step | -2.6% |
| same | 160K | 46.4 ms/step | 45.0 ms/step | -3.0% |
| Power 225 W -> 260 W | 32K | 37.5 ms/step | 37.3 ms/step | -0.5% (noise) |

### 3.3 Reference point

Official DFlash2 on **H200 + stock Qwen3.8-27B** (`temperature=1.0 / top_p=0.95 /
top_k=20`): acceptance length 4.10 (MT-Bench) to 5.46 (GSM8K), single-stream throughput
184-236 tok/s. On this machine the 32K tier reaches acceptance length 4.83 with a
position-0 hit rate of **88.7%**, against the officially published **85.4%** - so at short
context **the draft/target pairing is healthy**; the gap is hardware generation plus long
context behaviour.

---

## 4. Findings

1. **Draft sampling method is the single biggest win.** The draft defaults to `greedy`
   (argmax only) while production samples at `temperature=0.7`; flattening the target
   distribution makes most proposals miss. Switching to
   `draft_sample_method: probabilistic` raised long-context acceptance from 0.28 to 0.38
   and decode by 20%-52%.
2. **`VLLM_MARLIN_USE_ATOMIC_ADD=1` is worth ~3%.** vLLM itself prints a hint about this
   at startup; single-stream decode is exactly the small-`size_n` case. Caveat: the
   floating-point accumulation order becomes non-deterministic, so output is no longer
   bit-reproducible at a fixed seed.
3. **The power limit is nearly worthless.** 225 W -> 260 W is worth 1-3% (within noise)
   while pushing GPU temperature from 69 C to 79 C. Production stays at **225 W**.
4. **Bigger prefill chunks do not fit.** `--max-num-batched-tokens` 16384 -> 24576 costs
   1.93 GiB of peak activation and 1.93 GiB of KV pool (6.71 -> 4.78 GiB); vLLM refuses to
   start: `To serve ... max seq len (225280), 4.79 GiB KV cache is needed, which is larger
   than available 4.66 GiB`.
5. **Draft precision is not the bottleneck.** Swapping in the official lossless BF16 draft
   (3.849 GB, sha256 verified) changed acceptance by only +0.4% (32K) / +1.8% (160K) while
   making each step 4% slower - a net loss.
6. **The long-context acceptance decay is architectural** (the draft's sliding window is
   only 2048), not a model-mismatch or quantization issue. Per-position data: position 0
   loses 8 points, position 6 loses 25 points.
7. **GDN decode already runs on Triton**: `Falling back to the Triton GDN decode path: the
   fused CUDA kernel requires ... compute capability 8.0+`. That avenue is exhausted.
8. **KV cache is fp8**: `--kv-cache-dtype fp8`; the 6.71 GiB pool for the 16 full-attention
   layers holds 320,654 tokens (1.42x headroom for a single 225K request).

Full derivations and failure records: [docs/optimization.en.md](docs/optimization.en.md).

---

## 5. Production configuration

Key flags from `deploy/systemd/vllm-qwen38.service`:

```
--tensor-parallel-size 2
--max-model-len 225280
--max-num-seqs 1
--max-num-batched-tokens 16384
--kv-cache-dtype fp8
--gpu-memory-utilization 0.95
--enable-chunked-prefill --enable-prefix-caching
--speculative-config {"method":"dflash","num_speculative_tokens":7,
                      "model":"/root/models/Qwen3.8-27B-DFlash2-W4A16",
                      "attention_backend":"TRITON_ATTN","kv_cache_dtype":"float16",
                      "draft_sample_method":"probabilistic"}
--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8]}
```

Environment: `VLLM_MARLIN_USE_ATOMIC_ADD=1`, `VLLM_KV_BUDGET_FROM_FREE_MEMORY=1`,
`VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1`, `VLLM_DISABLE_TILELANG=1`, and more - see
`deploy/env/`.

---

## 6. One-click deploy

The repo ships an idempotent installer that puts the systemd units, helper scripts and
monitoring config in place in one go:

```bash
git clone https://github.com/ZXC331617/qwen38-27b-2x2080ti-inference-stack.git
cd qwen38-27b-2x2080ti-inference-stack

# see what it would do (changes nothing)
sudo ./deploy/one-click-deploy.sh --dry-run

# install and enable at boot (do not start yet)
sudo ./deploy/one-click-deploy.sh

# install and start now
sudo ./deploy/one-click-deploy.sh --start

# post-install checks / status / rollback
sudo ./deploy/one-click-deploy.sh --verify
sudo ./deploy/one-click-deploy.sh --status
sudo ./deploy/one-click-deploy.sh --uninstall
```

What it does: installs `deploy/scripts/*` into `/usr/local/bin`, renders
`deploy/systemd/*.service|*.timer` into `/etc/systemd/system`, runs `systemctl daemon-reload`,
then enables (and optionally starts) the services.

**Paths are not hard-coded.** The defaults (`/root/vLLM-2080Ti-Definitive-main`,
`/root/models/...`) can be replaced at install time — the script rewrites them inside the units:

```bash
sudo ./deploy/one-click-deploy.sh \
  --vllm-dir  /opt/vllm \
  --model-dir /opt/models/main \
  --draft-dir /opt/models/draft \
  --power-limit 250 \
  --with-monitoring --start
```

Other switches: `--no-fan-curve` (headless box, no X server), `--no-prefix-warm` (skip the
prefix-cache warm-up timer), `-y` (non-interactive), `--dry-run` (print only).

> Any unit it overwrites is backed up to `<unit>.bak-oneclick-<timestamp>`, and `--uninstall`
> rolls everything back. The script does **not** download weights; `--verify` checks that both
> model directories exist.

---

## 7. Layout

```
README.md                Chinese (default)
README.en.md             English
docs/
  benchmarks.md          per-tier measurements (Chinese)
  optimization.md        optimisation log and findings (Chinese)
  deployment.md          step-by-step deployment (Chinese)
bench/
  bench-harness-real.py  benchmark script (reproduces docs/benchmarks)
  run-bench.ps1          benchmark launcher
deploy/
  one-click-deploy.sh    one-click install / uninstall / verify
  systemd/               systemd unit sources
  scripts/               power-limit / fan-curve scripts
  monitoring/            Prometheus / Grafana / exporter config
  env/                   environment variable inventory (sanitised)
  inventory.md           full host inventory
```

---

## 8. License

Code and configuration: Apache-2.0 (see [LICENSE](LICENSE)).
Model weights and third-party components follow their own repositories' licenses.
