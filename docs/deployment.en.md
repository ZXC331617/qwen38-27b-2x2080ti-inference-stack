# From Scratch: Qwen3.8-27B Production Inference Stack on 2×RTX 2080 Ti

[简体中文](deployment.md) | **English**

This is the complete, from-zero reproduction procedure. All artifacts live under
[`../deploy/`](../deploy/); per-file details are in [`../deploy/README.md`](../deploy/README.md)
and [`../deploy/systemd/MANIFEST.md`](../deploy/systemd/MANIFEST.md).

---

## 1. Target environment

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.4 (kernel 6.8.0-138-generic) |
| GPU | 2 × RTX 2080 Ti (SM75 / CC 7.5), 22,528 MiB each, **NVLink bridge (topology NV2)** |
| Driver | 595.99.02 |
| CUDA | 13.0 (`/usr/local/cuda-13.0`, measured `V13.0.88`) |
| Python | 3.12 (uv-managed `cpython-3.12-linux-x86_64-gnu`) |
| PyTorch | 2.13.0+cu130 |
| Engine | **vLLM 2080 Ti Definitive Edition** (fork, `FORK_RELEASE=0.2.2-post2`, upstream github.com/weicj) |
| CPU / RAM / Disk | 36 logical cores / 31 GiB / 468 GB NVMe (≈194 GB free in production) |

> ⚠️ **SM75 is the dividing line of this design.** The 2080 Ti has no bf16 tensor cores and no
> FP8 tensor cores, and the fused GDN kernel requires CC ≥ 8.0. This design therefore fixes
> `--dtype half` + **int4 W4A16 weights** + **fp8 KV cache**, with GDN on the Triton path.
> Do not copy these trade-offs onto 30/40-series cards.

---

## 2. Model files

| Model | Path | Size | Notes |
|---|---|---|---|
| Main | `/root/models/Coder390-HUHUI-INC-A-W4A16` | ≈19 GB | W4A16 compressed-tensors, int4 group-128 symmetric |
| Draft | `/root/models/Qwen3.8-27B-DFlash2-W4A16` | ≈1.5 GB | DFlash2 draft, 5 layers / sliding window 2048 / block_size 8 |

The main model contains `model.safetensors` (18.57 GB) + `model_mtp.safetensors` (0.85 GB, MTP
head, unquantized). The `recipe.yaml` ignore list is
`[lm_head, re:.*embed_tokens.*, re:.*mtp.*, re:.*visual.*]` — i.e. **the vision tower, lm_head
and MTP head stay at full precision**.

Verbatim `config.json` / `generation_config.json` / `recipe.yaml` for both models:
[`../deploy/models/`](../deploy/models/).

---

## 3. Install order (nine steps)

`$REPO` denotes the repository root.

### Step 0 · Driver and toolchain

```bash
nvidia-smi                       # both cards visible, NVLink topology NV2
nvidia-smi topo -m
/usr/local/cuda-13.0/bin/nvcc --version   # CUDA 13.0
```

### Step 1 · vLLM fork and venv

```bash
git clone https://github.com/weicj/vLLM-2080Ti-Definitive   # or place the fork directory directly
cd /root/vLLM-2080Ti-Definitive-main
python3.12 -m venv .venv
.venv/bin/pip install torch==2.13.0+cu130
# install the fork's vllm wheel (0.29.1rc0+33.gb23433088b)
.venv/bin/python -c "import vllm,torch;print(vllm.__version__,torch.__version__)"
```

**FlashQLA** (the SM70/SM75 GDN prefill kernel) must sit at `.deps/FlashQLA-SM70-SM75` inside the
fork, referenced via `FLASHQLA_ROOT` + `PYTHONPATH` (see env vars below).

Version metadata: [`../deploy/versions/`](../deploy/versions/).

### Step 2 · Place the models

```bash
mkdir -p /root/models
# copy both models to the paths above
ls -l /root/models/Coder390-HUHUI-INC-A-W4A16
ls -l /root/models/Qwen3.8-27B-DFlash2-W4A16
```

### Step 3 · GPU power and cooling (before the inference service)

```bash
install -m 0755 "$REPO/deploy/scripts/gpu-power-limit.sh" /usr/local/bin/
install -m 0755 "$REPO/deploy/scripts/gpu-fan-curve.sh"    /usr/local/bin/
cp "$REPO/deploy/systemd/gpu-power-limit.service" /etc/systemd/system/
cp "$REPO/deploy/systemd/gpu-fan-curve.service"   /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now gpu-power-limit.service gpu-fan-curve.service
```

- `gpu-power-limit.sh` locks the power limit to **225 W** (factory default 260 W). At 260 W the
  performance gain is within noise, but temps rise from 69 °C to 80 °C — production chose 225 W.
- `gpu-fan-curve.sh` needs an X server; `gpu-fan-curve.service` starts it.

### Step 4 · The inference service

```bash
cp "$REPO/deploy/systemd/vllm-qwen38.service" /etc/systemd/system/
install -m 0755 "$REPO/deploy/scripts/vllm-warmup.sh" /usr/local/bin/
cp "$REPO/deploy/systemd/vllm-warmup.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now vllm-qwen38.service
```

`ExecStartPost` in `vllm-qwen38.service` **asynchronously** triggers `vllm-warmup.service` for
kernel warm-up (`--no-block`, non-blocking). First start compiles/warms kernels;
`TimeoutStartSec=1800` is normal.

### Step 5 · Health checks

```bash
curl -s http://127.0.0.1:8000/v1/models | head
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Coder390-HUHUI-INC-A-W4A16","messages":[{"role":"user","content":"1+1="}],"max_tokens":16}'
```

### Step 6 · Prefix-cache warm-up (optional but recommended)

```bash
install -m 0755 "$REPO/deploy/scripts/wb-prefix-warm.py" /usr/local/bin/
install -m 0755 "$REPO/deploy/scripts/wb-prefix-warm.sh" /usr/local/bin/
cp "$REPO/deploy/systemd/wb-prefix-warm.service" "$REPO/deploy/systemd/wb-prefix-warm.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now wb-prefix-warm.timer
```

> ⚠️ This timer replays prefix requests periodically. **Stop it before benchmarking**
> (see [optimization.en.md](optimization.en.md), section 5).

### Step 7 · Monitoring stack (optional)

```bash
cp "$REPO/deploy/monitoring/units/"*.service /etc/systemd/system/
install -m 0644 "$REPO/deploy/monitoring/prometheus.yml" /etc/prometheus/prometheus.yml
cp -r "$REPO/deploy/monitoring/grafana-provisioning/"* /etc/grafana/provisioning/
# use the sanitized grafana.ini as a template, then fill in your own admin_password / secret_key
cp "$REPO/deploy/monitoring/grafana.ini.sanitized" /etc/grafana/grafana.ini
systemctl daemon-reload
systemctl enable --now prometheus grafana-server node_exporter nvidia_gpu_exporter
```

Ports and health checks: [`../deploy/monitoring/ports.md`](../deploy/monitoring/ports.md).

### Step 8 · Verify autostart

```bash
systemctl is-enabled vllm-qwen38.service gpu-power-limit.service gpu-fan-curve.service \
                   vllm-warmup.service wb-prefix-warm.timer \
                   prometheus grafana-server node_exporter nvidia_gpu_exporter
```

---

## 4. Production service parameters

Full `ExecStart` at
[`../deploy/systemd/vllm-qwen38.service`](../deploy/systemd/vllm-qwen38.service):

```
--model /root/models/Coder390-HUHUI-INC-A-W4A16
--served-model-name Coder390-HUHUI-INC-A-W4A16 Qwen3.8-27B-Coder390-Abliterated-W4A16 Qwen3.8-27B-Uncensored-AWQ
--dtype half
--quantization compressed-tensors
--tensor-parallel-size 2
--gpu-memory-utilization 0.95
--max-model-len 225280
--max-num-seqs 1
--max-num-batched-tokens 16384
--kv-cache-dtype fp8
--mamba-cache-mode align
--enable-chunked-prefill --enable-prefix-caching --enable-prompt-tokens-details
--reasoning-parser qwen3
--enable-auto-tool-choice --tool-call-parser qwen3_xml
--limit-mm-per-prompt '{"image":64,"video":0,"audio":0}'
--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'
--per-request-spec-decode-metrics detailed
--speculative-config '{"method":"dflash","num_speculative_tokens":7,
  "model":"/root/models/Qwen3.8-27B-DFlash2-W4A16",
  "attention_backend":"TRITON_ATTN","kv_cache_dtype":"float16",
  "draft_sample_method":"probabilistic"}'
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8],
  "max_cudagraph_capture_size":8}'
```

Key environment variables (full list:
[`../deploy/env/production-env.md`](../deploy/env/production-env.md)):

```
CUDA_VISIBLE_DEVICES=0,1
TORCH_CUDA_ARCH_LIST=7.5
VLLM_DISABLE_TILELANG=1
VLLM_KV_CACHE_LAYOUT=BLHNC
VLLM_SM75_SPEC_SYNC_MODE=safe
VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1
VLLM_KV_BUDGET_FROM_FREE_MEMORY=1
VLLM_KV_BUDGET_REDUNDANCY_MIB=150
VLLM_MARLIN_USE_ATOMIC_ADD=1
FLASHQLA_ROOT=/root/vLLM-2080Ti-Definitive-main/.deps/FlashQLA-SM70-SM75
```

Common traps:

- `--max-num-seqs 1` + `cudagraph_capture_sizes [8]`: keeps the DFlash2 verify step inside one
  CUDA Graph — the standard pairing for single-stream.
- `VLLM_KV_BUDGET_FROM_FREE_MEMORY=1`: the KV pool is allocated from **free** VRAM, so peak
  activation and KV pool are zero-sum. Touching `--max-num-batched-tokens` directly squeezes KV
  (see optimization.en.md, section 6).
- `--enable-auto-tool-choice` + `--tool-call-parser qwen3_xml`: **required** for tool-calling
  clients such as harness / Codex, otherwise you get
  `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`.

---

## 5. Verification checklist

| Check | Command | Expected |
|---|---|---|
| Service up | `systemctl is-active vllm-qwen38` | `active` |
| Model list | `curl -s 127.0.0.1:8000/v1/models` | all 3 aliases |
| Single-stream limit | `curl -s 127.0.0.1:8000/metrics \| grep max_num_seqs` | 1 |
| KV pool | startup log `grep -i "GPU KV cache size"` | ≈320,654 tokens |
| Power limit | `nvidia-smi --query-gpu=power.limit --format=csv` | 225.00 W |
| Monitoring | `curl -s 127.0.0.1:9090/-/ready`, `curl -s 127.0.0.1:3000/api/health` | 200 |

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `c10::Half != c10::Float8_e4m3fn` | FP8 weights loaded; unsupported on SM75 — use W4A16 |
| `KV cache is needed, which is larger than available` | `--max-num-batched-tokens` too large; revert to 16384 |
| `"auto" tool choice requires ...` | missing `--enable-auto-tool-choice` / `--tool-call-parser` |
| Slow first start (tens of seconds) | initial kernel compile/warm-up; cached afterwards via `TORCHINDUCTOR_CACHE_DIR` |
| Long-context decode drops | check that `draft_sample_method` is `probabilistic` |
| Accept-rate numbers look absurd | `wb-prefix-warm.timer` is replaying requests; stop it before measuring |

---

## 7. Rollback

```bash
# Unit files: historical .bak-* versions live in deploy/systemd/backups/
cp "$REPO/deploy/systemd/backups/<needed-version>" /etc/systemd/system/vllm-qwen38.service
systemctl daemon-reload && systemctl restart vllm-qwen38.service

# Stop inference entirely
systemctl disable --now vllm-qwen38.service
```

Power/fan units are independent and can be stopped separately. This repo does **not** ship the
weight files; `deploy/models/` only contains configs.
