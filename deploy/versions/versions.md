# Versions & model configs (model host snapshot)

Source host: `ubuntu@192.168.5.88` (`ubuntu-desktop`), kernel `6.8.0-138-generic`.
Read-only pull on 2026-10-14. Nothing on the host was modified.

## Hardware / driver

| item | value |
|---|---|
| GPU 0 | NVIDIA GeForce RTX 2080 Ti, 22528 MiB |
| GPU 1 | NVIDIA GeForce RTX 2080 Ti, 22528 MiB |
| driver | 595.99.02 |
| compute capability | SM 7.5 (`TORCH_CUDA_ARCH_LIST=7.5` in the service) |

## vLLM fork: `/root/vLLM-2080Ti-Definitive-main`

| item | value |
|---|---|
| `VERSION` | `0.2.2-post2` |
| `FORK_RELEASE` | `0.2.2-post2` |
| `BASE_VLLM_VERSION` | `0.29.1rc0-33-gb23433088b` |
| `UPSTREAM_COMMIT` | `b23433088bf29980d20dff1bd64c753dd8883506` |
| `PROJECT_NAME` | `vLLM 2080 Ti Definitive Edition` |
| `RUNTIME_IDENTITY` | `vllm-def-cu130` |
| `VALIDATED_CUDA_VERSION` | `13.0` |
| `VALIDATED_TORCH_VERSION` | `2.13.0+cu130` |
| `PRIMARY_PYTHON_VERSION` | `3.12` |
| `PRIMARY_UBUNTU_VERSION` | `26.04+` (target), legacy branch `v0.1.x` = CUDA 12.8 / torch 2.11.0 |
| git | `NOT_A_GIT_REPO` — the directory has no `.git`; versioning is carried by `VERSION` / `PROJECT_RELEASE.env` only |

### Measured from the fork venv (`.venv/bin/python`)

```
Python 3.12.14
torch 2.13.0+cu130  cuda 13.0
vllm 0.29.1rc0+33.gb23433088b
transformers 5.17.0
```

The measured values match `PROJECT_RELEASE.env` exactly (`VALIDATED_TORCH_VERSION=2.13.0+cu130`,
`BASE_VLLM_VERSION=0.29.1rc0-33-gb23433088b`), so the published preset and the installed
venv are in sync.

### Runtime wiring used by `vllm-qwen38.service`

- `PATH` prepends `/root/vLLM-2080Ti-Definitive-main/.venv/bin`, CUDA 13.0 toolchain
  (`/usr/local/cuda-13.0`, `CUDACXX`, `CUDA_HOME`, `CUDA_PATH`).
- `PYTHONPATH` = fork root + `.deps/FlashQLA-SM70-SM75`; `FLASHQLA_ROOT` points at the same
  directory, so the SM70/SM75 FlashQLA GDN prefill kernel is loaded from `.deps`, not from
  site-packages.
- `VLLM_DISABLE_TILELANG=1`, `VLLM_KV_CACHE_LAYOUT=BLHNC`, `VLLM_SM75_SPEC_SYNC_MODE=safe`,
  `VLLM_MARLIN_USE_ATOMIC_ADD=1`, `VLLM_KV_BUDGET_FROM_FREE_MEMORY=1`,
  `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1`, `TORCH_CUDA_ARCH_LIST=7.5`.
- Dedicated caches: `TORCH_EXTENSIONS_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`
  all live under the fork directory (they are excluded from the pulled tree).

## Models

### `Coder390-HUHUI-INC-A-W4A16` — production model (port 8000)

Also served under the aliases `Qwen3.8-27B-Coder390-Abliterated-W4A16` and
`Qwen3.8-27B-Uncensored-AWQ`.

| item | value |
|---|---|
| architectures | `Qwen3_5ForConditionalGeneration` |
| model_type | `qwen3_5` |
| modality | text + vision (`model.visual.*` tower, `preprocessor_config.json`, `video_preprocessor_config.json`) |
| quantization | `compressed-tensors`, `pack-quantized`, W4A16 (int4, group_size 128, symmetric), targets `Linear` |
| quant ignore | `model.visual.*` (all qkv/proj/fc), `lm_head`, `re:^mtp.*` |
| recipe.yaml | `QuantizationModifier` → `scheme: W4A16`, ignore `[lm_head, re:.*embed_tokens.*, re:.*mtp.*, re:.*visual.*]` |
| generation_config | `temperature 1.0`, `top_k 20`, `top_p 0.95`, `do_sample true`, eos `[248046, 248044]`, bos/pad `248044` |
| transformers_version (in config) | `5.17.0` |
| weights on host | `model.safetensors` 18.57 GB, `model_mtp.safetensors` 0.85 GB (MTP head kept, unquantized) |

The separate `model_mtp.safetensors` plus the `re:^mtp.*` quant ignore list is what lets the
service offer MTP-style speculation while the main trunk stays W4A16.

### `Qwen3.8-27B-DFlash2-W4A16` — DFlash2 draft model

Used by `--speculative-config '{"method":"dflash","num_speculative_tokens":7, ...}'`.

| item | value |
|---|---|
| architectures | `DFlash2DraftModel` |
| model_type | `qwen3` (draft variant) |
| layers / hidden | 5 layers, hidden_size 5120, intermediate 17408 |
| attention | 32 heads / 8 KV heads, head_dim 128, `sliding_window 2048`, 5 sliding layers |
| vocab / ctx | vocab_size 248320, max_position_embeddings 262144 |
| dflash_config | `block_size 8`, `conv_group_size 16`, `conv_kernel_size 2`, `mask_token_id 248070`, `selector_rank 256`, `selector_top_k 16`, `target_layer_ids [5,19,33,47,61]`, `num_target_layers 64` |
| quantization | `compressed-tensors` W4A16 (int4, group 128, symmetric) on `fc` / `o_proj` / `mlp.{gate,up,down,gate_up}_proj` |
| dtype | `bfloat16` |
| missing on host | no `generation_config.json`, no `recipe.yaml` |
| weights on host | `model.safetensors` 1.51 GB |

The draft model taps the trunk at layers 5/19/33/47/61, so the production service runs
`--max-num-seqs 1` with `cudagraph_capture_sizes [8]` to keep the DFlash2 verify step inside
a single captured graph.

## Files pulled here

- `deploy/versions/VERSION`, `deploy/versions/PROJECT_RELEASE.env` — verbatim.
- `deploy/models/Coder390-HUHUI-INC-A-W4A16/{config.json,generation_config.json,recipe.yaml}` — verbatim.
- `deploy/models/Qwen3.8-27B-DFlash2-W4A16/config.json` — verbatim (the only config file present).

Byte counts and sha256 for every file are in `deploy/systemd/MANIFEST.md` and
`deploy/raw/C1/verify.txt` (all rows `OK`, 78/78).
