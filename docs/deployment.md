# 从零部署：Qwen3.8-27B 在 2×RTX 2080 Ti 上的生产推理栈

**简体中文** | [English](deployment.en.md)

本文是从**零开始复现**这套推理环境的完整步骤。所有配置件都在 [`../deploy/`](../deploy/) 下，
逐文件说明见 [`../deploy/README.md`](../deploy/README.md) 与
[`../deploy/systemd/MANIFEST.md`](../deploy/systemd/MANIFEST.md)。

---

## 一、目标环境

| 项 | 值 |
|---|---|
| 系统 | Ubuntu 22.04.4（内核 6.8.0-138-generic） |
| GPU | 2 × RTX 2080 Ti（SM75 / CC 7.5），22,528 MiB/卡，**NVLink 桥接（拓扑 NV2）** |
| 驱动 | 595.99.02 |
| CUDA | 13.0（`/usr/local/cuda-13.0`，实测 `V13.0.88`） |
| Python | 3.12（uv 管理的 `cpython-3.12-linux-x86_64-gnu`） |
| PyTorch | 2.13.0+cu130 |
| 推理引擎 | **vLLM 2080 Ti Definitive Edition**（fork，`FORK_RELEASE=0.2.2-post2`，上游 github.com/weicj） |
| CPU / 内存 / 盘 | 36 逻辑核 / 31 GiB / 468 GB NVMe（生产剩余约 194 GB） |

> ⚠️ **SM75 是本方案的分水岭。** 2080 Ti 没有 bf16 张量核、没有 FP8 张量核，
> GDN 融合核也要求 CC ≥ 8.0。因此本方案固定使用 `--dtype half` + **int4 W4A16 权重**
> + **fp8 KV cache**，GDN 走 Triton。换 30/40 系卡不要照抄这些取舍。

---

## 二、模型准备

| 模型 | 路径 | 大小 | 说明 |
|---|---|---|---|
| 主模型 | `/root/models/Coder390-HUHUI-INC-A-W4A16` | ≈19 GB | W4A16 compressed-tensors，int4 group-128 对称 |
| 草稿 | `/root/models/Qwen3.8-27B-DFlash2-W4A16` | ≈1.5 GB | DFlash2 草稿，5 层 / 滑窗 2048 / block_size 8 |

主模型含 `model.safetensors`（18.57 GB）+ `model_mtp.safetensors`（0.85 GB，MTP 头，未量化）。
`recipe.yaml` 的 ignore 列表为 `[lm_head, re:.*embed_tokens.*, re:.*mtp.*, re:.*visual.*]`
—— 即 **视觉塔、lm_head、MTP 头保持原精度**。

两模型的 `config.json` / `generation_config.json` / `recipe.yaml` 原文见
[`../deploy/models/`](../deploy/models/)。

---

## 三、安装顺序（九步）

`$REPO` 指本仓库根目录。

### 第 0 步 · 驱动与工具链

```bash
# 驱动（生产 595.99.02，需支持 CUDA 13.x）
nvidia-smi                       # 确认两张卡可见、NVLink 拓扑为 NV2
nvidia-smi topo -m
# CUDA Toolkit 13.0 → /usr/local/cuda-13.0
/usr/local/cuda-13.0/bin/nvcc --version
```

### 第 1 步 · vLLM fork 与 venv

```bash
git clone https://github.com/weicj/vLLM-2080Ti-Definitive   # 或直接放置 fork 目录
cd /root/vLLM-2080Ti-Definitive-main
python3.12 -m venv .venv
.venv/bin/pip install torch==2.13.0+cu130
# 安装 fork 提供的 vllm wheel（0.29.1rc0+33.gb23433088b）
.venv/bin/python -c "import vllm,torch;print(vllm.__version__,torch.__version__)"
```

**FlashQLA（SM70/SM75 专用 GDN prefill 核）** 必须放在 fork 的 `.deps/FlashQLA-SM70-SM75`，
并通过 `FLASHQLA_ROOT` + `PYTHONPATH` 指向它（见下节环境变量）。

版本元数据原文见 [`../deploy/versions/`](../deploy/versions/)。

### 第 2 步 · 模型落盘

```bash
mkdir -p /root/models
# 把两个模型放到上面的路径
ls -l /root/models/Coder390-HUHUI-INC-A-W4A16
ls -l /root/models/Qwen3.8-27B-DFlash2-W4A16
```

### 第 3 步 · GPU 功耗与散热（先于推理服务）

```bash
install -m 0755 "$REPO/deploy/scripts/gpu-power-limit.sh" /usr/local/bin/
install -m 0755 "$REPO/deploy/scripts/gpu-fan-curve.sh"    /usr/local/bin/
cp "$REPO/deploy/systemd/gpu-power-limit.service" /etc/systemd/system/
cp "$REPO/deploy/systemd/gpu-fan-curve.service"   /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now gpu-power-limit.service gpu-fan-curve.service
```

- `gpu-power-limit.sh` 把功耗墙锁到 **225 W**（出厂默认 260 W）。实测 260 W 的性能收益在
  噪声内，但温度从 69 °C 涨到 80 °C —— 生产选择 225 W。
- `gpu-fan-curve.sh` 依赖 X server，用 `gpu-fan-curve.service` 拉起。

### 第 4 步 · 推理服务本体

```bash
cp "$REPO/deploy/systemd/vllm-qwen38.service" /etc/systemd/system/
install -m 0755 "$REPO/deploy/scripts/vllm-warmup.sh" /usr/local/bin/
cp "$REPO/deploy/systemd/vllm-warmup.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now vllm-qwen38.service
```

`vllm-qwen38.service` 的 `ExecStartPost` 会**异步**触发 `vllm-warmup.service` 做 kernel 预热
（`--no-block`，不阻塞启动）。首次启动要编译 / 预热 kernel，`TimeoutStartSec=1800` 属正常。

### 第 5 步 · 健康检查

```bash
curl -s http://127.0.0.1:8000/v1/models | head
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Coder390-HUHUI-INC-A-W4A16","messages":[{"role":"user","content":"1+1="}],"max_tokens":16}'
```

### 第 6 步 · 前缀缓存预热（可选，但强烈建议）

```bash
install -m 0755 "$REPO/deploy/scripts/wb-prefix-warm.py" /usr/local/bin/
install -m 0755 "$REPO/deploy/scripts/wb-prefix-warm.sh" /usr/local/bin/
cp "$REPO/deploy/systemd/wb-prefix-warm.service" "$REPO/deploy/systemd/wb-prefix-warm.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now wb-prefix-warm.timer
```

> ⚠️ 该定时器会周期性重放前缀请求，**测速时必须先停掉**（见 [optimization.md](optimization.md) 第五节）。

### 第 7 步 · 监控栈（可选）

```bash
cp "$REPO/deploy/monitoring/units/"*.service /etc/systemd/system/
install -m 0644 "$REPO/deploy/monitoring/prometheus.yml" /etc/prometheus/prometheus.yml
cp -r "$REPO/deploy/monitoring/grafana-provisioning/"* /etc/grafana/provisioning/
# grafana.ini 用脱敏版作模板，重新填回自己的 admin_password / secret_key
cp "$REPO/deploy/monitoring/grafana.ini.sanitized" /etc/grafana/grafana.ini
systemctl daemon-reload
systemctl enable --now prometheus grafana-server node_exporter nvidia_gpu_exporter
```

端口与健康检查实测见 [`../deploy/monitoring/ports.md`](../deploy/monitoring/ports.md)。

### 第 8 步 · 开机自启确认

```bash
systemctl is-enabled vllm-qwen38.service gpu-power-limit.service gpu-fan-curve.service \
                   vllm-warmup.service wb-prefix-warm.timer \
                   prometheus grafana-server node_exporter nvidia_gpu_exporter
```

---

## 四、生产服务关键参数

完整 `ExecStart` 见 [`../deploy/systemd/vllm-qwen38.service`](../deploy/systemd/vllm-qwen38.service)：

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

关键环境变量（完整清单见 [`../deploy/env/production-env.md`](../deploy/env/production-env.md)）：

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

几个容易踩的点：

- `--max-num-seqs 1` + `cudagraph_capture_sizes [8]`：让 DFlash2 的 verify 步落进单张
  CUDA Graph，是单流场景的固定搭配。
- `VLLM_KV_BUDGET_FROM_FREE_MEMORY=1`：KV 池按**剩余**显存分配，所以峰值激活与 KV 池是
  零和 —— 动 `--max-num-batched-tokens` 会直接挤压 KV（见 optimization.md 第六节）。
- `--enable-auto-tool-choice` + `--tool-call-parser qwen3_xml`：harness / Codex 这类
  工具调用客户端**必须**这两个参数，否则会报
  `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`。

---

## 五、验证清单

| 检查项 | 命令 | 期望 |
|---|---|---|
| 服务在跑 | `systemctl is-active vllm-qwen38` | `active` |
| 模型列表 | `curl -s 127.0.0.1:8000/v1/models` | 3 个别名都在 |
| 单流约束 | `curl -s 127.0.0.1:8000/metrics \| grep max_num_seqs` | 1 |
| KV 池 | 启动日志 `grep -i "GPU KV cache size"` | 约 320,654 token |
| 功耗墙 | `nvidia-smi --query-gpu=power.limit --format=csv` | 225.00 W |
| 监控 | `curl -s 127.0.0.1:9090/-/ready`、`curl -s 127.0.0.1:3000/api/health` | 200 |

---

## 六、故障排查

| 症状 | 原因 / 处理 |
|---|---|
| `c10::Half != c10::Float8_e4m3fn` | 加载了 FP8 权重；SM75 不支持，换 W4A16 |
| `KV cache is needed, which is larger than available` | `--max-num-batched-tokens` 太大挤爆 KV 池，回退 16384 |
| `"auto" tool choice requires ...` | 缺 `--enable-auto-tool-choice` / `--tool-call-parser` |
| 启动很慢（几十秒～分钟） | 首次 kernel 编译/预热，正常；之后走 `TORCHINDUCTOR_CACHE_DIR` 缓存 |
| 长上下文 decode 明显掉 | 检查 `draft_sample_method` 是否为 `probabilistic` |
| 接受率数字 > 1 / 不合理 | `wb-prefix-warm.timer` 在重放请求，测速前先停 |

---

## 七、回滚

```bash
# 单元文件：deploy/systemd/backups/ 里有历史 .bak-* 版本
cp "$REPO/deploy/systemd/backups/<需要的版本>" /etc/systemd/system/vllm-qwen38.service
systemctl daemon-reload && systemctl restart vllm-qwen38.service

# 彻底停推理
systemctl disable --now vllm-qwen38.service
```

功耗/风扇单元相互独立，可单独停。本仓库**不含权重文件本体**，`deploy/models/` 只有配置。
