# 模型主机规格清单（实测）

采集时间：2026-10-14 20:28（主机本地时间，Asia/Shanghai）
采集方式：只读命令 + `sshpass`/`scp`，原始输出全部落盘在 `deploy/raw/C3/`

- 原始输出（普通用户）：`deploy/raw/C3/host-specs.txt`（767 行，含 `nvidia-smi -q` 全文）
- 原始输出（sudo）：`deploy/raw/C3/host-specs-root.txt`（300 行，`/root` 下模型与 fork 版本）
- 采集脚本：`deploy/raw/C3/collect.sh`、`deploy/raw/C3/collect-root.sh`
- 主机：`ubuntu@192.168.5.88`（实验网段内网地址，按约定保留）

> 脱敏约定：公网 IP / 域名统一写 `HOST`；token / 口令写 `TOKEN` / `PASSWORD`；内网 `192.168.x.x` 原样保留。

---

## 1. GPU

| 项 | 值 | 来源命令 |
|---|---|---|
| 型号 | NVIDIA GeForce RTX 2080 Ti（GeForce 品牌，消费卡） | `nvidia-smi -q` |
| 数量 | 2 | `nvidia-smi` |
| 单卡显存 | 22528 MiB（22 GiB GDDR6） | `nvidia-smi -q`（FB Memory Total） |
| 驱动版本 | 595.99.02 | `nvidia-smi` |
| CUDA 驱动版本 | 13.2 | `nvidia-smi` |
| VBIOS | 90.02.42.00.42 | `nvidia-smi -q` |
| 计算能力 | CC 7.5（SM75，Turing） | venv `torch.cuda.get_device_capability()` |
| Bus ID | GPU0 `00000000:03:00.0`，GPU1 `00000000:04:00.0` | `nvidia-smi` |
| PCIe | Gen3，链路宽度 x16（Max = Current = 16x） | `nvidia-smi -q`（GPU Link Info） |
| NVLink | 可用；每卡 2 条链路，单链路 25.781 GB/s；拓扑 `NV2`（两卡经 NVLink 直连，2 bond） | `nvidia-smi nvlink -s`、`nvidia-smi topo -m` |
| ECC | N/A（GeForce 不支持） | `nvidia-smi -q` |
| 功耗限值 | 当前 225.00 W；Requested 225.00 W；Default 260.00 W；Min 100.00 W；Max 290.00 W | `nvidia-smi -q -d POWER` |
| 采集瞬时功耗 | GPU0 223.88 W，GPU1 223.98 W | `nvidia-smi -q -d POWER` |
| 采集瞬时温度 | GPU0 78 C，GPU1 67 C | `nvidia-smi -q -d TEMPERATURE` |
| 采集瞬时风扇 | GPU0 75%，GPU1 44% | `nvidia-smi` |
| 采集瞬时利用率 | GPU0 96%，GPU1 95%；显存各占用 21924 MiB | `nvidia-smi` |
| 占用进程 | `VLLM::Worker_TP0`（PID 363353）、`VLLM::Worker_TP1`（PID 363354） | `nvidia-smi` |

双卡口径（TP=2 场景）：可用显存合计 45056 MiB；生产服务以 `--gpu-memory-utilization 0.95` 运行，实测两卡各占 21924 MiB。

## 2. CPU 与内存

| 项 | 值 | 来源命令 |
|---|---|---|
| 型号 | Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz | `lscpu` |
| 架构 | x86_64（32-bit, 64-bit 运行模式），Little Endian | `lscpu` |
| 逻辑 CPU | 36（在线 0-35） | `lscpu` |
| 物理核 | 18 核/座 × 1 座，每核 2 线程 | `lscpu` |
| 频率 | 最小 1200 MHz，最大 3600 MHz | `lscpu` |
| 缓存 | L1d 576 KiB / L1i 576 KiB / L2 4.5 MiB / L3 45 MiB | `lscpu` |
| NUMA | 1 个节点（节点 0 = CPU 0-35） | `lscpu` |
| 虚拟化 | VT-x（vmx 标志） | `lscpu` |
| 内存总量 | 31 GiB（MemTotal 32750972 kB） | `free -g` / `/proc/meminfo` |
| 内存现状 | used 9 GiB / free 0 GiB / buff-cache 21 GiB / available 21 GiB | `free -g` |
| Swap | `/swapfile` 2 GiB，已用 343.5 MiB，PRIO -2 | `swapon --show` |

## 3. 存储

| 项 | 值 | 来源命令 |
|---|---|---|
| 块设备 | 单块 NVMe `nvme0n1` 476.9G（ROTA=0，SSD） | `lsblk` |
| 分区 | `nvme0n1p1` 512M → `/boot/efi`；`nvme0n1p2` 476.4G → `/` | `lsblk` / `df -h` |
| 根分区用量 | 468G 总量 / 251G 已用 / 194G 可用 / 57% | `df -h` |
| `/dev/shm` | 16G（14M 已用） | `df -h` |
| 模型盘 | 与根分区同盘（`/root/models` 在 `/` 上），采集时剩余 194G | `df -h /root` |
| SMART | 未采集（脚本内 `smartctl` 未执行，视为「未安装/未采集」） | — |

## 4. 操作系统与内核

| 项 | 值 | 来源命令 |
|---|---|---|
| 主机名 | `ubuntu-desktop` | `hostname` / `uname -a` |
| 发行版 | Ubuntu 22.04.4 LTS（jammy） | `lsb_release -a` |
| 内核 | `6.8.0-138-generic #138~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC`（2026-08-07 构建） | `uname -a` |
| systemd | 249（249.11-0ubuntu3.22） | `systemd --version` |
| 运行时长 | up 1 day 11:56，load average 2.60 / 2.33 / 2.07 | `uptime` |

## 5. CUDA 与 Python 工具链

| 项 | 值 | 来源命令 |
|---|---|---|
| CUDA Toolkit | 13.0（`Cuda compilation tools, release 13.0, V13.0.88`，Build cuda_13.0.r13.0/compiler.36424714_0） | `/usr/local/cuda/bin/nvcc --version` |
| nvcc 路径 | `/usr/local/cuda-13.0/bin/nvcc`（另有 `/usr/local/cuda-13`、`/usr/local/cuda` 软链） | `ls -l /usr/local/cuda*/bin/nvcc` |
| 系统 Python | 3.10.12（`python3`） | `python3 -V` |
| 服务用 venv | `/root/vLLM-2080Ti-Definitive-main/.venv`（uv 管理） | `ls -l .venv/bin/python*` |
| venv Python | 3.12.14 → `/root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12` | venv python 实测 |
| torch | `2.13.0+cu130` | venv `import torch` |
| torch CUDA | 13.0 | venv `torch.version.cuda` |
| 可见设备 | `device_count()=2`，`get_device_name(0)=NVIDIA GeForce RTX 2080 Ti` | venv `torch.cuda` |
| vLLM（venv） | `0.29.1rc0+33.gb23433088b` | venv `import vllm` |

注意：`TARGET` 主机上 `nvcc` 不在普通用户 PATH 中（脚本内 `nvcc` 报「未找到命令」），必须用绝对路径或 sudo。

## 6. vLLM fork 与模型

| 项 | 值 | 来源 |
|---|---|---|
| fork 目录 | `/root/vLLM-2080Ti-Definitive-main` | `ls -d` |
| `VERSION` | `0.2.2-post2` | `cat VERSION` |
| PROJECT_NAME | `vLLM 2080 Ti Definitive Edition`（作者 `github.com/weicj`） | `PROJECT_RELEASE.env` |
| RUNTIME_IDENTITY | `vllm-def-cu130` | `PROJECT_RELEASE.env` |
| FORK_RELEASE / BASE_VLLM_VERSION | `0.2.2-post2` / `0.29.1rc0-33-gb23433088b` | `PROJECT_RELEASE.env` |
| UPSTREAM_COMMIT | `b23433088bf29980d20dff1bd64c753dd8883506` | `PROJECT_RELEASE.env` |
| 声明运行时 | CUDA `13.0`，torch `2.13.0+cu130`，Python `3.12` | `PROJECT_RELEASE.env` |
| 声明目标平台 | Ubuntu `26.04+`，内核主版本 ≥ `7`，GCC `15`，上游 Docker CUDA `13.0.3` / Ubuntu `22.04` | `PROJECT_RELEASE.env` |
| legacy 分支 | `v0.1.x`，CUDA `12.8`，torch `2.11.0` | `PROJECT_RELEASE.env` |
| git 仓库 | **不是 git 仓库**（无 `.git`），只能靠 `VERSION` / `PROJECT_RELEASE.env` 定位版本 | `git log` 报错原文 |

**实测与声明不一致（按事实记录）**：fork 声明的主目标平台是 Ubuntu 26.04+ / 内核 ≥7 / GCC 15，而本机实际为 Ubuntu 22.04.4 / 内核 6.8 / 系统 Python 3.10；生产服务用 venv（Python 3.12.14 + torch 2.13.0+cu130 + CUDA 13.0）运行正常。

### 6.1 主模型 `/root/models/Coder390-HUHUI-INC-A-W4A16`（19 GB）

| 项 | 值 |
|---|---|
| architectures | `Qwen3_5ForConditionalGeneration` |
| model_type | `qwen3_5` |
| dtype | `bfloat16`（权重已量化，见下） |
| quantization | `compressed-tensors` / `pack-quantized`，int4、group_size 128、symmetric、`memoryless_minmax`、targets = Linear |
| 多模态 | 是（`language_model_only=false`，`image_token_id=248056`，visual blocks 被排除在量化外） |
| 文件 | `model.safetensors` 18,568,390,152 B；`model_mtp.safetensors` 849,400,392 B（MTP 头）；`model.safetensors.index.json`；`tokenizer.json` 19,989,325 B；`chat_template.jinja`；`recipe.yaml`；`preprocessor_config.json`；`video_preprocessor_config.json`；`tokenizer_config.json` |
| 关键生成参数 | `do_sample=true`、`temperature=1.0`、`top_k=20`、`top_p=0.95`、`eos_token_id=[248046,248044]`、`pad_token_id=248044`、`transformers_version=5.17.0` |

### 6.2 草稿模型 `/root/models/Qwen3.8-27B-DFlash2-W4A16`（1.5 GB）

| 项 | 值 |
|---|---|
| architectures | `DFlash2DraftModel` |
| hidden_size | 5120 |
| num_hidden_layers | 5（`layer_types` 全为 `sliding_attention`） |
| num_target_layers | 64（`target_layer_ids = [5, 19, 33, 47, 61]`） |
| attention | 32 heads / 8 KV heads，head_dim 128，`sliding_window=2048` |
| intermediate_size | 17408 |
| vocab_size | 248320 |
| max_position_embeddings | 262144 |
| dflash_config | `block_size=8`、`conv_kernel_size=2`、`conv_group_size=16`、`selector_rank=256`、`selector_top_k=16`、`mask_token_id=248070` |
| 量化 | `compressed-tensors` int4，group_size 128，symmetric |
| 其他 | `dtype=bfloat16`、`transformers_version=5.15.0`、`tie_word_embeddings=false`、无 `generation_config.json` |

### 6.3 `/root/models/` 全量（`du -sh`）

```
55G  Coder390-BF16-ORIG
52G  Coder390-HUHUI-INC-A
19G  Coder390-HUHUI-INC-A-W4A16        <- 生产主模型
11M  _dl_coder390
19G  Huihui-Coder390-Abliterated-W4A16
2.3G Qwen3.8-27B-Coder390-DFlash2-FP8
19G  Qwen3.8-27B-Coder390-INT4-W4A16
3.6G Qwen3.8-27B-DFlash2-BF16
1.5G Qwen3.8-27B-DFlash2-W4A16         <- 生产草稿模型
20G  Qwen3.8-27B-EfficientThink-Uncensored-AWQ-W4A16
20G  Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ
```

## 7. 网络

| 项 | 值 | 来源命令 |
|---|---|---|
| 网卡 | `enp6s0`（UP） | `ip -br a` |
| IPv4 | `192.168.5.88/24` | `ip -br a` |
| IPv6 | 公网 IPv6 前缀 `HOST`（已脱敏）；另有 `fe80::/64` 链路本地 | `ip -br a` |
| 链路速率 | 未采集到（`ethtool` 无输出，需特权或驱动不支持） | `ethtool` |
| 监听端口（前 20 条） | `0.0.0.0:22`(ssh)、`0.0.0.0:8000`(vLLM OpenAI API)、`127.0.0.1:7899`(反向代理隧道)、`127.0.0.53%lo:53`(systemd-resolved)，其余为临时高位端口 | `ss -lntp` |

## 8. 运行中的相关服务（`systemctl list-units --state=running`）

```
gpu-fan-curve.service        GPU fan curve for 2x RTX 2080 Ti (needs nvidia X server)
grafana-server.service       Grafana instance
node_exporter.service        Prometheus Node Exporter
nvidia_gpu_exporter.service  NVIDIA GPU Exporter (nvidia-smi based)
prometheus.service           Prometheus (LLM inference monitoring)
vllm-qwen38.service          vLLM Qwen3.8-27B-Uncensored-AWQ Inference Server (RTX 2080 Ti x2, TP=2)
```

## 9. 生产服务关键启动参数（来自 `systemctl cat vllm-qwen38.service` 原文）

```
ExecStart=/root/vLLM-2080Ti-Definitive-main/.venv/bin/python -m vllm.entrypoints.openai.api_server
    --host 0.0.0.0 --port 8000
    --model /root/models/Coder390-HUHUI-INC-A-W4A16
    --served-model-name Coder390-HUHUI-INC-A-W4A16 Qwen3.8-27B-Coder390-Abliterated-W4A16 Qwen3.8-27B-Uncensored-AWQ
    --dtype half --quantization compressed-tensors
    --tensor-parallel-size 2
    --generation-config auto --override-generation-config '{"max_new_tokens": 24576}'
    --gpu-memory-utilization 0.95 --max-model-len 225280
    --enable-chunked-prefill --max-num-seqs 1 --max-num-batched-tokens 16384
    --kv-cache-dtype fp8 --mamba-cache-mode align --enable-prefix-caching
    --enable-prompt-tokens-details --reasoning-parser qwen3
    --limit-mm-per-prompt '{"image":64,"video":0,"audio":0}'
    --enable-auto-tool-choice --tool-call-parser qwen3_xml
    --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'
    --per-request-spec-decode-metrics detailed
    --speculative-config '{"method":"dflash","num_speculative_tokens":7,
        "model":"/root/models/Qwen3.8-27B-DFlash2-W4A16",
        "attention_backend":"TRITON_ATTN","kv_cache_dtype":"float16",
        "draft_sample_method":"probabilistic"}'
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes":[8],"max_cudagraph_capture_size":8}'
```

环境变量、`Restart` 策略、日志路径与 `ExecStartPost`（异步 kernel 预热）见 `systemd/vllm-qwen38.service` 原文与 `env/production-env.md`。

## 10. 未采集到的项（如实标注）

| 项 | 状态 |
|---|---|
| `ethtool` 链路速率/双工 | 无输出，未采集到 |
| SMART 健康信息 | 未执行（`smartctl` 未确认安装） |
| NVMe 型号/固件版本 | 未采集（`lsblk` 不含型号字段） |
| 主板/BIOS 信息 | 未采集 |
| 公网 IP / 域名 | 按脱敏约定写 `HOST` |
