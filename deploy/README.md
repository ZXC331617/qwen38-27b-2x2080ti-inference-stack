# 部署包总说明：Coder390 / Qwen3.8-27B 在 2×RTX 2080 Ti 上的生产部署

本目录是从**生产模型主机**（Ubuntu 22.04.4 / 2×RTX 2080 Ti / vLLM fork）**原样抽取**的部署快照，
用于把这套推理环境在另一台同规格机器上复现，或作为开源仓库的 `deploy/` 部分。

- 主机规格实测清单：`inventory.md`
- 所有文件均通过 `scp` 从主机原样拉取，**未手工重打内容**；每个文件都做了 `sha256` 与字节数比对（见 `raw/C*/`）
- 脱敏：token / 口令 → `TOKEN` / `PASSWORD`，公网 IP / 域名 → `HOST`；内网 `192.168.x.x` 保留

---

## 1. 目录结构

```
deploy/
  README.md                 ← 本文件：总说明 + 安装顺序
  inventory.md              ← 主机规格实测（GPU/CPU/内存/磁盘/驱动/CUDA/torch/vLLM 版本）
  systemd/                  ← 生产与实验用 systemd 单元原文
    (vllm-qwen38.service, vllm-warmup.service, gpu-power-limit.service,
     gpu-fan-curve.service, wb-prefix-warm.service/.timer)
    experimental/           ← 同主机上的实验单元（coder390 系列、prefilltest、uncensored-et）
    backups/                ← 单元的历史 .bak-* 备份，仅作对照
    dropins/                ← vllm-qwen38 / vllm-warmup 的 drop-in 覆盖（若存在）
    MANIFEST.md             ← 逐文件说明：生产/实验、关键参数、来源路径、sha256
  scripts/                  ← 运维脚本原文（GPU 功耗墙、风扇曲线、kernel 预热、前缀预热）
  monitoring/               ← Prometheus / Grafana / 导出器配置
    prometheus.yml          ← 抓取配置原文
    units/                  ← prometheus / grafana-server / node_exporter / nvidia_gpu_exporter 单元原文
    grafana-provisioning/   ← Grafana provisioning 原文（datasources / dashboards / ...）
    grafana.ini.sanitized   ← grafana.ini 脱敏版
    exporter-args.md        ← 导出器启动参数与进程 cmdline 证据
    ports.md                ← 监控端口与健康检查实测
  env/                      ← 生产服务环境变量（脱敏）
    production-env.md       ← 逐条变量说明表（变量名 / 值 / 作用 / 来源）
    production-env.env.example ← 可直接被 systemd EnvironmentFile= 引用的示例
  versions/                 ← vLLM fork 版本元数据（VERSION / PROJECT_RELEASE.env 等）
  models/                   ← 两个模型的 config.json / generation_config.json / recipe.yaml 原文
  raw/                      ← 采集原始证据（脚本 + 命令原始输出），供审计与复现
```

各子目录实际内容：

| 目录 | 文件数 | 说明 |
|---|---|---|
| `systemd/` | 68 | 生产单元 6 + 实验单元 15 + 备份 46（含 3 个脚本备份）+ `MANIFEST.md`（主机上无 drop-in，已确认） |
| `scripts/` | 5 | `gpu-power-limit.sh`、`gpu-fan-curve.sh`、`vllm-warmup.sh`、`wb-prefix-warm.py`、`wb-prefix-warm.sh` |
| `monitoring/` | 19 | `prometheus.yml`(+备份) / `units/` 4 个 / `grafana-provisioning/` 5 个 / `grafana.ini.sanitized` / `ldap.toml.sanitized` / `exporter-args.md` / `ports.md` / `systemctl-cat.txt` / `checksums.md` / 2 份脱敏说明 |
| `env/` | 4 | `production-env.md`（44 个变量 + 28 个启动参数含义表）、`production-env.env.example`、`raw/` 证据 2 份 |
| `versions/` | 3 | `VERSION`、`PROJECT_RELEASE.env`、`versions.md` |
| `models/` | 4 | 主模型 `config.json` / `generation_config.json` / `recipe.yaml`；草稿模型 `config.json`（草稿无 generation_config / recipe） |
| `raw/` | 57 | C1 / C2 / C3 三路采集脚本与原始输出 |

## 2. 依赖顺序（从零复现）

按以下顺序执行；每一步都给出对应文件。命令中的 `$REPO` 指代本仓库根目录。

**第 0 步 · 硬件与驱动前置**
1. 2×RTX 2080 Ti（SM75 / CC 7.5），NVLink 桥接（生产实测拓扑 `NV2`），PCIe Gen3 x16
2. NVIDIA 驱动（生产：595.99.02，支持 CUDA 13.2）
3. CUDA Toolkit 13.0（`/usr/local/cuda-13.0`，生产实测 `V13.0.88`）
4. Python 3.12（生产用 uv 管理的 `cpython-3.12-linux-x86_64-gnu`）

**第 1 步 · vLLM fork 与 venv**
5. 放置 vLLM fork：`/root/vLLM-2080Ti-Definitive-main`（版本见 `versions/`：`VERSION=0.2.2-post2`，上游 commit `b23433088b…`）
6. 创建 venv 并安装依赖：`python3.12 -m venv .venv` → 安装 `torch==2.13.0+cu130`、`vllm 0.29.1rc0+33.gb23433088b`
7. 编译/放置 FlashQLA 依赖到 `.deps/FlashQLA-SM70-SM75`（SM75 专用，见 `env/production-env.md` 的 `FLASHQLA_ROOT`）

**第 2 步 · 模型**
8. 主模型放 `/root/models/Coder390-HUHUI-INC-A-W4A16`（19 GB，int4 W4A16 compressed-tensors）
9. 草稿模型放 `/root/models/Qwen3.8-27B-DFlash2-W4A16`（1.5 GB，DFlash2 draft）
10. 两模型的 `config.json` 等配置见 `models/`；关键字段摘要在 `inventory.md` 第 6 节

**第 3 步 · GPU 功耗与散热（先于推理服务）**
11. 安装 `scripts/gpu-power-limit.sh` → `/usr/local/bin/gpu-power-limit.sh`（`chmod +x`）
12. 安装 `scripts/gpu-fan-curve.sh` → `/usr/local/bin/gpu-fan-curve.sh`（风扇曲线依赖 X server，生产用 `gpu-fan-curve.service` 拉起）
13. 安装单元：`systemd/gpu-power-limit.service`、`systemd/gpu-fan-curve.service` → `/etc/systemd/system/`
14. `systemctl daemon-reload && systemctl enable --now gpu-power-limit.service gpu-fan-curve.service`

**第 4 步 · 推理服务本体**
15. 安装单元 `systemd/vllm-qwen38.service`（+ `systemd/dropins/` 若有）→ `/etc/systemd/system/`
16. 安装预热脚本 `scripts/vllm-warmup.sh` → `/usr/local/bin/`，单元 `systemd/vllm-warmup.service`
17. `systemctl daemon-reload && systemctl enable --now vllm-qwen38.service`
    - 该服务 `ExecStartPost` 会异步触发 `vllm-warmup.service` 做 kernel 预热（不阻塞启动）
    - 首次启动要编译/预热 kernel，`TimeoutStartSec=1800`，属正常
18. 健康检查：`curl -s http://127.0.0.1:8000/v1/models`

**第 5 步 · 前缀缓存预热（可选）**
19. 安装 `scripts/wb-prefix-warm.py`、`scripts/wb-prefix-warm.sh` → `/usr/local/bin/`
20. 安装 `systemd/wb-prefix-warm.service` + `systemd/wb-prefix-warm.timer`，`systemctl enable --now wb-prefix-warm.timer`

**第 6 步 · 监控栈（可选）**
21. 安装 `monitoring/units/` 下 4 个单元 → `/etc/systemd/system/`（prometheus / grafana-server / node_exporter / nvidia_gpu_exporter）
22. 放置 `monitoring/prometheus.yml` → `/etc/prometheus/prometheus.yml`
23. 放置 Grafana provisioning：`monitoring/grafana-provisioning/*` → `/etc/grafana/provisioning/`；`grafana.ini` 用 `monitoring/grafana.ini.sanitized` 作模板并**重新填回你自己的口令**
24. `systemctl daemon-reload && systemctl enable --now prometheus grafana-server node_exporter nvidia_gpu_exporter`
25. 验证见 `monitoring/ports.md`

## 3. 生产服务关键参数（摘要）

完整 `ExecStart` 见 `inventory.md` 第 9 节与 `systemd/vllm-qwen38.service` 原文：

- `--model /root/models/Coder390-HUHUI-INC-A-W4A16`，`--served-model-name` 三个别名
- `--tensor-parallel-size 2`，`--dtype half`，`--quantization compressed-tensors`
- `--max-model-len 225280`，`--gpu-memory-utilization 0.95`，`--kv-cache-dtype fp8`
- `--max-num-seqs 1`，`--max-num-batched-tokens 16384`，`--enable-chunked-prefill`，`--enable-prefix-caching`
- 投机解码：`--speculative-config {"method":"dflash","num_speculative_tokens":7,...,"draft_sample_method":"probabilistic"}`
- 编译：`--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8],...}`
- 环境变量：`VLLM_MARLIN_USE_ATOMIC_ADD=1`、`VLLM_KV_CACHE_LAYOUT=BLHNC`、`TORCH_CUDA_ARCH_LIST=7.5`、`CUDA_VISIBLE_DEVICES=0,1` 等，逐条见 `env/production-env.md`

## 4. 平台约束（SM75）

- 2080 Ti 是 SM75（Turing）：**没有** bf16 / fp8 张量核加速，bf16 权重只能走软件路径；生产因此用 `--dtype half` + int4 W4A16 权重 + fp8 KV cache
- GDN（gated delta net）解码走 Triton 路径，CUDA 融合核需 CC 8.0+，本机不可用
- 显卡默认功耗 260 W，生产被 `gpu-power-limit.sh` 压到 225 W（实测性能差异在噪声级）

## 5. 脱敏与安全说明

- 本目录所有配置中的口令、token、公网 IP 均已占位符化（`TOKEN` / `PASSWORD` / `HOST`）
- `monitoring/grafana.ini.sanitized` 是**脱敏后**的 Grafana 配置，装机时必须改回自己的 `admin_password` 与 `secret_key`
- 内网地址 `192.168.5.88` 属实验网段，按约定保留
- **需要整改的项（实测发现，如实记录）**：主机上 `grafana.ini` 的 `admin_password` 处于注释态（仍是默认值），且 Grafana 监听 `*:3000`（对外可达）；同理 `nvidia_gpu_exporter`(9835)、`node_exporter`(9100)、`prometheus`(9090) 也监听 `*`。复现时请自行加认证或限制监听地址
- 全树明文密钥自检：对 `deploy/` 下全部文本文件扫描 HF token / `sk-` / 非占位 `password` / `api_key` 模式，去注释后命中数 **0**

## 6. 回滚

- 单元文件：`systemd/backups/` 保留了历史 `.bak-*` 版本，可直接替换回去后 `systemctl daemon-reload`
- 服务：`systemctl disable --now vllm-qwen38.service` 即回到无推理服务状态；功耗/风扇单元独立，可单独停
- 本包**不包含**权重文件本体（19 GB / 1.5 GB），需另行获取；`models/` 只放配置

## 7. 证据与可复现性

`raw/` 下是采集过程的原始证据：

| 路径 | 内容 |
|---|---|
| `raw/C1/` | systemd/脚本/版本/模型配置的拉取脚本与 sha256 校验输出 |
| `raw/C2/` | 监控配置拉取、脱敏自检、监控端点实测原始输出 |
| `raw/C3/` | 主机规格采集脚本与 `nvidia-smi -q` 等命令原始输出 |

校验结果（主 agent 独立复核，不采信子智能体自述）：

| 范围 | 校验方式 | 结果 |
|---|---|---|
| C1 写集（78 个文件） | 逐文件本机字节数 vs 主机 `stat -c %s`，本机 sha256 vs 主机 `sha256sum` | `total=78 bad=0`（`raw/C1/verify.txt`） |
| C2 写集（13 个原样文件） | 同上 | 13/13 一致（`monitoring/checksums.md`） |
| 关键文件交叉复核（22 个） | 主 agent 用 `sha-remote.sh` 重新取主机 sha256 后本地比对 | `MATCH=22 DIFF=0 NOFILE=0`（`raw/C3/sha-remote.txt`） |
| 监控端点 | `curl` 实测 | Prometheus `/-/ready` OK、4 个 scrape job 全 `health=up`、Grafana `/login` 200、`/api/health` `database: ok`（`raw/C2/probe.txt`） |

子智能体报告见仓库根的 `subagent-C1-systemd-scripts.md`、`subagent-C2-monitoring-env.md`、`subagent-C3-inventory.md`。
