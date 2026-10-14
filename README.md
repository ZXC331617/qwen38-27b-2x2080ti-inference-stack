# Qwen3.8-27B × 2×RTX 2080 Ti 推理栈

**简体中文** | [English](README.en.md)

在两张 Turing 老卡（RTX 2080 Ti / SM75，无原生 bf16、无原生 FP8）上用 NVLink 桥接 + TP=2，
把 **Qwen3.8-27B** 的消融量化版跑出 **225K 上下文 / 单流 77–136 tok/s** 的完整方案与实测数据。

> 本仓库包含：可直接复用的 systemd 部署件、监控栈配置、全档位实测数据、以及一份踩过坑的优化结论清单。

---

## 一、硬件

| 项 | 规格 |
|---|---|
| GPU | 2 × NVIDIA GeForce RTX 2080 Ti，22,528 MiB/卡，计算能力 **7.5（SM75）** |
| GPU 互联 | **NVLink NV2**（`nvidia-smi topo -m` 显示 GPU0↔GPU1 = NV2） |
| 驱动 / CUDA | 595.99.02 / CUDA 13.0（V13.0.88） |
| 功耗上限 | 默认 260 W，可调上限 290 W（生产锁定 225 W，见下文） |
| CPU | 36 逻辑核 |
| 内存 | 31 GiB |
| 系统盘 | 468 GB NVMe（生产时剩余约 194 GB） |
| PyTorch | 2.13.0+cu130 |
| 推理引擎 | **vLLM 2080 Ti Definitive Edition**（fork，`FORK_RELEASE=0.2.2-post2`，上游 github.com/weicj） |

**SM75 的三个硬约束**（决定了下面所有取舍）：
1. 没有原生 bf16 → 只能用 fp16 计算；
2. 没有原生 FP8 → FP8 权重无法直接推理（试过，加载即报 `c10::Half != c10::Float8_e4m3fn`）；
3. GDN 线性注意力的融合 CUDA 核要求 compute capability ≥ 8.0 → 自动回落到 Triton 路径。

---

## 二、模型

**目标模型**：`Coder390-HUHUI-INC-A-W4A16`

- 基座：Qwen3.8-27B-Coder390（EfficientThink 系列）的**消融（abliterated）**版
- 量化：W4A16，`compressed-tensors` int4 group-128 对称量化，约 18.6 GB
- 架构：**64 层混合注意力** —— 48 层 GDN 线性注意力 + 16 层全注意力（`full_attention_interval=4`）
  - 这一点很关键：只有 16 层需要读 KV cache，所以**单步延迟随上下文增长很慢**
- 带视觉塔（`Qwen3_5ForConditionalGeneration`），自带 MTP 头 `model_mtp.safetensors`（849 MB）

**投机解码草稿**：`Qwen3.8-27B-DFlash2-W4A16`

- DFlash2 = block-diffusion 草稿（官方来源 `incoai/Qwen3.8-27B-DFlash2`，本项目自行量化为 W4A16）
- 5 层、hidden 5120、**滑窗 2048**、block_size 8、候选选择器 rank 256 / top_k 16
- 从目标模型第 5/19/33/47/61 层取隐藏状态

> ⚠️ 已核实：**所有公开 DFlash2 草稿都是同一份权重**。`incoai/Qwen3.8-27B-DFlash2`、
> `nerkyor/*-DFlash2` 系列仓库里的 `DFlash2-FP8`，manifest 记录的 `source.bf16_sha256`
> 全部是 `67fc76d6…4bab65c`，只是各自重新量化。**换草稿拿不到精度**，详见 [docs/optimization.md](docs/optimization.md)。

---

## 三、实测数据

测速口径：**单流（`--max-num-seqs 1`）**，语料取 deepseek-harness 真实会话报文
（1 条 system + 58 个真实工具结果 + 38 个真实工具定义），采样参数取 harness 真实值
`temperature=0.7 / top_p=0.8 / top_k=20 / repetition_penalty=1.05`，
经本机转发器打到模型主机，与线上同一条链路。

### 3.1 全档位速度

| 上下文档位 | decode 速度 | 每步产出 token | 单步延迟 | TTFT（冷） | 备注 |
|---:|---:|---:|---:|---:|---|
| 32K | **122.6 tok/s** | 4.59 | 36.6 ms | 24.1 s | |
| 96K | **136.2 tok/s** | 4.66 | — | 84.3 s | 该次只生成了 124 token，样本偏小 |
| 160K | **77.2 tok/s** | 3.64 | 45.0 ms | 175.1 s | |

> 全部数据来自 `docs/benchmarks.md` 所列日志，逐条可溯源。

### 3.2 优化前后对照（同条件 A/B）

| 改动 | 档位 | 改前 | 改后 | 变化 |
|---|---:|---:|---:|---:|
| 草稿采样 `greedy` → `probabilistic` | 32K | 113.5 | 122.6 | +8% |
| 同上 | 96K | 89.8 | 136.2 | **+52%** |
| 同上 | 160K | 63.8 | 77.2 | **+21%** |
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` | 32K | 37.8 ms/步 | 36.8 ms/步 | −2.6% |
| 同上 | 160K | 46.4 ms/步 | 45.0 ms/步 | −3.0% |
| 功耗 225 W → 260 W | 32K | 37.5 ms/步 | 37.3 ms/步 | −0.5%（噪声） |

### 3.3 参考值对照

官方 DFlash2 在 **H200 + 官方 Qwen3.8-27B** 上（`temperature=1.0 / top_p=0.95 / top_k=20`）：
接受长度 4.10（MT-Bench）～5.46（GSM8K），单并发吞吐 184–236 tok/s。
本机 32K 档接受长度 4.83、第 0 位命中率 **88.7%**，官方公布的第 0 位命中率是 **85.4%**——
说明短上下文下**草稿与目标的配对是正常的**，差距主要来自硬件代差与长上下文。

---

## 四、优化要点（结论清单）

1. **草稿采样方式是最大单项收益。** 草稿默认 `greedy`（只提 argmax），而线上用 `temperature=0.7` 采样，
   目标分布被拉平后提案大量落空。改成 `draft_sample_method: probabilistic` 后，
   长上下文接受率从 0.28 → 0.38，decode 提升 20%–52%。
2. **`VLLM_MARLIN_USE_ATOMIC_ADD=1` 约 +3%。** vLLM 启动日志会主动提示这一项；
   单流解码正是小 size_n 场景。代价是浮点累加顺序不确定，同 seed 输出不再逐位可复现。
3. **功耗墙几乎没用。** 225 W → 260 W 只值 1–3%，在噪声内；温度却从 69 °C 涨到 79 °C。
   生产最终锁 **225 W**。
4. **加大 prefill chunk 不可行。** `--max-num-batched-tokens` 16384 → 24576 会让峰值激活
   +1.93 GiB、KV 池 −1.93 GiB（6.71 → 4.78 GiB），vLLM 直接拒绝启动：
   `To serve ... max seq len (225280), 4.79 GiB KV cache is needed, which is larger than available 4.66 GiB`。
5. **草稿精度不是瓶颈。** 换成官方 BF16 无损草稿（3.849 GB，sha256 校验通过）后，
   接受率只 +0.4%（32K）/ +1.8%（160K），但草稿变重导致每步慢 4%，净结果更差。
6. **长上下文的接受率下滑是架构决定的**（草稿滑窗只有 2048），不是模型不匹配，
   也不是量化精度问题 —— 逐位数据：第 0 位掉 8 个百分点，第 6 位掉 25 个百分点。
7. **GDN 解码已经在用 Triton**：`Falling back to the Triton GDN decode path: the fused CUDA kernel
   requires ... compute capability 8.0+`。这条提速路已经用尽。
8. **KV 缓存用 fp8**：`--kv-cache-dtype fp8`，16 层全注意力层的 KV 池 6.71 GiB 可容纳
   320,654 token（225K 单流 1.42× 余量）。

完整推导过程与失败记录见 [docs/optimization.md](docs/optimization.md)。

---

## 五、生产配置

`deploy/systemd/vllm-qwen38.service` 的关键参数：

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

环境变量：`VLLM_MARLIN_USE_ATOMIC_ADD=1`、`VLLM_KV_BUDGET_FROM_FREE_MEMORY=1`、
`VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1`、`VLLM_DISABLE_TILELANG=1` 等，逐条见 `deploy/env/`。

---

## 六、目录结构

```
README.md                中文说明（默认）
README.en.md             英文说明
docs/
  benchmarks.md          全档位实测数据（中文）
  optimization.md        优化过程与结论（中文）
  deployment.md          从零部署步骤（中文）
deploy/
  systemd/               systemd 单元原文
  scripts/               功耗墙 / 风扇曲线等脚本
  monitoring/            Prometheus / Grafana / exporter 配置
  env/                   环境变量清单（脱敏）
  inventory.md           主机完整清单
```

---

## 七、许可

代码与配置：Apache-2.0（见 [LICENSE](LICENSE)）。
模型权重与第三方组件的许可请遵循其各自仓库。
