# 优化过程与结论（Qwen3.8-27B × 2×RTX 2080 Ti）

**简体中文** | [English](optimization.en.md)

本文记录这台机器上为了「单流最大速度」做过的**全部**改动、实测数据与失败记录。
目标只有一个：**单流最快**，完全不需要并发（`--max-num-seqs 1`）。

测速口径统一为：

- 语料 = deepseek-harness 真实会话报文（1 条 system + 58 个真实工具结果 + 38 个真实工具定义）
- 采样参数 = harness 真实值 `temperature=0.7 / top_p=0.8 / top_k=20 / repetition_penalty=1.05`
- 链路 = 本机转发器 → 模型主机，与线上完全同一条
- `gen=256`，每档独立一次请求

> 逐条数据与日志文件见 [benchmarks.md](benchmarks.md)。

---

## 一、先说结论

按收益从大到小：

| # | 改动 | 收益 | 代价 | 是否保留 |
|---|---|---|---|---|
| 1 | 草稿采样 `greedy` → **`probabilistic`** | 长上下文 decode **+21% ~ +52%** | 无 | ✅ 保留 |
| 2 | `VLLM_MARLIN_USE_ATOMIC_ADD=1` | 单步延迟 **−2.6% ~ −3.0%** | 同 seed 输出不再逐位可复现 | ✅ 保留 |
| 3 | `--max-num-batched-tokens` 16384 → 24576 | 预期 prefill 提速 | **vLLM 直接拒绝启动** | ❌ 回滚 |
| 4 | 草稿换官方 BF16 无损版 | 接受率 +0.4% ~ +1.8%（噪声级） | 每步慢 4%、KV 池 −18% | ❌ 回滚 |
| 5 | 功耗墙 225 W → 260 W | 单步 −0.5%（噪声内） | 温度 69 → 79 °C | ❌ 生产锁 225 W |

一句话：**这台机器上真正值钱的是「草稿采样方式」和「Marlin atomic add」这两项，
其余都是噪声或负收益。**

---

## 二、硬件约束：SM75 决定了所有取舍

RTX 2080 Ti 是 Turing，计算能力 **7.5（SM75）**，三条硬约束贯穿全篇：

1. **没有原生 bf16 张量核** → 只能用 fp16 计算（`--dtype half`）。
2. **没有原生 FP8** → FP8 权重无法直接推理。实测把 NVFP4 / FP8 权重加载进来会直接报
   `c10::Half != c10::Float8_e4m3fn`。所以权重只能走 **int4 W4A16**（compressed-tensors），
   KV cache 的 fp8 是 vLLM 软件模拟的存储压缩，和「FP8 张量核」不是一回事。
3. **GDN 融合 CUDA 核要求 compute capability ≥ 8.0** → 自动回落 Triton：
   ```
   Falling back to the Triton GDN decode path: the fused CUDA kernel requires ... compute capability 8.0+
   ```
   也就是说 `VLLM_GDN_DECODE_KERNEL` 这条提速路在 SM75 上**已经用尽**，没有可开的开关。

两张卡用 **NVLink（`nvidia-smi topo -m` 显示 GPU0↔GPU1 = NV2）** 桥接，`--tensor-parallel-size 2`。

---

## 三、模型架构：为什么长上下文衰减慢，但接受率会掉

主模型 `Coder390-HUHUI-INC-A-W4A16` 是 **64 层混合注意力**：

- **48 层 GDN 线性注意力**（gated delta net）+ **16 层全注意力**（`full_attention_interval=4`）
- 只有 16 层需要读 KV cache → **单步延迟随上下文增长很慢**
- KV cache 用 fp8：16 层全注意力的 KV 池 6.71 GiB 可容纳 **320,654 token**（225K 单流 1.42× 余量）

这解释了一个反直觉现象：**长上下文变慢的主因不是单步延迟，而是投机接受率塌陷。**

| 档位 | 单步延迟 | 每步产出 token | decode |
|---:|---:|---:|---:|
| 32K | 37.5 ms | 4.27 | 113.5 t/s |
| 160K | 46.5 ms | 2.98 | 63.8 t/s |

单步只涨了 24%，每步产出却掉了 30%。所以优化的**靶心是接受率，不是算力**。

---

## 四、优化项 1：草稿采样 `greedy` → `probabilistic`（最大收益）

### 问题

DFlash2 草稿默认按 `greedy` 出提案（只提 argmax），而线上目标侧用 `temperature=0.7` 采样。
目标分布被温度拉平之后，贪心提案的命中率大幅下滑 —— 上下文越长越明显。

### 改动

在 `--speculative-config` 里加一个字段：

```json
{"method":"dflash","num_speculative_tokens":7,
 "model":"/root/models/Qwen3.8-27B-DFlash2-W4A16",
 "attention_backend":"TRITON_ATTN","kv_cache_dtype":"float16",
 "draft_sample_method":"probabilistic"}
```

### 实测（同档位同语料，`gen=256`）

| 档位 | 改前 decode | 改后 decode | 变化 | 改前接受率 | 改后接受率 |
|---:|---:|---:|---:|---:|---:|
| 32K | 113.47 | **122.58** | +8% | 0.467 | 0.513 |
| 96K | 89.82 | **136.18** | **+52%** | 0.395 | 0.523 |
| 160K | 63.80 | **77.22** | **+21%** | 0.282 | 0.377 |

长上下文收益最大，因为贪心提案在长上下文下退化最严重。

### 佐证：反过来测「全贪心」也能说明问题

把目标侧也改成贪心（`BENCH_TEMP=0`）后，160K 档 decode 从 64.2 → 98.7 t/s（+54%），
接受率 0.28 → 0.48。**同一份草稿，只改采样一致性，就能拿回一半以上的损失** —— 这坐实了
「采样方式不匹配」是主因。

> 生产环境不能改目标采样（用户要 `temperature=0.7` 的多样性），所以正解是**把草稿也改成概率采样**。

---

## 五、优化项 2：`VLLM_MARLIN_USE_ATOMIC_ADD=1`（约 +3%）

### 来源

vLLM 启动日志主动提示：

```
Marlin kernel can achieve better performance for small size_n with experimental use_atomic_add feature.
```

**单流解码正是小 `size_n` 场景**，所以这一项天然对口。

### 改动

unit 里加一行环境变量，重启即可 —— **不占显存、不动上下文**：

```
Environment=VLLM_MARLIN_USE_ATOMIC_ADD=1
```

生效确认：设置后上面那条提示会从启动日志里消失（`marlin_utils.py` 只在未开启时打印）。

### 对照方法（重要）

直接比 decode t/s 会被采样随机性干扰（同档位接受率在 0.28–0.55 间波动）。
因此改用**与接受率无关**的指标：

```
单步延迟 ms/step = (total_s − ttft_s) × 1000 / drafts
```

另外主机上 `wb-prefix-warm.timer` 会周期性重放前缀请求、把投机计数器搅脏
（表现为「接受 token 数 > 实际生成 token 数」）。测速期间必须停掉它，
只有计数器自洽（`drafts × mean_len ≈ 256`）的数据才计入。

### 实测（225 W / probabilistic / 固定 seed=1234）

| 档位 | 关闭 | 开启 | 单步延迟变化 | 等接受率下 decode 变化 |
|---|---:|---:|---:|---:|
| 32K | 37.8 ms/步 | **36.8 ms/步** | −2.6% | +2.7% |
| 160K | 46.4 ms/步 | **45.0 ms/步** | −3.0% | +3.1% |

两档方向一致、幅度一致 → 判定为真实收益（约 +3%）。

### 副作用

atomic add 的浮点累加顺序不确定，同一 seed 下输出不再逐位可复现
（实测同 seed 接受率从 4.83 变到 4.197）。对 `temperature=0.7` 的对话场景无影响。

回滚：`bash /tmp/revert-marlin-eadd.sh`
备份：`/etc/systemd/system/vllm-qwen38.service.bak-marlinadd-20261014-185244`

---

## 六、失败记录 A：`--max-num-batched-tokens 16384 → 24576`

**结论：直接起不来，vLLM 拒绝启动，已回滚。**

启动报错：

```
ValueError: To serve at least one request with the model's max seq len (225280),
4.79 GiB KV cache is needed, which is larger than the available KV cache memory (4.66 GiB).
Based on the available memory, the estimated maximum model length is 215888.
```

实测剖面变化（单卡）：

| 项 | 16384（现行） | 24576（试跑） | 差值 |
|---|---:|---:|---:|
| 权重 + 非 torch | 11.17 GiB | 11.17 GiB | 0 |
| 峰值激活 | 3.08 GiB | ≈ 5.0 GiB | **+1.93 GiB** |
| KV 池 | 6.71 GiB | 4.78 GiB | **−1.93 GiB** |
| KV token 容量 | 320,654 | ≈ 234,000 | −27% |
| 225,280 单流是否可服务 | 可以（1.42× 余量） | **不可以** | — |

附带成本：

- 峰值激活与 KV 池是**零和**关系（`VLLM_KV_BUDGET_FROM_FREE_MEMORY=1`，KV 按剩余显存分配），
  chunk 每加 1 token 约吃掉 0.24 MiB KV。
- 改 chunk 会改变 `torch.compile` 的动态形状区间（1,16384）→（1,24576），
  首次启动触发完整重编译（实测 41 s，之后走缓存）。
- 反推可容纳 225,280 的 chunk 上限约 **24,100** —— 24576 只超了约 0.13 GiB，属「贴线」，
  余量为零，后续任何改动都会再撑爆。

**若确实要上 24576**，必须同时把 `--max-model-len` 从 225,280 降到约 212,992
（转发器 guard 相应变为 212,992 − 24,576 = 188,416，仍高于 180,224 的压缩阈值），
即用约 12K 上下文换 prefill 速度。本次未采用。

回滚：`bash /tmp/revert-batchtokens.sh`（已执行）
备份：`/etc/systemd/system/vllm-qwen38.service.bak-bt24576-20261014-181412`

---

## 七、失败记录 B：草稿换官方 BF16 无损版（接受率几乎不动）

### 先证明：所有公开 DFlash2 草稿都是同一份权重

用 sha256 链条证明（不是推测）：

| 来源 | 文件 | sha256 |
|---|---|---|
| `incoai/Qwen3.8-27B-DFlash2`（官方） | model.safetensors 3.849 GB | `67fc76d6…4bab65c` |
| `nerkyor/Qwen3.8-27B-Coder390-…-MTP-DFlash2` → `BF16/DFlash2-FP8` | 2.407 GB | `a7b48bc2…cbfb93`，manifest 记 `source.bf16_sha256 = 67fc76d6…` |
| `nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-…-DFlash2` → `BF16/DFlash2-FP8` | 2.407 GB | `1f3636a3…fa25d1`，同样记 `source.bf16_sha256 = 67fc76d6…` |

即：所谓「Coder390 专用草稿」「Uncensored 专用草稿」都只是**官方草稿的重新量化版本**，
底层权重完全相同。**换草稿拿不到精度。**

### 再实测：BF16 草稿 vs 当前 W4A16 草稿

把官方 bf16 草稿（3.849 GB，sha256 `67fc76d6…` 校验通过）下到主机，切换后 A/B：

| 档位 | 指标 | W4A16 草稿 | BF16 草稿 | 变化 |
|---|---|---:|---:|---:|
| 32K | 接受率（mean_len） | 4.830 | 4.849 | **+0.4%** |
| 32K | 单步延迟 | 37.8 ms | 39.5 ms | +4.5% |
| 32K | decode | 127.44 t/s | 121.73 t/s | **−4.5%** |
| 160K | 接受率（mean_len） | 3.493 | 3.556 | **+1.8%** |
| 160K | 单步延迟 | 46.4 ms | 48.2 ms | +3.9% |
| 160K | decode | 75.34 t/s | 73.52 t/s | **−2.4%** |

**接受率几乎不动（噪声量级），但草稿变重导致每步慢 4%，净结果更差。**

显存代价：权重+非 torch 11.17 → 12.40 GiB（+1.23 GiB/卡），KV 池 6.71 → 5.49 GiB
（320,654 → 252,313 token），被迫把 `max-model-len` 从 225,280 降到 196,608。

→ **已回滚**，草稿仍是 `Qwen3.8-27B-DFlash2-W4A16`，`max-model-len` 回到 225,280。

### 真正的瓶颈：草稿与目标模型不同源

- 草稿是从 **base `Qwen/Qwen3.8-27B`** 蒸馏的；
- 我们的目标是 **Coder390-HUHUI-INC-A**（消融 + 重度微调版）；
- 两者分布有系统性偏移，这才是长上下文接受率只有 0.37 的根因。

参考值：fork 官方用「官方目标 + 官方草稿」同源配对时，32K decode 能到 209–220 t/s；
以我们 37 ms 的单步延迟反推，对方每步产出要接近 7.7 token（上限 8）—— 即接近满接受率。
**差距来自配对同源性，而非量化精度。**

> 唯一能真正提升接受率的路径：用当前目标模型**自己蒸馏**一个 DFlash2 草稿（本地训练，非下载可得）。

---

## 八、失败记录 C：功耗墙 225 W → 260 W

| 档位 | 225 W | 260 W | 变化 |
|---|---:|---:|---:|
| 32K | 37.5 ms/步 | 37.3 ms/步 | −0.5% |
| 160K | 46.5 ms/步 | 45.2 ms/步 | −2.8% |

**基本在噪声内。** 代价：加载态温度从 69/57 °C 涨到 80/70 °C，风扇 80%/46%。
生产最终**锁 225 W**（温度压不住，且收益不真实）。

---

## 九、已经用尽 / 无需再试的项

| 项 | 状态 |
|---|---|
| GDN 解码走 CUDA 融合核 | ❌ SM75 不支持，已回落 Triton |
| FP8 权重直接推理 | ❌ SM75 无 FP8 张量核，加载即报错 |
| bf16 权重 | ❌ SM75 无原生 bf16，只能 fp16 |
| 换草稿提升精度 | ❌ 所有公开草稿同源，换不出精度 |
| 加大 prefill chunk | ❌ 挤爆 KV 池，起不来 |
| 提高功耗墙 | ❌ 收益在噪声内 |
| KV cache fp8 | ✅ 已在用（320,654 token） |
| 前缀缓存 | ✅ 已启用 |
| Marlin atomic add | ✅ 已在用 |
| probabilistic 草稿采样 | ✅ 已在用（最大收益项） |

---

## 十、复现命令

```bash
# 1) 停掉会搅脏投机计数器 / 占用带宽的定时器
echo ubuntu | sudo -S systemctl stop wb-prefix-warm.timer

# 2) 跑全档位扫描（在测速机上，BENCH_TEMP 取线上真实温度 0.7）
BENCH_TEMP=0.7 pwsh -NoProfile -ExecutionPolicy Bypass \
  -File "D:\1\靶场\全层消融\run-bench.ps1" \
  -Tag final-prod -TiersCsv "8192,32768,65536,98304,131072,163840,196608"

# 3) 测完恢复定时器
echo ubuntu | sudo -S systemctl start wb-prefix-warm.timer
```

回滚脚本与备份文件位置见上文字节。生产单元原文见
[`../deploy/systemd/vllm-qwen38.service`](../deploy/systemd/vllm-qwen38.service)。
