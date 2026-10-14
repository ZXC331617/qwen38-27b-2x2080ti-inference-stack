# 内部事实底稿（中英双语文档共用，不作为交付物）

本文件是 benchmarks / optimization / deployment 三份文档（中英各一）的唯一数据源。
**所有数字必须来自本底稿或底稿标注的原始文件；未实测的项写「未实测 / not measured」，禁止推测。**
中文为默认语言；英文文档必须与中文标题结构一一对应。

---

## 0. 交付物与结构

```
docs/benchmarks.md        docs/benchmarks.en.md
docs/optimization.md      docs/optimization.en.md
docs/deployment.md        docs/deployment.en.md
```

固定章节（中英一一对应）：

- benchmarks：1 测试环境与口径 / 2 关键结论速览 / 3 草稿采样策略 greedy→probabilistic /
  4 单步延迟与 TTFT（prefill）/ 5 功耗墙 225W vs 260W / 6 VLLM_MARLIN_USE_ATOMIC_ADD /
  7 草稿精度 W4A16 vs BF16 / 8 max-num-batched-tokens 24576 显存剖面 /
  9 长上下文尾部与失败档位 / 10 早期全档位扫描（探索性）/ 附录 A 原始日志索引 / 附录 B 其他语料口径
- optimization：1 结论清单 / 2 根因分析 / 3 生效改动与回滚 / 4 已实测失败并回滚 /
  5 尚未动用的候选 / 6 与官方参考口径对比 / 7 未实测与待验证
- deployment：1 范围 / 2 硬件与软件基线 / 3 权重与 sha256 校验 / 4 vLLM 启动配置 /
  5 转发器与上下文守卫 / 6 测速复现 / 7 运维（功耗墙、保温定时器）/ 8 校验清单 / 9 未实测项

---

## 1. 测试环境与口径（来源：REPORT-单流提速-20261014.md 首段；bench-harness-real.py）

- 主机：`192.168.5.88`（内网实验网段），Linux + systemd
- GPU：2× RTX 2080 Ti，NVLink NV2 桥接，张量并行 TP=2
- 计算能力 SM75（compute capability 7.5）→ 无原生 bf16 / fp8 支持；CUDA 13.0
- vLLM fork：`/root/vLLM-2080Ti-Definitive-main`（含 `.venv`）
- 目标模型：`/root/models/Coder390-HUHUI-INC-A-W4A16`（W4A16 量化，约 19 GB）
- 草稿模型：`/root/models/Qwen3.8-27B-DFlash2-W4A16`
- `max_model_len = 225280`；输出预留 `24576`
- 转发器：本机 `127.0.0.1:8788`；主机指标端点 `http://192.168.5.88:8000/metrics`
- 采样参数（deepseek harness 真实值，已由 req-dump 实测确认）：
  `temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05`
- 测速语料：`D:\1\靶场\codex模型管理\req-dump\fail-2026-10-14T05-44-25-018Z-400-qwen-vllm-main-qwen3.8-27b.json`
  （真实会话：1 条 system = 28238 字符 + 38 个工具定义 + 58 个填充块，含 1 个 >50KB 大块）
- 统一生成长度 `gen = 256`；请求走转发器，与 harness 同链路
- 指标口径：
  - `decode_t/s = (completion_tokens − 1) / (total_s − ttft_s)`
  - `acc_rate = accepted_tokens / draft_tokens`（来自 vLLM `spec_decode` 指标）
  - `mean_len = acc_rate 口径的每步接受 token 数 + 1`（脚本内 `acc/drafts + 1`）
  - `单步延迟 ms/step = (total_s − ttft_s) × 1000 / drafts`（与接受率无关，用于 A/B 对照）

---

## 2. 表 1 — 草稿采样策略 greedy → probabilistic（decode t/s 与接受率）

改前 = `draft_sample_method: greedy`（225W，probabilistic 之前）；
改后 = `draft_sample_method: probabilistic`。

| 档位 | prompt_tok | 改前 decode t/s | 改后 decode t/s | 变化 |
|---|---:|---:|---:|---:|
| 32K | 35,672 | 113.47 | 122.58 | +8% |
| 96K | 95,953 | 89.82 | 136.18 | +52% |
| 160K | 160,703 | 63.80 | 77.22 | +21% |
| 192K | 197,218 | — | 被 CTX_GUARD 拦（设计内） | — |

接受率与每步产出：

| 档位 | 改前 acc_rate | 改后 acc_rate | 改前 每步产出 | 改后 每步产出 |
|---|---:|---:|---:|---:|
| 32K | 0.467 | 0.513 | 4.27 tok | 4.59 tok |
| 96K | 0.395 | 0.523 | 3.77 tok | 4.66 tok |
| 160K | 0.282 | 0.377 | 2.98 tok | 3.64 tok |

来源：REPORT 第二节；原始日志 `logs/bench-225w.log`（改前，32K/96K/160K 行 acc 0.4667/0.395/0.2824）、
`logs/bench-prob.log`（改后 32K/160K）、`logs/bench-prob-t2.log`（改后 96K）。

日志原文行（供核对）：
```
bench-225w.log  32768 35672 0 256 256 24.912 27.16 1431.9 113.47 60 0.4667 4.267
bench-225w.log  98304 95953 0 256 256 86.451 89.29 1109.9  89.82 68 0.395  3.765
bench-225w.log 163840 160703 93936 256 256 96.152 100.149 694.4 63.8 86 0.2824 2.977
bench-prob.log  32768 35672 0 256 256 24.107 26.187 1479.7 122.58 56 0.5128 4.589
bench-prob.log 163840 160703 0 256 256 175.106 178.408 917.7 77.22 97 0.377 3.639
bench-prob-t2.log 98304 95953 0 124 60 84.261 85.164 1138.8 136.18 44 0.5227 4.659
```

---

## 3. 表 2 — 单步延迟与 TTFT（prefill，无缓存）

| 档位 | TTFT | prefill t/s |
|---|---:|---:|
| 32K | 24.1 s | 1,479 |
| 96K | 84.3 s | 1,139 |
| 160K | 175.1 s | 918 |

来源：REPORT 第四节；`logs/bench-prob.log`（32K 24.107/1479.7；160K 175.106/917.7）、
`logs/bench-prob-t2.log`（96K 84.261/1138.8）。

单步延迟（REPORT 第三节，改前 → 改后）：32K 37.5 → 37.3 ms；96K 41.7 → 40.1 ms；160K 46.5 → 45.2 ms。

参考口径（REPORT 第四节）：fork 官方 2×2080Ti 32K prefill ≈ 1265 t/s。

---

## 4. 表 3 — 功耗墙 225W vs 260W

REPORT 第三节结论：功耗 225W→260W 只值 1–3%，基本在噪声内。
单步延迟（REPORT 第三节）：37.5→37.3 / 41.7→40.1 / 46.5→45.2 ms。
代价：加载态温度 80/70 °C、风扇 80%/46%（225W 时为 69/57 °C）。

原始日志（口径不同，注意 260W 那次带前缀缓存，TTFT 不可比）：

| 档位 | 225W decode t/s | 260W decode t/s | 225W TTFT_s | 260W TTFT_s |
|---|---:|---:|---:|---:|
| 32K | 113.47 | 126.45 | 24.912 | 2.394（cached=32,960） |
| 96K | 89.82 | 118.08（gen=129/reason=65，非 256 满档） | 86.451 | 2.916（cached=93,936） |
| 160K | 63.80 | 64.15 | 96.152（cached=93,936） | 4.745（cached=158,208） |

来源：`logs/bench-225w.log`、`logs/bench-260w.log`。
说明：260W 组 96K 行 `gen=129 reason=65`、32K/96K 行 `cached` 非 0，与 225W 组不同口径，
故以 REPORT 的 ms/step 口径为准。

---

## 5. 表 4 — VLLM_MARLIN_USE_ATOMIC_ADD=1

来源：启动日志提示
`Marlin kernel can achieve better performance for small size_n with experimental use_atomic_add feature.`
生效确认：设置后该提示从日志消失（`marlin_utils.py:612` 仅在未开启时打印）。
对照口径：`ms/step`（与接受率无关）。测试条件：225W / probabilistic 草稿 / 固定 `seed=1234` / 已停保温定时器。

| 档位 | 关闭 | 开启 | 单步延迟变化 | 等接受率下 decode 变化 |
|---|---:|---:|---:|---:|
| 32K | 37.8 ms/步 | 36.8 ms/步 | −2.6% | +2.7% |
| 160K | 46.4 ms/步 | 45.0 ms/步 | −3.0% | +3.1% |

副作用：atomic add 浮点累加顺序不确定，同 seed 输出不再逐位可复现
（实测同 seed 接受率从 4.83 变到 4.197）。对 temperature=0.7 的对话场景无影响。
来源：REPORT 第八节；原始日志 `logs/bench-base225.log`（关闭）、`logs/bench-marlineadd.log` +
`logs/bench-marlineadd-t2.log`（开启）、`logs/bench-ea-clean.log`（开启 + 停保温定时器的干净复跑）。

日志原文行（供核对）：
```
bench-base225.log    32768 35672 0 256 256 24.805 26.786 1438.1 128.74 53 0.5472 4.83
bench-base225.log   163840 160703 0 256 256 178.92 181.97 898.2 83.6 66 0.4156 3.909
bench-marlineadd.log 32768 35672 0 256 256 25.377 27.429 1405.7 124.29 56 0.523 4.661
bench-marlineadd.log 163840 160703 0 256 256 181.894 185.239 883.5 76.24 97 0.3726 3.608
bench-marlineadd-t2.log 163840 160703 158208 256 256 49.341 53.528 50.6 60.9 120 0.2845 2.992
bench-ea-clean.log   32768 35672 9888 256 256 21.845 24.089 1180.3 113.62 61 0.4567 4.197
bench-ea-clean.log  163840 160703 158208 256 256 4.869 8.156 512.5 77.58 73 0.3581 3.507
```

---

## 6. 表 5 — 草稿精度：W4A16 vs BF16（无损草稿）

条件：225W / probabilistic / 固定 `seed=1234` / 已停保温定时器。

| 档位 | 指标 | W4A16 草稿 | BF16 草稿 | 变化 |
|---|---|---:|---:|---:|
| 32K | mean_len（接受率） | 4.830 | 4.849 | +0.4% |
| 32K | 单步延迟 | 37.8 ms | 39.5 ms | +4.5% |
| 32K | decode | 127.44 t/s | 121.73 t/s | −4.5% |
| 160K | mean_len（接受率） | 3.493 | 3.556 | +1.8% |
| 160K | 单步延迟 | 46.4 ms | 48.2 ms | +3.9% |
| 160K | decode | 75.34 t/s | 73.52 t/s | −2.4% |

显存代价：权重+非torch 11.17 → 12.40 GiB（+1.23 GiB/卡）；KV 池 6.71 → 5.49 GiB
（320,654 → 252,313 token）；被迫把 max-model-len 从 225,280 降到 196,608。
结论：**已回滚**，草稿仍为 `Qwen3.8-27B-DFlash2-W4A16`，max-model-len 回到 225,280，KV 池 6.71 GiB。
来源：REPORT 第九节；`logs/bench-base-clean.log`（W4A16）、`logs/bench-draftbf16.log`（BF16）。

日志原文行（供核对）：
```
bench-base-clean.log 32768 35672 0 256 256 25.503 27.504 1398.7 127.44 53 0.5472 4.83
bench-base-clean.log 163840 160703 0 256 256 181.024 184.409 887.7 75.34 73 0.3562 3.493
bench-draftbf16.log  32768 35672 0 256 256 37.695 39.789 946.3 121.73 53 0.5499 4.849
bench-draftbf16.log 163840 160703 0 256 256 180.297 183.766 891.3 73.52 72 0.3651 3.556
```

---

## 7. 表 6 — max-num-batched-tokens 16384 → 24576（单卡显存剖面）

结论：直接起不来，vLLM 拒绝启动，已回滚。
启动报错原文：
```
ValueError: To serve at least one request with the model's max seq len (225280),
4.79 GiB KV cache is needed, which is larger than the available KV cache memory (4.66 GiB).
Based on the available memory, the estimated maximum model length is 215888.
```

| 项 | 16384（现行） | 24576（试跑） | 差值 |
|---|---:|---:|---:|
| 权重 + 非 torch | 11.17 GiB | 11.17 GiB | 0 |
| 峰值激活 | 3.08 GiB | ≈ 5.0 GiB | +1.93 GiB |
| KV 池 | 6.71 GiB | 4.78 GiB | −1.93 GiB |
| KV token 容量 | 320,654 | ≈ 234,000 | −27% |
| 视觉 encoder cache 预算 | 16,384 | 24,576 | +50% |
| 225,280 单流是否可服务 | 可以（1.42× 余量） | 不可以 | — |

附带成本：
- 峰值激活与 KV 池零和（`VLLM_KV_BUDGET_FROM_FREE_MEMORY=1`，KV 按剩余显存分配），
  chunk 每加 1 token 约吃掉 0.24 MiB KV。
- 改 chunk 会改变 torch.compile 动态形状区间（1,16384）→（1,24576），首次启动触发完整重编译（实测 41 s，之后走缓存）。
- 反推可容纳 225,280 的 chunk 上限约 24,100；24,576 只超约 0.13 GiB，属「贴线」，余量为零。

若确实要上 24576，必须同时把 `--max-model-len` 从 225,280 降到约 212,992
（转发器 guard 相应变为 212,992 − 24,576 = 188,416，仍高于 180,224 压缩阈值），
即用约 12K 上下文换 prefill 速度。本次未采用。
来源：REPORT 第七节（试跑时间 2026-10-14 18:14）。
回滚脚本 `/tmp/revert-batchtokens.sh`（已执行）；备份
`/etc/systemd/system/vllm-qwen38.service.bak-bt24576-20261014-181412`。

---

## 8. 表 7 — 长上下文尾部与失败档位

来自 `logs/bench-tail2.log`：

| 档位 | prompt_tok | cached | TTFT_s | decode t/s | acc_rate | mean_len | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| 215000 | 211,742 | 194,464 | 34.551 | 58.65 | 0.2759 | 2.931 | 成功 |
| 224000 | — | — | — | — | — | — | FAILED: HTTP 400 |

192K 档 400 的原因（REPORT 第三节第 5 条）：转发器 `CTX_GUARD prompt ~203707 > limit 200704`。
`200704 = 225280（max_model_len）− 24576（输出预留）`，属设计内拦截；
实际可用 prompt 上限约 193K，高于 180,224 压缩阈值，不影响正常使用。
`logs/bench-harness-real.log` 的 229,376 档亦为 `FAILED: HTTPError: HTTP Error 400: Bad Request`。

---

## 9. 表 8 — 早期全档位扫描（探索性，含前缀缓存，非最终口径）

来源：`logs/bench-harness-real.log`（该次运行各档 `cached` 非 0，配置为探索期状态）。

| 档位 | prompt_tok | cached | TTFT_s | total_s | prefill t/s | decode t/s | acc_rate | mean_len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 8,247 | 6,592 | 1.182 | 4.212 | 1,399.7 | 84.18 | 0.2908 | 3.035 |
| 8192 | 8,247 | 6,592 | 1.135 | 3.665 | 1,458.8 | 100.75 | 0.3722 | 3.606 |
| 32768 | 35,672 | 6,592 | 21.219 | 23.594 | 1,370.5 | 107.35 | 0.4354 | 4.048 |
| 65536 | 65,102 | 6,592 | 48.395 | 50.855 | 1,209.0 | 103.65 | 0.4447 | 4.113 |
| 98304 | 95,953 | 62,624 | 37.608 | 40.803 | 886.2 | 79.81 | 0.3365 | 3.355 |
| 131072 | 130,249 | 93,936 | 62.425 | 65.569 | 581.7 | 81.1 | 0.3763 | 3.634 |
| 163840 | 160,703 | 128,544 | 50.866 | 55.403 | 632.2 | 56.21 | 0.2356 | 2.649 |
| 196608 | 197,218 | 158,208 | 70.695 | 74.323 | 551.8 | 70.3 | 0.3494 | 3.446 |
| 229376 | FAILED: HTTPError: HTTP Error 400: Bad Request | | | | | | | |

---

## 10. 诊断用对照：目标侧改贪心采样

REPORT 第三节第 2 条：160K 档把**目标侧**改成贪心采样后，decode 从 64.2 → 98.7 t/s（+54%），
接受率 0.28 → 0.48。来源 `logs/bench-greedy.log`：
```
bench-greedy.log 32768 35672 32960 256 256 2.424 4.51 1118.9 122.25 57 0.5088 4.561
bench-greedy.log 163840 160703 158208 256 256 4.652 7.236 536.4 98.68 59 0.4818 4.373
```
该对照说明：目标分布被 temperature 拉平是接受率塌陷的直接原因（见 optimization 第 2 节）。

---

## 11. sha256 链条 — 所有公开 DFlash2 草稿同源（REPORT 第九节结论一）

| 来源 | 文件 | sha256 |
|---|---|---|
| `incoai/Qwen3.8-27B-DFlash2`（官方） | model.safetensors 3.849 GB | `67fc76d6…4bab65c` |
| `nerkyor/Qwen3.8-27B-Coder390-…-MTP-DFlash2` → `BF16/DFlash2-FP8` | model.safetensors 2.407 GB | `a7b48bc2…cbfb93`，manifest 记 `source.bf16_sha256 = 67fc76d6…` |
| `nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-…-DFlash2` → `BF16/DFlash2-FP8` | model.safetensors 2.407 GB | `1f3636a3…fa25d1`，同样记 `source.bf16_sha256 = 67fc76d6…` |

完整值（来源 `logs/verify.out` 的 DFlash2-FP8 SHA256SUMS 段）：
`a7b48bc293a989494ce2db64ac01a0261c370499641b0f5a668d8005c8cbfb93  model.safetensors`
含义：所谓「Coder390 专用草稿」「Uncensored 专用草稿」都只是官方草稿的重新量化版本，底层权重完全相同。

目标模型自带 MTP 头（REPORT 第九节末尾）：
`/root/models/Coder390-HUHUI-INC-A-W4A16/model_mtp.safetensors`（849 MB）sha256
`1d8268aa…`，与 `huihui-ai/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated` 的
`mtp-bf16.safetensors` sha256 `90fa0e3e…` 不同（本地那份是重新导出/量化过的）。

---

## 12. 根因结论（REPORT 第三节 + 第九节结论三）

1. 长上下文变慢的主因是**接受率塌陷**，不是单步延迟：单步延迟只从 37.5 ms（32K）涨到 46.5 ms（160K，+24%），
   而每步产出从 4.27 掉到 2.98（−30%）。
2. 温度采样 vs 贪心草稿是接受率塌陷的根源：草稿侧原为 `greedy`（只提 argmax），
   目标侧 harness 用 `temperature=0.7`，目标实际分布变平 → 提案命中率大跌。
   改 `probabilistic` 后接受率回到 0.377，拿回约 2/3 的损失。
3. 真正的瓶颈是「草稿与目标模型不匹配」，不是精度：
   - 草稿从 base `Qwen/Qwen3.8-27B` 蒸馏；
   - 目标是我们自己消融+重度微调的 `Coder390-HUHUI-INC-A`；
   - 两者分布有系统性偏移，这是长上下文接受率只有 0.37 的根因。
   - 参考值：fork 官方用「官方目标 + 官方草稿」同源配对时，32K decode 能到 209–220 t/s；
     以我们 37 ms 的单步延迟反推，对方每步产出要接近 7.7 token（上限 8）——即接近满接受率。
   - 唯一能真正提升接受率的路径：用当前目标模型自己蒸馏一个 DFlash2 草稿（本地训练，非下载可得）。
4. 主机侧架构（REPORT 第三节第 4 条）：Qwen3.5 混合模型 64 层 = 48 层 GDN 线性注意力 + 16 层全注意力，
   所以单步延迟随上下文增长很慢；attention 后端 FLASHINFER，`V2 Model Runner` 已启用，fp8 KV，
   DFlash2 草稿为 5 层 / 滑窗 2048 的块式草稿（block_size=8，单次前向出块）。
5. GDN 解码已跑在 triton，不是 cuda：日志写
   `Falling back to the Triton GDN decode path: the fused CUDA kernel requires … compute capability 8.0+`。
   即 `VLLM_GDN_DECODE_KERNEL` 这条提速路已经用尽。

---

## 13. 生效改动与回滚（REPORT 第一/二/八/九节；change-*.sh / revert-*.sh）

| # | 改动 | 位置 | 是否保留 | 备份 |
|---|---|---|---|---|
| 1 | GPU 功耗墙 225W → 260W | `/etc/systemd/system/gpu-power-limit.service` | 试过，收益 1–3%（噪声） | `set-power.sh` 生成 `/root/gpu-power-limit.service.bak-<ts>` |
| 2 | 草稿采样 `greedy` → `probabilistic` | `vllm-qwen38.service` 的 `--speculative-config` | **保留（主收益）** | `…bak-draftprob-20261014-180407`；回滚 `bash /tmp/revert-draftprob.sh` |
| 3 | `VLLM_MARLIN_USE_ATOMIC_ADD=1` | `vllm-qwen38.service` 环境变量 | **保留（约 +3%）** | `…bak-marlinadd-20261014-185244`；回滚 `bash /tmp/revert-marlin-eadd.sh` |
| 4 | `--max-num-batched-tokens 16384 → 24576` | 同 unit | **已回滚（起不来）** | `…bak-bt24576-20261014-181412`；回滚 `/tmp/revert-batchtokens.sh` |
| 5 | 草稿换 BF16（无损） | `--speculative-config` 的 model + `--max-model-len 196608` | **已回滚（净更差）** | `…bak-draftbf16-20261014-191433` |

改动脚本（本机副本）：`change-draftprob.sh`、`change-marlin-eadd.sh`、`change-batchtokens.sh`、
`change-ctx220-fp8.sh`、`change-draft-bf16.sh`、`set-power.sh`、`revert-draftprob.sh`、
`revert-marlin-eadd.sh`、`revert-batchtokens.sh`、`revert-draft-bf16.sh`、`revert-draft-w4a16.sh`、`toggle-timers.sh`。

---

## 14. 尚未动用的候选（REPORT 第五节）

1. ~~`--max-num-batched-tokens 16384 → 24576`~~ — 已实测不可行（见第 7 节）。
2. 草稿模型换非量化版 — 已实测，净更差（见第 6 节），不再是候选。
3. `VLLM_MARLIN_USE_ATOMIC_ADD=1` — 已采用（见第 5 节）。
4. 全档位复测（32K/64K/96K/128K/160K/192K/211K）出最终表。

---

## 15. 与官方参考口径对比（REPORT 第四节；blog-dflash2.txt；gh-dflash-README.md）

- fork 官方 2×2080Ti：32K prefill ≈ 1265 t/s；我们实测 32K TTFT 24.1 s / 1479 t/s（同口径持平）。
- fork 官方同源配对：32K decode 209–220 t/s；我们 122.58–128.74 t/s。
- 官方博客（`blog-dflash2.txt`，Inco AI，2026-08-18）：Qwen3.8-27B 上 DFlash 2 相对自回归解码
  吞吐 2.7–3.4×；Qwen3.8-27B mean acceptance length：MTP 4.28 / DSpark 3.62 / DFlash 2 4.80（block size 8）。
  该表口径为「模型默认采样」，与我们 harness 语料口径不同，仅作参考。
- `gh-dflash-README.md`：DFlash 2 支持 Qwen3.8-27B；vLLM 走 PR 52816，配置
  `--speculative-config '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'`。

---

## 16. 硬件 / 软件基线（deployment 用；来源 spec.sh / dir.env / A-下载报告.md / REPORT）

- GPU：2× RTX 2080 Ti，NVLink NV2，`CUDA_VISIBLE_DEVICES=0,1`，TP=2，SM75（`TORCH_CUDA_ARCH_LIST=7.5`）
- CUDA：`/usr/local/cuda-13.0`（CUDA_HOME / CUDA_PATH 同）
- 主机内存：31 GB RAM（A-下载报告.md）
- 磁盘：`/` 468G，可用 234G（2026-10-13）；主机剩余 194 GB（taskC 记录）
- vLLM fork 环境（`remote/bfetch/dir.env` 快照）：
  `PATH` 前缀 `/root/vLLM-2080Ti-Definitive-main/.venv/bin`；
  `FLASHQLA_ROOT` / `PYTHONPATH` = `/root/vLLM-2080Ti-Definitive-main/.deps/FlashQLA-SM70-SM75`；
  `TORCH_EXTENSIONS_DIR=…/FlashQLA-SM70-SM75/.torch_extensions_vllm_flashqla_legacy`；
  `TORCHINDUCTOR_CACHE_DIR=…/torchinductor-cache`；`TRITON_CACHE_DIR=…/triton-cache`；
  `FLASHINFER_ENABLE_AOT=1`；`FLASHINFER_WORKSPACE_BASE=/root/vLLM-2080Ti-Definitive-main`；
  `VLLM_DISABLE_TILELANG=1`；`VLLM_SM75_SPEC_SYNC_MODE=safe`；`VLLM_KV_CACHE_LAYOUT=LBNHC`；
  `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1`；`VLLM_ENFORCE_STRICT_TOOL_CALLING=0`；
  `VLLM_KV_BUDGET_FROM_FREE_MEMORY=1`；`VLLM_KV_BUDGET_REDUNDANCY_MIB=150`；
  `PYTHONUNBUFFERED=1`；`PYTHONSAFEPATH=1`；`LANG=zh_CN.UTF-8`
- 追加到 unit 的环境变量：`VLLM_MARLIN_USE_ATOMIC_ADD=1`
- torch / vLLM 具体版本号：**未实测**（`spec.sh` 只打印 `torch.__version__`，本机未保存其输出）

---

## 17. vLLM 启动配置要素（deployment 用）

来源：`remote/bfetch/current-vllm-cmd.txt`（同 fork 的既有启动命令，模型为 Uncensored-Aggressive 变体）+
`change-*.sh` 记录的改动项 + REPORT 的口径。**逐项来源见右列；未逐字核对的标「未实测」。**

| 参数 | 值 | 来源 |
|---|---|---|
| 启动方式 | `python -m vllm.entrypoints.openai.api_server` | current-vllm-cmd.txt |
| `--host` / `--port` | `0.0.0.0` / `8000` | current-vllm-cmd.txt |
| `--model` | `/root/models/Coder390-HUHUI-INC-A-W4A16` | REPORT-INC-A.md 第七节（实际切换记录） |
| `--dtype` | `half` | current-vllm-cmd.txt |
| `--quantization` | `compressed-tensors` | current-vllm-cmd.txt |
| `--tensor-parallel-size` | `2` | current-vllm-cmd.txt |
| `--generation-config` / `--override-generation-config` | `auto` / `{"max_new_tokens":24576}` | current-vllm-cmd.txt |
| `--gpu-memory-utilization` | `0.95` | change-ctx220-fp8.sh（0.975 → 0.95） |
| `--max-model-len` | `225280` | change-ctx220-fp8.sh（262144 → 225280） |
| `--enable-chunked-prefill` | 开 | current-vllm-cmd.txt |
| `--max-num-seqs` | `1` | current-vllm-cmd.txt |
| `--max-num-batched-tokens` | `16384` | REPORT 第七节（现行值；试跑 24576 已回滚） |
| `--kv-cache-dtype` | `fp8` | current-vllm-cmd.txt + REPORT 第三节 |
| `--mamba-cache-mode` | `align` | current-vllm-cmd.txt |
| `--enable-prefix-caching` | 开 | current-vllm-cmd.txt |
| `--enable-prompt-tokens-details` | 开 | current-vllm-cmd.txt |
| `--reasoning-parser` | `qwen3` | current-vllm-cmd.txt |
| `--limit-mm-per-prompt` | `{"image":64,"video":0,"audio":0}` | current-vllm-cmd.txt |
| `--enable-auto-tool-choice` / `--tool-call-parser` | 开 / `qwen3_xml` | current-vllm-cmd.txt |
| `--additional-config` | `{"gdn_prefill_backend":"flashqla_legacy"}` | current-vllm-cmd.txt |
| `--per-request-spec-decode-metrics` | `detailed` | current-vllm-cmd.txt |
| `--speculative-config` | `{"method":"dflash","num_speculative_tokens":7,"model":"/root/models/Qwen3.8-27B-DFlash2-W4A16","attention_backend":"TRITON_ATTN","kv_cache_dtype":"float16","draft_sample_method":"probabilistic"}` | current-vllm-cmd.txt + change-draftprob.sh + revert-draft-w4a16.sh |
| `--compilation-config` | `{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8],"max_cudagraph_capture_size":8}` | current-vllm-cmd.txt |
| `--served-model-name` | `Qwen3.8-27B-Coder390-Abliterated-W4A16` `Qwen3.8-27B-Uncensored-AWQ`（切换 INC-A 时未改） | REPORT-INC-A.md 第七节 |

启动耗时参考：切换模型后 restart → `:8000` 返回 200 约 230 s（REPORT-INC-A.md 第七节）；
chunk 改 24576 时首次启动触发完整重编译实测 41 s（REPORT 第七节）。

**未实测**：`--max-num-batched-tokens` 之外的 unit 原文逐字内容；unit 的 `ExecStart` 完整单行文本。

---

## 18. 转发器与上下文守卫（deployment 用）

- 转发器监听 `127.0.0.1:8788`，把 harness/客户端请求打到模型主机 `:8000`（REPORT 首段；run-bench.ps1 探活 `http://127.0.0.1:8788/v1/models`）
- `CTX_GUARD`：prompt 上限 `200704 = 225280 − 24576`；超限直接 400（REPORT 第三节第 5 条；第 8 节实测 203,707 > 200,704）
- 主机侧 `wb-prefix-warm.timer` 会周期性重放前缀请求，把投机计数器搅脏
  （表现为「接受 token 数 > 实际生成 token 数」），测速期间需停掉（REPORT 第八节）
- 停/起脚本：`toggle-timers.sh stop|start`（停 `wb-prefix-warm.timer` / `wb-prefix-warm.service` / `vllm-warmup.service`）

---

## 19. 测速复现（deployment 用）

1. 探活转发器：`http://127.0.0.1:8788/v1/models` 返回 200（`run-bench.ps1` 内置该检查，非 200 直接退出）
2. 跑档位：`python -u bench-harness-real.py [档位…]`，档位缺省
   `[1024, 8192, 32768, 65536, 98304, 131072, 163840, 196608, 229376]`；`gen` 固定 256
3. 环境变量：`BENCH_TEMP`（默认 0.7）、`BENCH_SEED`（可选；A/B 对照用 1234）
4. 启动器：`run-bench.ps1 -Tag <name> [-TiersCsv "32768,163840"]`，日志写 `logs/bench-<Tag>.log`
5. 结果落盘：`D:\1\靶场\全层消融\bench-harness-real.json`（每档一条，含 `accept_rate` / `mean_accept_len`）
6. 每档前后各读一次 `/metrics` 的 `spec_decode` 三个计数器差分，得到 drafts / drafted / accepted 与 `pos0..6`
7. 单步延迟口径：`(total_s − ttft_s) × 1000 / drafts`（与接受率无关，A/B 首选）
8. 复现纪律：测速期间停掉 `wb-prefix-warm.timer`；用固定 `seed` 做 A/B；接受率波动区间 0.28–0.55，
   单跑不可作为结论，需两档方向一致才判定（REPORT 第八节）

---

## 20. 其他语料口径（附录 B；来自 `remote/bfetch/bench-prod-huh.json`、`bench-prod.json`）

与第 1 节 harness 语料**不同**（prompt 长度不同、`reasoning_tokens` 参与统计），仅作参考。

bench-prod-huh.json（目标 = huihui 生产版，128 token 输出）：

| target | prompt_tok | TTFT_s | decode t/s | prefill t/s |
|---|---:|---:|---:|---:|
| 1024 | 1,249 | 1.22 | 159.76 | 1,024.0 |
| 32768 | 36,658 | 26.915 | 133.51 | 1,362.0 |
| 65536 | 73,200 | 63.574 | 132.75 | 1,151.4 |
| 131072 | 146,308 | 166.138 | 91.33 | 880.6 |
| 196608 | 219,389 | 320.498 | 100.38 | 684.5 |
| 229376 | 255,958 | 391.429 | 152.37 | 653.9 |

bench-prod.json（同前一轮，无 prefill_t/s 字段）：1024→103.74、32768→106.44、65536→130.7、
131072→177.33、196608→153.06 t/s。

**注意**：本表数值明显受「输出仅 128 token、reasoning 占比不定」影响，不可与 harness 语料口径的表 1–8 混用。

---

## 21. 本底稿的溯源清单（每张表的来源文件名）

| 文档章节 | 来源文件 |
|---|---|
| benchmarks 表 1 | REPORT 第二节；logs/bench-225w.log、bench-prob.log、bench-prob-t2.log |
| benchmarks 表 2 | REPORT 第三/四节；logs/bench-prob.log、bench-prob-t2.log |
| benchmarks 表 3 | REPORT 第三节；logs/bench-225w.log、bench-260w.log |
| benchmarks 表 4 | REPORT 第八节；logs/bench-base225.log、bench-marlineadd.log、bench-marlineadd-t2.log、bench-ea-clean.log |
| benchmarks 表 5 | REPORT 第九节；logs/bench-base-clean.log、bench-draftbf16.log |
| benchmarks 表 6 | REPORT 第七节 |
| benchmarks 表 7 | logs/bench-tail2.log；REPORT 第三节第 5 条 |
| benchmarks 表 8 | logs/bench-harness-real.log |
| benchmarks 附录 B | remote/bfetch/bench-prod-huh.json、bench-prod.json |
| optimization 第 6 节 | REPORT 第四节；blog-dflash2.txt；gh-dflash-README.md |
| deployment 第 3 节 | logs/verify.out；A-下载报告.md；REPORT 第九节 |
| deployment 第 4 节 | remote/bfetch/current-vllm-cmd.txt；change-*.sh；remote/bfetch/dir.env |
