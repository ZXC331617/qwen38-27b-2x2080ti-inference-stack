# 测速工具

**简体中文** | English below

这套脚本用来复现 [docs/benchmarks.md](../docs/benchmarks.md) 里的全部数字。

## 文件

| 文件 | 说明 |
|---|---|
| `bench-harness-real.py` | 主测速脚本。按档位发请求，记录 TTFT / total / prefill / decode，并读 vLLM 的投机解码计数器 |
| `run-bench.ps1` | Windows 启动器，后台运行并把输出写进 `logs/bench-<tag>.log` |

## 口径

- 语料 = 一份 deepseek-harness 真实会话报文（1 system + 58 工具结果 + 38 工具定义）
- 采样 = `temperature=0.7 / top_p=0.8 / top_k=20 / repetition_penalty=1.05`（`BENCH_TEMP` 可覆盖温度）
- 单流（`--max-num-seqs 1`），`gen=256`

## 用法

```powershell
# 1) 先停掉会污染投机计数器的前缀保温定时器
ssh <model-host> "sudo systemctl stop wb-prefix-warm.timer"

# 2) 跑档位扫描
$env:BENCH_TEMP="0.7"
pwsh -NoProfile -ExecutionPolicy Bypass -File .\run-bench.ps1 `
  -Tag final-prod2 -TiersCsv "32768,65536,98304,131072,163840,196608"

# 3) 测完恢复
ssh <model-host> "sudo systemctl start wb-prefix-warm.timer"
```

## 需要按自己环境修改的地方

脚本里写死了本项目的本机路径，移植时改这几处：

- `bench-harness-real.py` 顶部的**模板 JSON 路径**（真实会话报文）与转发器地址（默认 `http://127.0.0.1:8788`）
- `run-bench.ps1` 里的 Python 解释器路径与 `logs` 输出目录

## 注意

`drafts / acc_rate / mean_len` 来自 vLLM 的**全局** Prometheus 计数器，
测量窗口内只要有并发流量就会被一起算进来。要拿到干净的接受率，
必须保证测量期间没有别的客户端在用同一个模型端点。

---

## English

These scripts reproduce every number in [docs/benchmarks.en.md](../docs/benchmarks.en.md).

| File | Purpose |
|---|---|
| `bench-harness-real.py` | main benchmark: per-tier requests, records TTFT / total / prefill / decode and reads vLLM's speculative-decode counters |
| `run-bench.ps1` | Windows launcher; runs in background and writes `logs/bench-<tag>.log` |

Protocol: corpus = a real deepseek-harness session payload; sampling =
`temperature=0.7 / top_p=0.8 / top_k=20 / repetition_penalty=1.05` (`BENCH_TEMP` overrides
temperature); single stream (`--max-num-seqs 1`), `gen=256`.

Before running, stop `wb-prefix-warm.timer` on the model host, otherwise the speculative
counters are polluted. Paths in the scripts are project-local; adjust the template JSON path,
the forwarder URL and the Python interpreter path for your environment.

`drafts / acc_rate / mean_len` come from vLLM's **global** Prometheus counters — any concurrent
traffic during the measurement window is mixed in.
