#!/usr/bin/env python3
"""在「deepseek harness 真实任务」语料上做全档位单流测速。

模板取自转发器落盘的真实请求：
  req-dump/fail-2026-10-14T05-44-25-018Z-400-qwen-vllm-main-qwen3.8-27b.json
（harness 真实会话：1 条 system + 9 user + 58 assistant + 73 tool + 38 个真实工具定义）

做法：保留真实 system 提示与真实工具表，用该会话里真实的 user/assistant 文本消息
循环填充到目标档位，末条用真实的用户提问收尾；请求走本机转发器 127.0.0.1:8788
（与 harness 同一条链路），并逐档读取投机接受率。
"""
import json
import os
import sys
import time
import urllib.request
import uuid

ROUTER = "http://127.0.0.1:8788"
METRICS = "http://192.168.5.88:8000/metrics"
MODEL = "qwen3.8-27b"
TEMP = float(os.environ.get("BENCH_TEMP", "0.7"))
SEED = os.environ.get("BENCH_SEED", "")
DUMP = r"D:\1\靶场\codex模型管理\req-dump\fail-2026-10-14T05-44-25-018Z-400-qwen-vllm-main-qwen3.8-27b.json"
GEN = 256
TIERS = [1024, 8192, 32768, 65536, 98304, 131072, 163840, 196608, 229376]
SPEC_KEYS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def est_tokens(text: str) -> int:
    """粗估：中英混排约 3.6 字符/token（只用来自适应拼到目标档位）。"""
    return max(1, int(len(text) / 3.6))


def load_template():
    with open(DUMP, "r", encoding="utf-8") as fh:
        body = json.load(fh)
    msgs = body["messages"]
    system = next(m for m in msgs if m.get("role") == "system")
    # 按「assistant(+tool_calls) + 紧随其后的 tool 结果」整块切分：
    # 整块复用能保证 tool_call_id 配对合法，且与 harness 真实报文同构。
    blocks = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m.get("role") == "system":
            i += 1
            continue
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids = {c.get("id") for c in m["tool_calls"]}
            blk = [m]
            j = i + 1
            while j < len(msgs) and msgs[j].get("role") == "tool" and msgs[j].get("tool_call_id") in ids:
                blk.append(msgs[j])
                j += 1
            blocks.append(blk)
            i = j
            continue
        blocks.append([m])
        i += 1

    def blk_chars(b):
        return sum(len(json.dumps(x, ensure_ascii=False)) for x in b)

    user_blocks = [b for b in blocks if b[0].get("role") == "user"]
    final_block = user_blocks[-1] if user_blocks else [{"role": "user", "content": "继续"}]
    final_user = final_block[0]
    # 填充素材：跳过超大块（那份 126KB skills reminder 单独用）与末条提问块
    pad_blocks = [
        b for b in blocks
        if b is not final_block and blk_chars(b) < 20000
    ]
    big_user = max(user_blocks, key=blk_chars) if user_blocks else None
    return {
        "system": system,
        "tools": body["tools"],
        "pad_blocks": pad_blocks,
        "big_user_block": big_user if (big_user and blk_chars(big_user) > 50000) else None,
        "final_block": final_block,
        "final_user": final_user,
    }


def build_messages(tpl, target_tokens: int):
    msgs = [tpl["system"]]
    cur = est_tokens(tpl["system"]["content"])
    # 大于 40K 的档位带上真实会话里那份 skills system-reminder（真实存在的 126KB 大块）
    if target_tokens > 40000 and tpl["big_user_block"]:
        for m in tpl["big_user_block"]:
            msgs.append(m)
            cur += est_tokens(json.dumps(m, ensure_ascii=False))
    i = 0
    pads = tpl["pad_blocks"]
    while cur < target_tokens - 400:
        blk = pads[i % len(pads)]
        for m in blk:
            msgs.append(m)
            cur += est_tokens(json.dumps(m, ensure_ascii=False))
        i += 1
        if i > 2000:
            break
    for m in tpl["final_block"]:
        msgs.append(m)
    return msgs


def spec_metrics():
    with urllib.request.urlopen(METRICS, timeout=15) as r:
        txt = r.read().decode("utf-8", "ignore")
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for k in SPEC_KEYS:
            if line.startswith(k + "{"):
                out[k] = float(line.rsplit(" ", 1)[1])
        if line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{"):
            pos = line.split('position="', 1)[1].split('"', 1)[0]
            out["pos:" + pos] = float(line.rsplit(" ", 1)[1])
    return out


def run_one(msgs, max_tokens: int):
    payload = {
        "model": MODEL,
        "messages": msgs,
        "max_completion_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": TEMP,
    }
    if TEMP > 0:
        payload["top_p"] = 0.8
        payload["top_k"] = 20
        payload["repetition_penalty"] = 1.05
    if SEED != "":
        payload["seed"] = int(SEED)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        ROUTER + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    ttft = None
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                d = json.loads(body)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                if any(k in delta for k in ("content", "reasoning_content", "tool_calls")):
                    if ttft is None:
                        ttft = time.time() - t0
    total = time.time() - t0
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    win = total - (ttft or 0)
    return {
        "prompt_tokens": pt,
        "cached_tokens": cached,
        "completion_tokens": ct,
        "reasoning_tokens": rt,
        "ttft_s": round(ttft or 0, 3),
        "total_s": round(total, 3),
        "prefill_tps": round((pt - cached) / (ttft or 1), 1) if ttft else 0.0,
        "decode_tps": round((ct - 1) / win, 2) if ct > 1 and win > 0 else 0.0,
    }


def main() -> int:
    tpl = load_template()
    tiers = [int(x) for x in sys.argv[1:]] or TIERS
    print(
        f"### BENCH harness-real  model={MODEL}  via={ROUTER}  gen={GEN}\n"
        f"模板：{DUMP}\n"
        f"模板规模：system={len(tpl['system']['content'])}字符 tools={len(tpl['tools'])}个 "
        f"填充块={len(tpl['pad_blocks'])} 大块(>50KB)={'有' if tpl['big_user_block'] else '无'}",
        flush=True,
    )
    print(
        f"{'档位':>8} {'prompt_tok':>10} {'cached':>8} {'gen':>5} {'reason':>7} {'TTFT_s':>9} "
        f"{'total_s':>9} {'prefill_t/s':>11} {'decode_t/s':>11} {'drafts':>7} {'acc_rate':>9} "
        f"{'mean_len':>9} {'pos0..6':>26}",
        flush=True,
    )
    rows = []
    for target in tiers:
        msgs = build_messages(tpl, target)
        m0 = spec_metrics()
        try:
            r = run_one(msgs, GEN)
        except Exception as e:  # noqa: BLE001
            print(f"{target:>8} FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        m1 = spec_metrics()
        drafts = m1[SPEC_KEYS[0]] - m0[SPEC_KEYS[0]]
        drafted = m1[SPEC_KEYS[1]] - m0[SPEC_KEYS[1]]
        acc = m1[SPEC_KEYS[2]] - m0[SPEC_KEYS[2]]
        poss = [int(m1.get("pos:%d" % i, 0) - m0.get("pos:%d" % i, 0)) for i in range(7)]
        rate = round(acc / drafted, 4) if drafted else 0.0
        mlen = round(acc / drafts + 1, 3) if drafts else 0.0
        r["tier"] = target
        r["accept_rate"] = rate
        r["mean_accept_len"] = mlen
        rows.append(r)
        print(
            f"{target:>8} {r['prompt_tokens']:>10} {r['cached_tokens']:>8} {r['completion_tokens']:>5} "
            f"{r['reasoning_tokens']:>7} {r['ttft_s']:>9} {r['total_s']:>9} {r['prefill_tps']:>11} "
            f"{r['decode_tps']:>11} {drafts:>7.0f} {rate:>9} {mlen:>9} {str(poss):>26}",
            flush=True,
        )
        with open(r"D:\1\靶场\全层消融\bench-harness-real.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
    print("### DONE harness-real", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
