#!/bin/bash
# Warm up vLLM kernels across context lengths so speculative-decoding Triton
# kernels are compiled before real traffic. Each run uses a unique prefix so
# prefix-caching cannot skip the prefill (which is what compiles the kernels).
#
# Hardening (2026-10-10):
#   - require the server to be healthy TWICE in a row (a restarting instance can
#     briefly answer /health from the process that is still shutting down);
#   - verify every warmup request really returned a completion, and retry,
#     instead of silently reporting "done in 0s" when the server is not up yet;
#   - exit non-zero when any context length could not be warmed, so the unit
#     shows up as failed instead of pretending success.
set -u
BASE=${WARMUP_BASE:-http://127.0.0.1:8000}
MODEL=${WARMUP_MODEL:-Qwen3.8-27B-Uncensored-AWQ}
SIZES=${WARMUP_SIZES:-1024 16384 32768 65536}
PY=${WARMUP_PY:-/root/vLLM-2080Ti-Definitive-main/.venv/bin/python}

health_ok() {
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$BASE/health" 2>/dev/null || echo 000)
  [ "$code" = "200" ]
}

ready=0
for i in $(seq 1 120); do
  if health_ok; then
    sleep 3
    if health_ok; then ready=1; break; fi
  fi
  sleep 5
done
if [ "$ready" != "1" ]; then
  echo "warmup: server never became stably healthy"
  exit 1
fi
echo "warmup: server healthy, starting $(date -Is)"

RUN=$(date +%s)
FAILED=0
for n in $SIZES; do
  TMP=$(mktemp /tmp/warmup-XXXXXX.json)
  "$PY" - "$TMP" "$n" "$RUN" "$MODEL" <<'PY'
import json, sys
path, n, run, model = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
UNIT = ("the quick brown fox jumps over the lazy dog while the system keeps "
        "processing tokens in a steady and predictable manner ")
reps = max(1, n // 12)
prompt = "[warmup-%s-%d] " % (run, n) + UNIT * reps
json.dump({"model": model, "prompt": prompt, "max_tokens": 96,
           "temperature": 0.0, "ignore_eos": True}, open(path, "w"))
PY
  ok=0
  for attempt in 1 2 3 4 5; do
    t0=$(date +%s)
    body=$(curl -s -m 1200 -w '\n%{http_code}' "$BASE/v1/completions" \
      -H 'Content-Type: application/json' --data-binary @"$TMP" 2>/dev/null)
    code=$(printf '%s' "$body" | tail -n1)
    t1=$(date +%s)
    if [ "$code" = "200" ] && printf '%s' "$body" | grep -q '"choices"'; then
      ok=1
      echo "warmup: ctx=$n done in $((t1-t0))s (attempt $attempt)"
      break
    fi
    echo "warmup: ctx=$n attempt $attempt failed (http=$code), retry in 10s"
    sleep 10
  done
  [ "$ok" = "1" ] || { echo "warmup: ctx=$n FAILED after retries"; FAILED=1; }
  rm -f "$TMP"
done
if [ "$FAILED" = "0" ]; then
  echo "warmup: complete"
  exit 0
fi
echo "warmup: incomplete"
exit 1
