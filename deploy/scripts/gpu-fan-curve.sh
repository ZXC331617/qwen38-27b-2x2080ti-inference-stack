#!/bin/bash
# GPU fan curve for 2x RTX 2080 Ti (Turing). Requires an X server with the
# nvidia driver loaded (Xorg :1) so that NV-CONTROL fan attributes exist.
# Curve: <50C=40%  50-60C=55%  60-70C=70%  70-80C=85%  >=80C=100%
set -u
DISP=${FAN_DISPLAY:-:1}
INTERVAL=${FAN_INTERVAL:-10}

fan_for_gpu() { echo $(( $1 * 2 )); }

set_fan() {
  local gpu=$1 speed=$2
  nvidia-settings -c "$DISP" -a "[gpu:${gpu}]/GPUFanControlState=1" >/dev/null 2>&1 || return 1
  local f0 f1
  f0=$(fan_for_gpu "$gpu"); f1=$(( f0 + 1 ))
  nvidia-settings -c "$DISP" -a "[fan:${f0}]/GPUTargetFanSpeed=${speed}" >/dev/null 2>&1
  nvidia-settings -c "$DISP" -a "[fan:${f1}]/GPUTargetFanSpeed=${speed}" >/dev/null 2>&1
}

speed_for_temp() {
  local t=$1
  if   [ "$t" -lt 50 ]; then echo 40
  elif [ "$t" -lt 60 ]; then echo 55
  elif [ "$t" -lt 70 ]; then echo 70
  elif [ "$t" -lt 80 ]; then echo 85
  else echo 100
  fi
}

restore_auto() {
  for g in 0 1; do
    nvidia-settings -c "$DISP" -a "[gpu:${g}]/GPUFanControlState=0" >/dev/null 2>&1
  done
}
trap restore_auto EXIT

# 2026-10-11 修：连续失败计数。Xorg :1 挂掉时 nvidia-settings 会一直失败，
# 旧版会静默空转（风扇停在自动曲线）。现在连续 3 轮（每轮 2 卡 = 6 次）失败即
# 退出，由 systemd（Restart=always）重启并触发 ExecStartPre 重建 Xorg :1。
fails=0
while true; do
  for g in 0 1; do
    t=$(nvidia-smi -i "$g" --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
    if [ -z "$t" ]; then
      fails=$((fails + 1))
      continue
    fi
    s=$(speed_for_temp "$t")
    if set_fan "$g" "$s"; then
      fails=0
    else
      fails=$((fails + 1))
    fi
  done
  if [ "$fails" -ge 6 ]; then
    echo "$(date '+%F %T') fan curve: display $DISP unreachable, exiting for restart"
    exit 1
  fi
  sleep "$INTERVAL"
done
