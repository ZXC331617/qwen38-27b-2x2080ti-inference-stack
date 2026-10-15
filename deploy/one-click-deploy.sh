#!/usr/bin/env bash
#
# one-click-deploy.sh — install / uninstall / verify the
# Qwen3.8-27B x 2x RTX 2080 Ti inference stack on Ubuntu (systemd).
#
#   sudo ./one-click-deploy.sh                 # install units, enable at boot
#   sudo ./one-click-deploy.sh --start         # install + start now
#   sudo ./one-click-deploy.sh --verify        # post-install health checks
#   sudo ./one-click-deploy.sh --uninstall     # stop + remove what we installed
#
# Docs: ../docs/deployment.md  |  ../deploy/README.md
#
set -uo pipefail

VERSION="1.0.0"

# ---------------------------------------------------------------- defaults
VLLM_DIR="/root/vLLM-2080Ti-Definitive-main"
MODEL_DIR="/root/models/Coder390-HUHUI-INC-A-W4A16"
DRAFT_DIR="/root/models/Qwen3.8-27B-DFlash2-W4A16"
POWER_LIMIT="225"

WITH_MONITORING=0
WITH_FAN_CURVE=1
WITH_PREFIX_WARM=1
DO_START=0
DO_UNINSTALL=0
DO_STATUS=0
DO_VERIFY=0
DRY_RUN=0
ASSUME_YES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

UNIT_DIR="/etc/systemd/system"
BIN_DIR="/usr/local/bin"
STAMP="$(date +%Y%m%d-%H%M%S)"

# units owned by this installer (basename -> source path under deploy/)
CORE_UNITS=(
  "gpu-power-limit.service"
  "gpu-fan-curve.service"
  "vllm-qwen38.service"
  "vllm-warmup.service"
  "wb-prefix-warm.service"
  "wb-prefix-warm.timer"
)
MON_UNITS=(
  "prometheus.service"
  "grafana-server.service"
  "node_exporter.service"
  "nvidia_gpu_exporter.service"
)

# ---------------------------------------------------------------- helpers
C_R="\033[31m"; C_G="\033[32m"; C_Y="\033[33m"; C_B="\033[36m"; C_0="\033[0m"
say()  { printf '%b\n' "${C_B}==>${C_0} $*"; }
ok()   { printf '%b\n' "  ${C_G}ok${C_0}   $*"; }
warn() { printf '%b\n' "  ${C_Y}warn${C_0} $*"; }
err()  { printf '%b\n' "  ${C_R}fail${C_0} $*" >&2; }
die()  { err "$*"; exit 1; }

run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '%b\n' "  ${C_Y}dry${C_0}  $*"
  else
    "$@"
  fi
}

run_shell() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '%b\n' "  ${C_Y}dry${C_0}  $*"
  else
    bash -c "$*"
  fi
}

confirm() {
  [ "$ASSUME_YES" = "1" ] && return 0
  printf '%s [y/N] ' "$1"
  read -r a
  case "$a" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

usage() {
  sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --vllm-dir DIR       vLLM fork directory      (default /root/vLLM-2080Ti-Definitive-main)
  --model-dir DIR      main model directory     (default /root/models/Coder390-HUHUI-INC-A-W4A16)
  --draft-dir DIR      DFlash2 draft directory  (default /root/models/Qwen3.8-27B-DFlash2-W4A16)
  --power-limit W      GPU power limit in watts (default 225; 0 = do not install the unit)
  --with-monitoring    also install Prometheus / Grafana / node_exporter / nvidia_gpu_exporter
  --no-fan-curve       skip gpu-fan-curve.service (headless box, no X server)
  --no-prefix-warm     skip the prefix-cache warm-up service + timer
  --start              enable AND start the services (default: enable only)
  --uninstall          stop and remove everything this script installed
  --status             print current status and exit
  --verify             run health checks and exit
  --dry-run            print what would happen, change nothing
  -y, --yes            assume yes, never prompt
  -h, --help           this text
EOF
}

# ---------------------------------------------------------------- arg parse
while [ $# -gt 0 ]; do
  case "$1" in
    --vllm-dir)        VLLM_DIR="${2:?}"; shift 2 ;;
    --model-dir)       MODEL_DIR="${2:?}"; shift 2 ;;
    --draft-dir)       DRAFT_DIR="${2:?}"; shift 2 ;;
    --power-limit)     POWER_LIMIT="${2:?}"; shift 2 ;;
    --with-monitoring) WITH_MONITORING=1; shift ;;
    --no-fan-curve)    WITH_FAN_CURVE=0; shift ;;
    --no-prefix-warm)  WITH_PREFIX_WARM=0; shift ;;
    --start)           DO_START=1; shift ;;
    --uninstall)       DO_UNINSTALL=1; shift ;;
    --status)          DO_STATUS=1; shift ;;
    --verify)          DO_VERIFY=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    -y|--yes)          ASSUME_YES=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    *)                 err "unknown option: $1"; usage; exit 2 ;;
  esac
done

need_root() {
  [ "$(id -u)" = "0" ] || die "run as root:  sudo $0 $*"
}

have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- status
do_status() {
  say "service status"
  for u in vllm-qwen38.service vllm-warmup.service gpu-power-limit.service \
           gpu-fan-curve.service wb-prefix-warm.service wb-prefix-warm.timer \
           prometheus.service grafana-server.service node_exporter.service nvidia_gpu_exporter.service; do
    if systemctl list-unit-files "$u" >/dev/null 2>&1 && [ -n "$(systemctl list-unit-files "$u" --no-legend 2>/dev/null)" ]; then
      printf '  %-32s %-10s %s\n' "$u" "$(systemctl is-active "$u" 2>/dev/null)" "$(systemctl is-enabled "$u" 2>/dev/null)"
    fi
  done
  echo
  say "GPU"
  nvidia-smi --query-gpu=index,name,power.limit,power.draw,temperature.gpu,utilization.gpu,memory.used \
             --format=csv,noheader 2>/dev/null || warn "nvidia-smi unavailable"
  echo
  say "API"
  curl -s -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 \
    && ok "http://127.0.0.1:8000/v1/models reachable" \
    || warn "http://127.0.0.1:8000/v1/models not reachable"
}

# ---------------------------------------------------------------- verify
do_verify() {
  local rc=0
  say "post-install verification"

  if systemctl is-active --quiet vllm-qwen38.service; then
    ok "vllm-qwen38.service active"
  else
    err "vllm-qwen38.service not active  (journalctl -u vllm-qwen38 -n 80)"
    rc=1
  fi

  local models
  models="$(curl -s -m 8 http://127.0.0.1:8000/v1/models 2>/dev/null)"
  if printf '%s' "$models" | grep -q '"id"'; then
    ok "model endpoint answers:"
    printf '%s' "$models" | grep -o '"id":"[^"]*"' | sed 's/^/       /'
  else
    err "/v1/models gave no model list"
    rc=1
  fi

  if [ -f "$MODEL_DIR/config.json" ]; then
    ok "main model config present: $MODEL_DIR/config.json"
  else
    err "missing $MODEL_DIR/config.json"
    rc=1
  fi
  if [ -f "$DRAFT_DIR/config.json" ]; then
    ok "draft model config present: $DRAFT_DIR/config.json"
  else
    warn "missing $DRAFT_DIR/config.json (speculative decoding will fail)"
  fi

  local pl
  pl="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1)"
  if [ -n "$pl" ]; then
    ok "GPU0 power limit: ${pl} W (target ${POWER_LIMIT} W)"
  fi

  local kv
  kv="$(journalctl -u vllm-qwen38.service --no-pager -n 400 2>/dev/null | grep -i 'GPU KV cache size' | tail -1)"
  [ -n "$kv" ] && ok "KV pool: ${kv#*] }" || warn "KV cache line not found in the last 400 journal lines"

  if curl -s -m 5 http://127.0.0.1:9090/-/ready >/dev/null 2>&1; then ok "prometheus ready"; fi
  if curl -s -m 5 http://127.0.0.1:3000/api/health >/dev/null 2>&1; then ok "grafana healthy"; fi

  echo
  [ "$rc" = "0" ] && say "verification PASSED" || { err "verification FAILED"; return 1; }
}

# ---------------------------------------------------------------- uninstall
do_uninstall() {
  need_root "$@"
  say "uninstalling"
  local all=("${CORE_UNITS[@]}" "${MON_UNITS[@]}")
  for u in "${all[@]}"; do
    if systemctl list-unit-files "$u" >/dev/null 2>&1 && [ -n "$(systemctl list-unit-files "$u" --no-legend 2>/dev/null)" ]; then
      run systemctl disable --now "$u" 2>/dev/null || true
    fi
  done
  for u in "${all[@]}"; do
    [ -f "$UNIT_DIR/$u" ] && run rm -f "$UNIT_DIR/$u"
  done
  for s in gpu-power-limit.sh gpu-fan-curve.sh vllm-warmup.sh wb-prefix-warm.sh wb-prefix-warm.py; do
    [ -f "$BIN_DIR/$s" ] && run rm -f "$BIN_DIR/$s"
  done
  run systemctl daemon-reload
  run systemctl reset-failed 2>/dev/null || true
  ok "units and helper scripts removed (model weights and $VLLM_DIR were left untouched)"
  warn "backups of any replaced units are under $UNIT_DIR/*.bak-oneclick-*"
}

# ---------------------------------------------------------------- install
render_unit() {
  # $1 = source unit, $2 = destination path
  sed \
    -e "s|/root/vLLM-2080Ti-Definitive-main|${VLLM_DIR}|g" \
    -e "s|/root/models/Coder390-HUHUI-INC-A-W4A16|${MODEL_DIR}|g" \
    -e "s|/root/models/Qwen3.8-27B-DFlash2-W4A16|${DRAFT_DIR}|g" \
    -e "s|^Environment=GPU_POWER_LIMIT_W=.*|Environment=GPU_POWER_LIMIT_W=${POWER_LIMIT}|" \
    "$1" > "$2"
}

install_unit() {
  # $1 = source path, $2 = basename
  local src="$1" base="$2" dst="$UNIT_DIR/$2"
  if [ -f "$dst" ] && ! cmp -s "$src" "$dst"; then
    run cp -a "$dst" "$dst.bak-oneclick-$STAMP"
    warn "existing $base backed up to $base.bak-oneclick-$STAMP"
  fi
  if [ "$DRY_RUN" = "1" ]; then
    printf '%b\n' "  ${C_Y}dry${C_0}  render+install $base"
    return
  fi
  render_unit "$src" "$dst"
  chmod 0644 "$dst"
}

do_install() {
  need_root "$@"
  say "one-click deploy v$VERSION   (repo: $REPO_DIR)"
  echo "  vllm dir   : $VLLM_DIR"
  echo "  model dir  : $MODEL_DIR"
  echo "  draft dir  : $DRAFT_DIR"
  echo "  power limit: ${POWER_LIMIT} W"
  echo

  # ---- preflight -------------------------------------------------------
  say "preflight"
  have systemctl || die "systemd not found - this installer targets systemd Linux"
  ok "systemd present"

  if have nvidia-smi; then
    local n
    n="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
    ok "nvidia-smi found, ${n} GPU(s)"
    [ "$n" -ge 2 ] || warn "this stack is tuned for 2 GPUs (TP=2); found ${n}"
    nvidia-smi --query-gpu=name,compute_cap,power.limit --format=csv,noheader 2>/dev/null | sed 's/^/       /'
    local cc
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
    [ "$cc" = "7.5" ] || warn "compute capability ${cc} != 7.5; the SM75-specific flags in this stack assume Turing"
  else
    die "nvidia-smi not found - install the NVIDIA driver first"
  fi

  if [ -x "$VLLM_DIR/.venv/bin/python" ]; then
    ok "vLLM venv python: $VLLM_DIR/.venv/bin/python"
  else
    warn "no venv at $VLLM_DIR/.venv - place the vLLM fork + venv there (see docs/deployment.md step 1)"
  fi

  if [ -f "$MODEL_DIR/config.json" ]; then
    ok "main model found: $MODEL_DIR"
  else
    warn "main model NOT found at $MODEL_DIR (services will fail to start until it is placed)"
  fi
  if [ -f "$DRAFT_DIR/config.json" ]; then
    ok "draft model found: $DRAFT_DIR"
  else
    warn "draft model NOT found at $DRAFT_DIR (speculative decoding will fail)"
  fi

  if [ "$WITH_PREFIX_WARM" = "1" ] && [ ! -f /root/wb_warmup_payload.json ]; then
    warn "/root/wb_warmup_payload.json missing - prefix warm-up will fail; use --no-prefix-warm or add the payload"
  fi

  if [ "$WITH_FAN_CURVE" = "1" ] && [ ! -f /etc/X11/xorg-nvidia-fan.conf ]; then
    warn "/etc/X11/xorg-nvidia-fan.conf missing - the fan curve needs Xorg :1; use --no-fan-curve if headless"
  fi
  echo

  confirm "proceed with installation?" || { say "aborted"; exit 0; }

  # ---- 1. helper scripts ----------------------------------------------
  say "1/6 installing helper scripts into $BIN_DIR"
  run mkdir -p "$BIN_DIR"
  for s in gpu-power-limit.sh gpu-fan-curve.sh vllm-warmup.sh wb-prefix-warm.sh; do
    [ -f "$SCRIPT_DIR/scripts/$s" ] && run install -m 0755 "$SCRIPT_DIR/scripts/$s" "$BIN_DIR/$s"
  done
  [ -f "$SCRIPT_DIR/scripts/wb-prefix-warm.py" ] && run install -m 0755 "$SCRIPT_DIR/scripts/wb-prefix-warm.py" "$BIN_DIR/wb-prefix-warm.py"
  ok "scripts installed"

  # ---- 2. systemd units ------------------------------------------------
  say "2/6 installing systemd units"
  local install_list=("vllm-qwen38.service" "vllm-warmup.service")
  [ "$POWER_LIMIT" != "0" ] && install_list+=("gpu-power-limit.service")
  [ "$WITH_FAN_CURVE" = "1" ] && install_list+=("gpu-fan-curve.service")
  if [ "$WITH_PREFIX_WARM" = "1" ]; then
    install_list+=("wb-prefix-warm.service" "wb-prefix-warm.timer")
  fi
  for u in "${install_list[@]}"; do
    if [ -f "$SCRIPT_DIR/systemd/$u" ]; then
      install_unit "$SCRIPT_DIR/systemd/$u" "$u"
      ok "$u"
    else
      warn "unit not found in repo: $u"
    fi
  done

  # ---- 3. monitoring (optional) ---------------------------------------
  if [ "$WITH_MONITORING" = "1" ]; then
    say "3/6 installing monitoring stack"
    run mkdir -p /etc/prometheus /etc/grafana/provisioning
    for u in "${MON_UNITS[@]}"; do
      [ -f "$SCRIPT_DIR/monitoring/units/$u" ] && install_unit "$SCRIPT_DIR/monitoring/units/$u" "$u"
    done
    [ -f "$SCRIPT_DIR/monitoring/prometheus.yml" ] && run install -m 0644 "$SCRIPT_DIR/monitoring/prometheus.yml" /etc/prometheus/prometheus.yml
    if [ -d "$SCRIPT_DIR/monitoring/grafana-provisioning" ]; then
      run cp -r "$SCRIPT_DIR/monitoring/grafana-provisioning/." /etc/grafana/provisioning/
    fi
    if [ -f /etc/grafana/grafana.ini ]; then
      warn "/etc/grafana/grafana.ini exists - left untouched. Template: $SCRIPT_DIR/monitoring/grafana.ini.sanitized"
      warn "  (remember to set your own admin_password / secret_key)"
    else
      [ -f "$SCRIPT_DIR/monitoring/grafana.ini.sanitized" ] && \
        run install -m 0640 "$SCRIPT_DIR/monitoring/grafana.ini.sanitized" /etc/grafana/grafana.ini
      warn "grafana.ini installed from the sanitized template - edit admin_password / secret_key before use"
    fi
    ok "monitoring configured"
  else
    say "3/6 monitoring skipped (add --with-monitoring to install it)"
  fi

  # ---- 4. daemon-reload -----------------------------------------------
  say "4/6 systemctl daemon-reload"
  run systemctl daemon-reload

  # ---- 5. enable / start ----------------------------------------------
  say "5/6 enabling services"
  local enable_list=("vllm-qwen38.service")
  [ "$POWER_LIMIT" != "0" ] && enable_list+=("gpu-power-limit.service")
  [ "$WITH_FAN_CURVE" = "1" ] && enable_list+=("gpu-fan-curve.service")
  [ "$WITH_PREFIX_WARM" = "1" ] && enable_list+=("wb-prefix-warm.timer")
  [ "$WITH_MONITORING" = "1" ] && enable_list+=("prometheus.service" "grafana-server.service" "node_exporter.service" "nvidia_gpu_exporter.service")

  for u in "${enable_list[@]}"; do
    if [ "$DO_START" = "1" ]; then
      run systemctl enable --now "$u" 2>/dev/null || warn "enable --now $u failed"
    else
      run systemctl enable "$u" 2>/dev/null || warn "enable $u failed"
    fi
    ok "$u"
  done

  if [ "$DO_START" != "1" ]; then
    warn "services are enabled but NOT started - run:  systemctl start vllm-qwen38.service"
    warn "  (first start compiles/warms kernels; TimeoutStartSec is 1800s, this is normal)"
  fi

  # ---- 6. summary ------------------------------------------------------
  say "6/6 done"
  cat <<EOF

  installed units : ${install_list[*]}
  helper scripts  : $BIN_DIR/{gpu-power-limit.sh,gpu-fan-curve.sh,vllm-warmup.sh,wb-prefix-warm.sh,wb-prefix-warm.py}
  logs            : journalctl -u vllm-qwen38.service -f
                    $VLLM_DIR/run-logs/systemd-vllm.log
  health check    : curl -s http://127.0.0.1:8000/v1/models
  verify          : sudo $0 --verify
  rollback        : sudo $0 --uninstall

EOF
}

# ---------------------------------------------------------------- main
case "1" in
  1) : ;;
esac

if [ "$DO_STATUS" = "1" ]; then do_status; exit 0; fi
if [ "$DO_VERIFY" = "1" ]; then do_verify; exit $?; fi
if [ "$DO_UNINSTALL" = "1" ]; then do_uninstall "$@"; exit $?; fi
do_install "$@"
