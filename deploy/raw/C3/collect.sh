#!/bin/bash
# read-only host inventory collector for model host
echo "===== uname ====="; uname -a
echo "===== lsb_release ====="; lsb_release -a 2>&1
echo "===== systemd_version ====="; systemd --version | head -1
echo "===== uptime ====="; uptime
echo "===== lscpu ====="; lscpu
echo "===== free_g ====="; free -g
echo "===== swapon ====="; swapon --show
echo "===== meminfo_head ====="; head -5 /proc/meminfo
echo "===== df_h ====="; df -h
echo "===== lsblk ====="; lsblk -o NAME,SIZE,ROTA,TYPE,MOUNTPOINT
echo "===== nvidia_smi ====="; nvidia-smi
echo "===== nvidia_smi_q ====="; nvidia-smi -q
echo "===== nvlink ====="; nvidia-smi nvlink -s 2>&1
echo "===== topo_m ====="; nvidia-smi topo -m
echo "===== nvcc ====="; which nvcc; nvcc --version 2>&1
echo "===== python3 ====="; python3 -V; pip3 list 2>/dev/null | grep -Ei 'torch|vllm|transformers|flashinfer|xformers|numpy'
echo "===== ip ====="; ip -br a
echo "===== ss ====="; ss -lntp 2>/dev/null | head -20
echo "===== du_models ====="; du -sh /root/models/* 2>/dev/null
echo "===== vllm_version ====="; cat /root/vLLM-2080Ti-Definitive-main/VERSION 2>&1; cat /root/vLLM-2080Ti-Definitive-main/PROJECT_RELEASE.env 2>&1
echo "===== vllm_git ====="; git -C /root/vLLM-2080Ti-Definitive-main log -1 --format='%H%n%ci%n%s' 2>&1
echo "===== venv_locate ====="; ls -d /root/vLLM-2080Ti-Definitive-main/.venv /root/vLLM-2080Ti-Definitive-main/*venv* /root/.venv* /root/venv* 2>/dev/null
echo "===== unit_execstart ====="; systemctl cat vllm-qwen38.service 2>&1
echo "===== running_services ====="; systemctl list-units --type=service --state=running | grep -Ei 'vllm|gpu|prometheus|grafana|exporter'
echo "===== top ====="; top -bn1 | head -15
echo "===== ethtool ====="; for i in $(ls /sys/class/net | grep -v lo); do echo "-- $i"; ethtool "$i" 2>/dev/null | grep -Ei 'Speed|Duplex|Link detected'; done
echo "===== END ====="
