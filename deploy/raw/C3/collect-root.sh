#!/bin/bash
# read-only root-level inventory (run under sudo)
echo "===== du_models ====="; du -sh /root/models/* 2>&1
echo "===== vllm_VERSION ====="; cat /root/vLLM-2080Ti-Definitive-main/VERSION 2>&1
echo "===== vllm_RELEASE_env ====="; cat /root/vLLM-2080Ti-Definitive-main/PROJECT_RELEASE.env 2>&1
echo "===== vllm_git ====="; git -C /root/vLLM-2080Ti-Definitive-main log -1 --format='%H%n%ci%n%s' 2>&1
echo "===== venv_locate ====="; ls -d /root/vLLM-2080Ti-Definitive-main/.venv /root/vLLM-2080Ti-Definitive-main/*venv* /root/.venv* /root/venv* /root/vllm-venv* 2>&1
echo "===== venv_python_list ====="; ls -l /root/vLLM-2080Ti-Definitive-main/.venv/bin/python* 2>&1
echo "===== torch_probe ====="
for P in /root/vLLM-2080Ti-Definitive-main/.venv/bin/python /root/vllm-venv/bin/python /root/venv/bin/python /root/.venv/bin/python; do
  if [ -x "$P" ]; then echo "-- $P"; "$P" -c "import sys,torch;print(sys.version.split()[0]);print(torch.__version__);print(torch.version.cuda);print(torch.cuda.get_device_capability());print(torch.cuda.device_count());print(torch.cuda.get_device_name(0))" 2>&1; "$P" -c "import vllm;print('vllm',vllm.__version__)" 2>&1; fi
done
echo "===== nvcc ====="; ls -l /usr/local/cuda*/bin/nvcc 2>&1; /usr/local/cuda/bin/nvcc --version 2>&1
echo "===== ethtool ====="; for i in $(ls /sys/class/net | grep -v lo); do echo "-- $i"; ethtool "$i" 2>&1 | grep -Ei 'Speed|Duplex|Link detected'; done
echo "===== model_configs ====="
for D in /root/models/Coder390-HUHUI-INC-A-W4A16 /root/models/Qwen3.8-27B-DFlash2-W4A16; do
  echo "---- $D"; ls -l "$D" 2>&1 | head -20
  echo "-- config.json head"; head -c 3000 "$D/config.json" 2>&1
  echo; echo "-- generation_config.json"; cat "$D/generation_config.json" 2>&1 | head -40
done
echo "===== models_dir ====="; ls -l /root/models/ 2>&1
echo "===== END ====="
