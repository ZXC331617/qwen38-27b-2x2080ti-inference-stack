# 生产环境变量清单（脱敏）

- 采集方式：主机只读。`systemctl cat <unit>`（含 drop-in，本机无 drop-in）+ `cat /etc/environment` + `grep` `/root/.bashrc`、`/root/.profile`、`/etc/profile.d/*` + `/proc/<MainPID>/environ`
- 原始输出：`deploy/env/raw/systemctl-cat-env-units.txt`、`deploy/env/raw/env-sources.txt`
- 脱敏规则：token / api key / 口令 / HF token → `TOKEN` / `PASSWORD`；公网 IP / 外部域名 → `HOST`；内网 `192.168.x.x` 原样保留
- 实测结论：**本批生产单元中不含任何 token / 口令类环境变量**，因此下表 0 处被占位；唯一外部域名 `HF_ENDPOINT` 是公开镜像站，按规则保留原值

## 1. `vllm-qwen38.service`（`/etc/systemd/system/vllm-qwen38.service`，28 个 Environment=）

| 变量名 | 值 | 作用说明 | 来源 |
|-------|----|---------|------|
| `HOME` | `/root` | 服务运行用户主目录；模型/缓存相对路径解析依赖 | vllm-qwen38.service |
| `LANG` | `zh_CN.UTF-8` | 中文日志与文件名编码 | 同上 |
| `LANGUAGE` | `zh_CN:zh` | gettext 语言回退链 | 同上 |
| `XDG_DATA_DIRS` | `/usr/share/gnome:/usr/local/share:/usr/share:/var/lib/snapd/desktop` | 桌面数据目录（从登录会话带入，对推理无影响） | 同上 |
| `PATH` | `/root/vLLM-2080Ti-Definitive-main/.venv/bin:/usr/local/cuda-13.0/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin` | 让 venv python、nvcc 优先；子进程（ninja/nvcc/gcc）编译扩展时用 | 同上 |
| `CUDACXX` | `/usr/local/cuda-13.0/bin/nvcc` | Torch / Triton 编译 CUDA 扩展时的 nvcc 路径 | 同上 |
| `CUDA_HOME` | `/usr/local/cuda-13.0` | CUDA 工具链根目录 | 同上 |
| `CUDA_PATH` | `/usr/local/cuda-13.0` | 同上（部分构建脚本读这个名） | 同上 |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` | CUDA 设备编号按 PCI 总线顺序，保证与 `nvidia-smi` 顺序一致 | 同上 |
| `CUDA_VISIBLE_DEVICES` | `0,1` | 只暴露两张 RTX 2080 Ti，正好对应 `--tensor-parallel-size 2` | 同上 |
| `FLASHQLA_ROOT` | `/root/vLLM-2080Ti-Definitive-main/.deps/FlashQLA-SM70-SM75` | 自编译 FlashQLA（SM70/SM75 专用 GDN prefill 后端）根目录 | 同上 |
| `PYTHONPATH` | `/root/vLLM-2080Ti-Definitive-main:/root/vLLM-2080Ti-Definitive-main/.deps/FlashQLA-SM70-SM75` | 让 vLLM 与 FlashQLA 包可被导入 | 同上 |
| `PYTHONSAFEPATH` | `1` | 禁止把脚本所在目录隐式加入 `sys.path`，避免误导入 | 同上 |
| `PYTHONUNBUFFERED` | `1` | Python 输出不缓冲，systemd 日志实时可见 | 同上 |
| `TORCH_CUDA_ARCH_LIST` | `7.5` | 只编译 sm_75 内核（2080 Ti 是 Turing sm_75），显著缩短 JIT 编译时间 | 同上 |
| `TORCH_EXTENSIONS_DIR` | `/root/vLLM-2080Ti-Definitive-main/.deps/FlashQLA-SM70-SM75/.torch_extensions_vllm_flashqla_legacy` | Torch C++/CUDA 扩展编译缓存目录，避免每次启动重编 | 同上 |
| `TORCHINDUCTOR_CACHE_DIR` | `/root/vLLM-2080Ti-Definitive-main/torchinductor-cache` | torch.compile / inductor 编译缓存目录 | 同上 |
| `TRITON_CACHE_DIR` | `/root/vLLM-2080Ti-Definitive-main/triton-cache` | Triton kernel 缓存目录（重启后不必重编译） | 同上 |
| `FLASHINFER_ENABLE_AOT` | `1` | 启用 FlashInfer 预编译（AOT）内核，跳过运行时 JIT | 同上 |
| `FLASHINFER_WORKSPACE_BASE` | `/root/vLLM-2080Ti-Definitive-main` | FlashInfer JIT 工作区基目录 | 同上 |
| `VLLM_DISABLE_TILELANG` | `1` | 关闭 TileLang 后端（sm_75 上不可用/不稳定） | 同上 |
| `VLLM_SM75_SPEC_SYNC_MODE` | `safe` | 投机解码在 sm_75 上走保守同步路径，规避竞态 | 同上 |
| `VLLM_KV_CACHE_LAYOUT` | `BLHNC` | KV cache 内存布局为 batch/len/head/dim/block | 同上 |
| `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH` | `1` | 允许 Mamba/GDN 层在投机解码时进入整图 CUDA Graph | 同上 |
| `VLLM_ENFORCE_STRICT_TOOL_CALLING` | `0` | 放宽工具调用解析校验（配合 `qwen3_xml` parser 容错） | 同上 |
| `VLLM_KV_BUDGET_FROM_FREE_MEMORY` | `1` | KV 预算按启动时空闲显存推算，而非按总显存比例 | 同上 |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | `1` | Marlin W4A16 kernel 使用 atomicAdd 累加，sm_75 上更快更稳 | 同上 |
| `VLLM_KV_BUDGET_REDUNDANCY_MIB` | `150` | KV 预算额外预留 150 MiB 冗余，防 OOM | 同上 |

## 2. `vllm-warmup.service`（4 个）

| 变量名 | 值 | 作用说明 | 来源 |
|-------|----|---------|------|
| `WARMUP_BASE` | `http://127.0.0.1:8000` | 预热请求打向的 vLLM OpenAI 兼容端点 | vllm-warmup.service |
| `WARMUP_MODEL` | `Qwen3.8-27B-Uncensored-AWQ` | 预热用的 served-model-name | 同上 |
| `WARMUP_PY` | `/root/vLLM-2080Ti-Definitive-main/.venv/bin/python` | 预热脚本使用的解释器 | 同上 |
| `WARMUP_SIZES` | `1024 16384 32768 65536` | 四档上下文长度，逐档各发一次请求，把对应 kernel 编译/缓存打热（注释记录整轮约 213s） | 同上 |

## 3. `gpu-power-limit.service`（1 个）

| 变量名 | 值 | 作用说明 | 来源 |
|-------|----|---------|------|
| `GPU_POWER_LIMIT_W` | `225` | 两张 2080 Ti 的功耗上限（瓦），oneshot 脚本 `gpu-power-limit.sh` 读取后 `nvidia-smi -pl` 固化 | gpu-power-limit.service |

## 4. `gpu-fan-curve.service`（2 个）

| 变量名 | 值 | 作用说明 | 来源 |
|-------|----|---------|------|
| `FAN_DISPLAY` | `:1` | 风扇曲线所依赖的 X server 显示号（`nvidia-settings` 需要） | gpu-fan-curve.service |
| `FAN_INTERVAL` | `10` | 风扇转速调节间隔（秒） | 同上 |

## 5. `wb-prefix-warm.service`（4 个）

| 变量名 | 值 | 作用说明 | 来源 |
|-------|----|---------|------|
| `WB_WARMUP_BASE` | `http://127.0.0.1:8000` | WorkBuddy 前缀缓存预热的目标端点 | wb-prefix-warm.service |
| `WB_WARMUP_MODEL` | `Qwen3.8-27B-Uncensored-AWQ` | 预热使用的 served-model-name | 同上 |
| `WB_WARMUP_PAYLOAD` | `/root/wb_warmup_payload.json` | 预置的 tools+system 前缀请求体（重放以命中 prefix cache） | 同上 |
| `WB_WARMUP_PY` | `/root/vLLM-2080Ti-Definitive-main/.venv/bin/python` | 预热脚本解释器 | 同上 |

## 6. shell / 系统级

| 变量名 | 值 | 作用说明 | 来源 |
|-------|----|---------|------|
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 走国内镜像站，加速权重下载（公开镜像，非机密） | `/root/.bashrc` 第 103 行；`/etc/profile.d/hf-mirror.sh` 第 1 行 |
| `PATH` | `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin` | 系统默认 PATH；仅作用于登录会话，systemd 服务不继承 | `/etc/environment` |
| `LANG` | `zh_CN.UTF-8` | systemd manager 环境（`systemctl show-environment`） | systemd manager |
| `LANGUAGE` | `zh_CN:zh` | 同上 | systemd manager |
| `PATH` | `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin` | systemd manager 的 PATH（注意比 `/etc/environment` 少了 games 两项） | systemd manager |

### 未发现项（已确认）

- `/etc/default/vllm*`：不存在
- `/etc/default/gpu*`、`/etc/default/cuda*`：不存在
- `/root/.profile`：无 `VLLM_*` / `CUDA_*` / `HF_*` / `NCCL_*` 导出
- 全部单元与 shell 配置中：**无 `NCCL_*` 变量**（TP=2 走默认 NCCL 行为，未做手工调优）

## 7. `grafana-server.service` 的 `EnvironmentFile=/etc/default/grafana-server`（11 个）

| 变量名 | 值 | 作用说明 |
|-------|----|---------|
| `GRAFANA_USER` | `grafana` | 运行用户 |
| `GRAFANA_GROUP` | `grafana` | 运行组 |
| `GRAFANA_HOME` | `/usr/share/grafana` | 安装根目录 |
| `LOG_DIR` | `/var/log/grafana` | 日志目录（经 `cfg:default.paths.logs` 传入） |
| `DATA_DIR` | `/var/lib/grafana` | 数据目录（sqlite 库、插件、仪表盘） |
| `MAX_OPEN_FILES` | `10000` | 文件句柄上限 |
| `CONF_DIR` | `/etc/grafana` | 配置目录 |
| `CONF_FILE` | `/etc/grafana/grafana.ini` | 主配置文件（即已脱敏的 `grafana.ini`） |
| `RESTART_ON_UPGRADE` | `true` | 包升级后重启 |
| `PLUGINS_DIR` | `/var/lib/grafana/plugins` | 插件目录 |
| `PROVISIONING_CFG_DIR` | `/etc/grafana/provisioning` | provisioning 目录（已整目录拉取） |
| `PID_FILE_DIR` | `/run/grafana` | pid 文件目录 |

## 8. vLLM 启动参数（`ExecStart`，非环境变量，但属生产参数，一并说明）

| 参数 | 值 | 含义 |
|------|----|------|
| `--host` / `--port` | `0.0.0.0` / `8000` | OpenAI 兼容 API 监听地址 |
| `--model` | `/root/models/Coder390-HUHUI-INC-A-W4A16` | 实际加载的权重路径 |
| `--served-model-name` | `Coder390-HUHUI-INC-A-W4A16 Qwen3.8-27B-Coder390-Abliterated-W4A16 Qwen3.8-27B-Uncensored-AWQ` | 对外暴露的 3 个别名（旧名保留以兼容调用方） |
| `--dtype` | `half` | 计算精度 fp16（2080 Ti 无 bf16 原生支持） |
| `--quantization` | `compressed-tensors` | W4A16 压缩张量量化格式 |
| `--tensor-parallel-size` | `2` | 两卡张量并行 |
| `--generation-config` | `auto` | 从模型目录读 generation_config.json |
| `--override-generation-config` | `{"max_new_tokens": 24576}` | 覆盖默认最大生成长度 |
| `--gpu-memory-utilization` | `0.95` | 显存占用上限比例 |
| `--max-model-len` | `225280` | 最大上下文长度 |
| `--enable-chunked-prefill` | — | 分块 prefill，降低首 token 阻塞 |
| `--max-num-seqs` | `1` | 单批并发序列数（单用户/单流场景） |
| `--max-num-batched-tokens` | `16384` | 单批 token 上限 |
| `--kv-cache-dtype` | `fp8` | KV cache 用 fp8 存储，省显存 |
| `--mamba-cache-mode` | `align` | GDN/Mamba 状态缓存对齐模式 |
| `--enable-prefix-caching` | — | 启用前缀缓存（配合 wb-prefix-warm 预热） |
| `--enable-prompt-tokens-details` | — | 返回 prompt token 明细 |
| `--reasoning-parser` | `qwen3` | 解析 Qwen3 的 reasoning 段 |
| `--limit-mm-per-prompt` | `{"image":64,"video":0,"audio":0}` | 多模态输入限额 |
| `--enable-auto-tool-choice` | — | 启用自动工具选择 |
| `--tool-call-parser` | `qwen3_xml` | 工具调用解析器 |
| `--additional-config` | `{"gdn_prefill_backend":"flashqla_legacy"}` | GDN prefill 走后端 FlashQLA legacy |
| `--per-request-spec-decode-metrics` | `detailed` | 投机解码按请求输出详细指标 |
| `--speculative-config` | `{"method":"dflash","num_speculative_tokens":7,"model":"/root/models/Qwen3.8-27B-DFlash2-W4A16","attention_backend":"TRITON_ATTN","kv_cache_dtype":"float16","draft_sample_method":"probabilistic"}` | 投机解码：DFlash2 草稿模型、7 个投机 token、草稿用 TRITON_ATTN + fp16 KV |
| `--compilation-config` | `{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8],"max_cudagraph_capture_size":8}` | CUDA Graph 捕获策略与尺寸 |

## 9. 脱敏说明

- 本节所列变量中**没有任何 token / api key / 口令**，故无占位符替换。
- `HF_ENDPOINT=https://hf-mirror.com` 是公开镜像域名，非机密，按规则保留。
- 全流程未出现公网 IP；`127.0.0.1` 为回环地址，`192.168.x.x` 未出现。
