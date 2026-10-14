# MANIFEST - systemd units, operator scripts, version/model configs

Source host: `ubuntu@192.168.5.88` (read-only pull; `/root` via sudo).
Generated: 2026-10-14 20:31:24 +08:00
Files: 78  (production=6, experimental=15, backup=46, script=5, version=2, model-config=4)

All rows verified: local byte count == host `stat -c %s`, local sha256 == host `sha256sum` (see `deploy/raw/C1/verify.txt`).

## Production units (`deploy/systemd/`)

| file | kind | key params | host path | bytes | sha256 |
|---|---|---|---|---|---|
| `deploy\systemd\vllm-qwen38.service` | production | PROD inference: port 8000; TP=2; ctx=225280; model=/root/models/Coder390-HUHUI-INC-A-W4A16; dtype=half; quant=compressed-tensors; kv-cache=fp8; max-num-seqs=1; max-num-batched-tokens=16384; gpu-mem-util=0.95; spec=dflash n=7 draft=/root/models/Qwen3.8-27B-DFlash2-W4A16 (probabilistic); cudagraph FULL_AND_PIECEWISE cap=8; ExecStartPost -> vllm-warmup.service | `/etc/systemd/system/vllm-qwen38.service` | 3792 | `85642d8ebe4389cc025473190a575bfdb12fea26f88b8c5bf7029245de08b741` |
| `deploy\systemd\vllm-warmup.service` | production | PROD kernel warmup (oneshot): WARMUP_BASE=http://127.0.0.1:8000; WARMUP_MODEL=Qwen3.8-27B-Uncensored-AWQ; WARMUP_SIZES=1024 16384 32768 65536; ExecStart=/usr/local/bin/vllm-warmup.sh; ExecStartPost -> wb-prefix-warm.service | `/etc/systemd/system/vllm-warmup.service` | 1493 | `03ac33824643c23cd3eb0a08fc52eccad8ae24e5a30a3a5fe849787917c0a917` |
| `deploy\systemd\gpu-power-limit.service` | production | PROD GPU power lock (oneshot): GPU_POWER_LIMIT_W=225; ExecStart=/usr/local/bin/gpu-power-limit.sh | `/etc/systemd/system/gpu-power-limit.service` | 387 | `268d840e7813235d951228b6f33656d1ebf4ffbc0346d1b15ebffbd62c9fa45d` |
| `deploy\systemd\gpu-fan-curve.service` | production | PROD GPU fan curve (oneshot): ExecStartPre spawns Xorg :1 (-config /etc/X11/xorg-nvidia-fan.conf); ExecStart=/usr/local/bin/gpu-fan-curve.sh | `/etc/systemd/system/gpu-fan-curve.service` | 1191 | `d7826887b984e711cc3e7a38d5810a7bc6a2f431c5bbaf65031c6fd8bf26a925` |
| `deploy\systemd\wb-prefix-warm.service` | production | PROD WorkBuddy prefix-cache warmup (oneshot): WB_WARMUP_BASE=http://127.0.0.1:8000; WB_WARMUP_MODEL=Qwen3.8-27B-Uncensored-AWQ; WB_WARMUP_PAYLOAD=/root/wb_warmup_payload.json; ExecStart=/usr/local/bin/wb-prefix-warm.sh | `/etc/systemd/system/wb-prefix-warm.service` | 918 | `67116d6d78c878dde54d27ad3cdc1fb9a49ae69a35d4e0e6b42e929a268068e1` |
| `deploy\systemd\wb-prefix-warm.timer` | production | PROD timer: OnCalendar=*-*-* *:00,10,20,30,40,50:00; OnUnitActiveSec=10min; Unit=wb-prefix-warm.service | `/etc/systemd/system/wb-prefix-warm.timer` | 207 | `e2195f918937df9c939f2594596925e03d63dccf42597b05a97c6eb733c8ce58` |
| `deploy\scripts\gpu-power-limit.sh` | script | Script: nvidia-smi -pl 225 for both GPUs (target of gpu-power-limit.service) | `/usr/local/bin/gpu-power-limit.sh` | 376 | `bf3153119f1106535f57ce1ebb14539db51c864a8d71106be9aea8569b00d20e` |
| `deploy\scripts\gpu-fan-curve.sh` | script | Script: closed-loop fan curve via nvidia-settings -c :1 (target of gpu-fan-curve.service) | `/usr/local/bin/gpu-fan-curve.sh` | 1843 | `59464a4f0dc23374e727751e494203b152d3bb1e09d2b3c59f4ddedcb24b3913` |
| `deploy\scripts\vllm-warmup.sh` | script | Script: POST /v1/chat/completions at 4 context sizes (target of vllm-warmup.service) | `/usr/local/bin/vllm-warmup.sh` | 2670 | `33784a85594c6306648b8605ca2b607c4c0e703b9f9b632796645f110625cb71` |
| `deploy\scripts\wb-prefix-warm.py` | script | Script: replay WorkBuddy cacheable prefix (system+tools) to warm vLLM prefix cache | `/usr/local/bin/wb-prefix-warm.py` | 2023 | `ab5858dacf21201e7166435c5f9b9caca6c49f17a9a3d28ac3d833b7a11fe184` |
| `deploy\scripts\wb-prefix-warm.sh` | script | Script: wrapper that invokes wb-prefix-warm.py (target of wb-prefix-warm.service) | `/usr/local/bin/wb-prefix-warm.sh` | 86 | `5b539b5dffb4de5122501ba63a9c34bd5c6224228e191355c95d33643221d73b` |
| `deploy\versions\VERSION` | version | Fork release version string | `/root/vLLM-2080Ti-Definitive-main/VERSION` | 12 | `380c0ec996114daaa015c1cd9f727450155367e1348352b04c8fee8c874f4d73` |
| `deploy\versions\PROJECT_RELEASE.env` | version | Fork release metadata (base vLLM commit, CUDA/torch pins) | `/root/vLLM-2080Ti-Definitive-main/PROJECT_RELEASE.env` | 1054 | `c8a1d77a84a035de207b246fa52762a95a28498c188cbb072ac186adb46a1b9d` |
| `deploy\models\Coder390-HUHUI-INC-A-W4A16\config.json` | model-config | PROD model config: Qwen3_5ForConditionalGeneration; W4A16 pack-quantized group=128 sym int4; ignore visual+lm_head+mtp | `/root/models/Coder390-HUHUI-INC-A-W4A16/config.json` | 9578 | `0f4ce583e39e2f18f864b5692064993b382a73f9800ac196e17fe070b1447cf8` |
| `deploy\models\Coder390-HUHUI-INC-A-W4A16\generation_config.json` | model-config | PROD model generation defaults: temp=1.0 top_k=20 top_p=0.95 | `/root/models/Coder390-HUHUI-INC-A-W4A16/generation_config.json` | 214 | `df6f86c3fdce573ecdb55cec35f502ddda70e70abe78288a78164aef120232e7` |
| `deploy\models\Coder390-HUHUI-INC-A-W4A16\recipe.yaml` | model-config | Quantization recipe: QuantizationModifier W4A16 on Linear | `/root/models/Coder390-HUHUI-INC-A-W4A16/recipe.yaml` | 263 | `46a02ffa9993c19bf8a7ec8cccc00170b381e71e7380f9e87ad8fca67cb1d1bc` |
| `deploy\models\Qwen3.8-27B-DFlash2-W4A16\config.json` | model-config | Draft model config: DFlash2DraftModel (5 layers, hidden 5120, block_size 8, target_layers 5/19/33/47/61, W4A16) | `/root/models/Qwen3.8-27B-DFlash2-W4A16/config.json` | 2238 | `9cf9e16f08733cc4ff2ba085e9b2994e720ea9ad50ae175e26b07628f8d0be62` |

## Experimental units (`deploy/systemd/experimental/`)

| file | kind | key params | host path | bytes | sha256 |
|---|---|---|---|---|---|
| `deploy\systemd\experimental\vllm-coder390-abit2-fast.service` | experimental | EXP: port 8003; model=...INT4-W4A16-ABIT2; TP=2; ctx=16384; max-num-seqs=16; max-num-batched-tokens=8192; no spec | `/etc/systemd/system/vllm-coder390-abit2-fast.service` | 2380 | `8852bf7d50c5101a358712850e49465feaf8d0a88d33de706a2d05ea32861fc6` |
| `deploy\systemd\experimental\vllm-coder390-abit60.service` | experimental | EXP: port 8002; model=...INT4-W4A16-ABLIT60; TP=2; ctx=32768; max-num-seqs=1; no spec | `/etc/systemd/system/vllm-coder390-abit60.service` | 2334 | `aae7f21de0c75e8e51d40d646e5d40052769548f43acfcc7ef3eaf3a41914473` |
| `deploy\systemd\experimental\vllm-coder390-abit-fast.service` | experimental | EXP: port 8003; model=...INT4-W4A16-ABLIT; TP=2; ctx=16384; max-num-seqs=16; max-num-batched-tokens=8192; no spec | `/etc/systemd/system/vllm-coder390-abit-fast.service` | 2378 | `94147e470293914f182471abed8cf0eab4c1f55dbeaf236aa39c3016f19f4680` |
| `deploy\systemd\experimental\vllm-coder390-abit.service` | experimental | EXP: port 8001; model=Qwen3.8-27B-Coder390-INT4-W4A16-ABLIT; TP=2; ctx=32768; max-num-seqs=1; no speculation | `/etc/systemd/system/vllm-coder390-abit.service` | 3111 | `0594b84d8a639cc1261bcffc1c8902289b5145df4d702575dea7401461a741d0` |
| `deploy\systemd\experimental\vllm-coder390-bigeval.service` | experimental | EXP: port 8004; model=${EVAL_MODEL} (EnvironmentFile); TP=2; ctx=16384; max-num-seqs=64; max-num-batched-tokens=16384 | `/etc/systemd/system/vllm-coder390-bigeval.service` | 2408 | `dd9767651408f857b186e6f887385a09a2ce640dcb6c1a91c24da78631aa94cc` |
| `deploy\systemd\experimental\vllm-coder390-dflash-blh8k.service` | experimental | EXP: port 8001; model=...INT4-W4A16; TP=2; ctx=262144; kv=fp8; batch=8192; spec=dflash n=7 draft=Qwen3.8-27B-DFlash2-W4A16 (greedy); BLHNC | `/etc/systemd/system/vllm-coder390-dflash-blh8k.service` | 3192 | `a0f496b43239a7a0a5772d838ad83c766062d4c029e88ec08808dc0ff24da204` |
| `deploy\systemd\experimental\vllm-coder390-dflash.service` | experimental | EXP: port 8001; model=Qwen3.8-27B-Coder390-INT4-W4A16; TP=2; ctx=262144; kv=fp8; batch=2048; spec=dflash n=7 draft=Qwen3.8-27B-Coder390-DFlash2-FP8 (greedy); LBNHC | `/etc/systemd/system/vllm-coder390-dflash.service` | 3196 | `1ac5bb829883362c811419decd8a66f38ec0d472c41929dde51cdc511ddf82e0` |
| `deploy\systemd\experimental\vllm-coder390-dflash-w4a16.service` | experimental | EXP: port 8001; model=...INT4-W4A16; TP=2; ctx=262144; kv=fp8; batch=2048; spec=dflash n=7 draft=Qwen3.8-27B-DFlash2-W4A16 (greedy) | `/etc/systemd/system/vllm-coder390-dflash-w4a16.service` | 3195 | `eff6fc74e6479334f19e3b34b87dc6bc7457f2844d0b7bf98cacb3544968dbab` |
| `deploy\systemd\experimental\vllm-coder390-eval.service` | experimental | EXP: port 8003; model=${EVAL_MODEL}; TP=2; ctx=16384; max-num-seqs=16; batch=8192; no spec | `/etc/systemd/system/vllm-coder390-eval.service` | 2413 | `a53a1b1dcad4523478e19e648d4b99058688d19558b473a4febdeaf4d8e0105a` |
| `deploy\systemd\experimental\vllm-coder390-eval-tp1.service` | experimental | EXP: port 8005; model=${EVAL_MODEL}; TP=1 (GPU1 only); ctx=4096; max-num-seqs=8; batch=4096 | `/etc/systemd/system/vllm-coder390-eval-tp1.service` | 2466 | `f2fed69e9e44c460034dcb35ad6d98010adb4454dd69de4a8999bc657b3a8804` |
| `deploy\systemd\experimental\vllm-coder390-kern@.service` | experimental | EXP template (@linear-backend): port 8001; model=...INT4-W4A16; TP=2; ctx=262144; kv=fp8; batch=2048 | `/etc/systemd/system/vllm-coder390-kern@.service` | 2914 | `75b9175b870fda2e53e836c72beeb856bd757aa5774f2ec869ff3b763cad7f4e` |
| `deploy\systemd\experimental\vllm-coder390-mtp.service` | experimental | EXP: port 8001; model=...INT4-W4A16; TP=2; ctx=262144; kv=fp8; batch=2048; spec=mtp n=3; LBNHC | `/etc/systemd/system/vllm-coder390-mtp.service` | 3201 | `d8bc8cbd7369ed67d4bb8010e725bf2c2f5ec427356df4587f7f42bb09c0af40` |
| `deploy\systemd\experimental\vllm-coder390-test.service` | experimental | EXP: port 8001; model=...INT4-W4A16; TP=2; ctx=262144; kv=fp8; batch=2048; production-identical args, no speculation | `/etc/systemd/system/vllm-coder390-test.service` | 3258 | `92993b2f17a65754bd03808623d9f30578a7b74402382e1df82fa604cb337350` |
| `deploy\systemd\experimental\vllm-prefilltest.service` | experimental | EXP: port 8000; model=/root/models/Huihui-Coder390-Abliterated-W4A16; served-name=Qwen3.8-27B-Uncensored-AWQ; TP=2; ctx=262144; kv=fp8; batch=16384; spec=dflash n=7 (greedy) | `/etc/systemd/system/vllm-prefilltest.service` | 3607 | `08e9b53932dc83cbbe03a117126bc1f7d03221497808b296294b88be58bc9d02` |
| `deploy\systemd\experimental\vllm-uncensored-et.service` | experimental | EXP: port 8001; model=Qwen3.8-27B-EfficientThink-Uncensored-AWQ-W4A16; TP=2; ctx=262144; kv=fp8; batch=2048; spec=mtp n=3; LBNHC | `/etc/systemd/system/vllm-uncensored-et.service` | 3335 | `3289e608bc8518ae7c291d7b7e5741408b60910f6ac82029c0654e60b8a8670c` |

## Backups (`deploy/systemd/backups/`)

| file | host path | bytes | sha256 |
|---|---|---|---|
| `vllm-qwen38.service.bak-attn-20261004-134709` | `/etc/systemd/system/vllm-qwen38.service.bak-attn-20261004-134709` | 3472 | `b2f324b8a47123f24825afd47259c3db67867f435cdaf07c8a9124f040b54a81` |
| `vllm-qwen38.service.bak-batch8192-20261003-225438` | `/etc/systemd/system/vllm-qwen38.service.bak-batch8192-20261003-225438` | 3384 | `789a4e2b64e29d7a7afbb24e39e1b26eee6ddff7e6e07c3c476729b183d5458a` |
| `vllm-qwen38.service.bak-beforedflashfix-20261001-183757` | `/etc/systemd/system/vllm-qwen38.service.bak-beforedflashfix-20261001-183757` | 3413 | `db3cd67cf775e552d459f51e8bb424a3cbfa7c4f9ca1421987d405bc80820d5a` |
| `vllm-qwen38.service.bak-bt24576-20261014-181412` | `/etc/systemd/system/vllm-qwen38.service.bak-bt24576-20261014-181412` | 3751 | `dfa42760e50ab0dffa6b04e3db314b073e60dc62e2509ffa3f2e88e2ca78b401` |
| `vllm-qwen38.service.bak-bt8192-20261004-131939` | `/etc/systemd/system/vllm-qwen38.service.bak-bt8192-20261004-131939` | 3472 | `b2f324b8a47123f24825afd47259c3db67867f435cdaf07c8a9124f040b54a81` |
| `vllm-qwen38.service.bak-ctx220-fp8-20261014-172816` | `/etc/systemd/system/vllm-qwen38.service.bak-ctx220-fp8-20261014-172816` | 3745 | `284ef21f3121d7f9fda64d70b9fc6e62bf60b4c35f7ae1279ac46b3999c0e46e` |
| `vllm-qwen38.service.bak-ctx262k-20261003-131055` | `/etc/systemd/system/vllm-qwen38.service.bak-ctx262k-20261003-131055` | 3395 | `fbfe498c79a05fd7725e1217f5cfac40f26d7ae19974306c0c8e0104622ef37c` |
| `vllm-qwen38.service.bak-draftbf16-20261014-191433` | `/etc/systemd/system/vllm-qwen38.service.bak-draftbf16-20261014-191433` | 3792 | `85642d8ebe4389cc025473190a575bfdb12fea26f88b8c5bf7029245de08b741` |
| `vllm-qwen38.service.bak-draftprob-20261014-180407` | `/etc/systemd/system/vllm-qwen38.service.bak-draftprob-20261014-180407` | 3744 | `0624adf4a798eabdf93de2862594612ffc96790036e12853680d85a8a1b4e39a` |
| `vllm-qwen38.service.bak-draftw4a16-20261003-122931` | `/etc/systemd/system/vllm-qwen38.service.bak-draftw4a16-20261003-122931` | 3389 | `785288bcf143b0da4d00ddb7044f7bd4579d513c894255995b1e71474c1dc2e7` |
| `vllm-qwen38.service.bak-draftw4a16-20261003-123337` | `/etc/systemd/system/vllm-qwen38.service.bak-draftw4a16-20261003-123337` | 3389 | `785288bcf143b0da4d00ddb7044f7bd4579d513c894255995b1e71474c1dc2e7` |
| `vllm-qwen38.service.bak-draftw4a16-20261003-125345` | `/etc/systemd/system/vllm-qwen38.service.bak-draftw4a16-20261003-125345` | 3389 | `785288bcf143b0da4d00ddb7044f7bd4579d513c894255995b1e71474c1dc2e7` |
| `vllm-qwen38.service.bak-enablemetrics-20261003-220459` | `/etc/systemd/system/vllm-qwen38.service.bak-enablemetrics-20261003-220459` | 3410 | `d80387d9d9f07beef5aa35719d498bf3ff007ce86c2f9babe56b5a580d508d01` |
| `vllm-qwen38.service.bak-env-` | `/etc/systemd/system/vllm-qwen38.service.bak-env-` | 3384 | `789a4e2b64e29d7a7afbb24e39e1b26eee6ddff7e6e07c3c476729b183d5458a` |
| `vllm-qwen38.service.bak-fix-20261010-152333` | `/etc/systemd/system/vllm-qwen38.service.bak-fix-20261010-152333` | 3535 | `42a62732bec675fc4b805b38215ae208dc71ba5ad4b5f04a1cc31c31020c4226` |
| `vllm-qwen38.service.bak-fix-20261010-153139` | `/etc/systemd/system/vllm-qwen38.service.bak-fix-20261010-153139` | 3535 | `0ae1a2500a294dfc8c4c6f79fb185512a224d76723a7ec31b513f797b8c461ea` |
| `vllm-qwen38.service.bak-gpuutil985-20261003-224225` | `/etc/systemd/system/vllm-qwen38.service.bak-gpuutil985-20261003-224225` | 3384 | `789a4e2b64e29d7a7afbb24e39e1b26eee6ddff7e6e07c3c476729b183d5458a` |
| `vllm-qwen38.service.bak-lb-20261004-140332` | `/etc/systemd/system/vllm-qwen38.service.bak-lb-20261004-140332` | 3472 | `b2f324b8a47123f24825afd47259c3db67867f435cdaf07c8a9124f040b54a81` |
| `vllm-qwen38.service.bak-lmhead-20261003-233507` | `/etc/systemd/system/vllm-qwen38.service.bak-lmhead-20261003-233507` | 3384 | `789a4e2b64e29d7a7afbb24e39e1b26eee6ddff7e6e07c3c476729b183d5458a` |
| `vllm-qwen38.service.bak-marlinadd-20261014-183714` | `/etc/systemd/system/vllm-qwen38.service.bak-marlinadd-20261014-183714` | 3751 | `dfa42760e50ab0dffa6b04e3db314b073e60dc62e2509ffa3f2e88e2ca78b401` |
| `vllm-qwen38.service.bak-marlinadd-20261014-185244` | `/etc/systemd/system/vllm-qwen38.service.bak-marlinadd-20261014-185244` | 3751 | `dfa42760e50ab0dffa6b04e3db314b073e60dc62e2509ffa3f2e88e2ca78b401` |
| `vllm-qwen38.service.bak-marlinint8-20261004-123633` | `/etc/systemd/system/vllm-qwen38.service.bak-marlinint8-20261004-123633` | 3439 | `43622c67258cb52f73c5f273d20edd5deed8c7027e2a7eceea5f25a707f75f6e` |
| `vllm-qwen38.service.bak-maxout24576-20261007-055822` | `/etc/systemd/system/vllm-qwen38.service.bak-maxout24576-20261007-055822` | 3472 | `b2f324b8a47123f24825afd47259c3db67867f435cdaf07c8a9124f040b54a81` |
| `vllm-qwen38.service.bak-mtp4-20261001-170009` | `/etc/systemd/system/vllm-qwen38.service.bak-mtp4-20261001-170009` | 3193 | `19821ea4597ea5dba889e281e657fe47383795a34ec80d1db05ad5b838a99a21` |
| `vllm-qwen38.service.bak-nosync-20261004-140955` | `/etc/systemd/system/vllm-qwen38.service.bak-nosync-20261004-140955` | 3503 | `8e5b9ad8169e5d69aa5094fe2b438216a6fc7be70fca880a2185dd8812c80cff` |
| `vllm-qwen38.service.bak-rename-20261004-002553` | `/etc/systemd/system/vllm-qwen38.service.bak-rename-20261004-002553` | 3476 | `7e1c36959db6ab7ee47a9dbba9142c7662ec63fe78cb0980c0970714cf08760e` |
| `vllm-qwen38.service.bak-rename-inca-20261014-121032` | `/etc/systemd/system/vllm-qwen38.service.bak-rename-inca-20261014-121032` | 3718 | `ca8a41494608486acbeeadbcba76cc3117ea3b81b10eb7cf1639348c965ae03b` |
| `vllm-qwen38.service.bak-rename-inca-20261014-121049` | `/etc/systemd/system/vllm-qwen38.service.bak-rename-inca-20261014-121049` | 3718 | `ca8a41494608486acbeeadbcba76cc3117ea3b81b10eb7cf1639348c965ae03b` |
| `vllm-qwen38.service.bak-restore-now` | `/etc/systemd/system/vllm-qwen38.service.bak-restore-now` | 3480 | `e6693d61b488af6bd0a96be46f63627f905176347282d3e85a3fee171853ddd5` |
| `vllm-qwen38.service.bak-revertdraft-20261014-173401` | `/etc/systemd/system/vllm-qwen38.service.bak-revertdraft-20261014-173401` | 3751 | `411bcf962191ab26cc9be151e25bacd124e58a96fbe211f22a9e4c5b6f0d50d6` |
| `vllm-qwen38.service.bak-single-20261004-132533` | `/etc/systemd/system/vllm-qwen38.service.bak-single-20261004-132533` | 3511 | `ab33b78a963fe6c49be0a8f56fe1f63b04efad87199828653abe89de1e30cf95` |
| `vllm-qwen38.service.bak-spec15-20261004-142628` | `/etc/systemd/system/vllm-qwen38.service.bak-spec15-20261004-142628` | 3472 | `b2f324b8a47123f24825afd47259c3db67867f435cdaf07c8a9124f040b54a81` |
| `vllm-qwen38.service.bak-switch-20261004-122117` | `/etc/systemd/system/vllm-qwen38.service.bak-switch-20261004-122117` | 3472 | `b2f324b8a47123f24825afd47259c3db67867f435cdaf07c8a9124f040b54a81` |
| `vllm-qwen38.service.bak-switch-awq-20261003-215015` | `/etc/systemd/system/vllm-qwen38.service.bak-switch-awq-20261003-215015` | 3395 | `1edac48e24e6be6ad36a62405b7cf37e85bcefc18461a38c06861fe7e8717b0e` |
| `vllm-qwen38.service.bak-switch-gptqpro-20261003-212452` | `/etc/systemd/system/vllm-qwen38.service.bak-switch-gptqpro-20261003-212452` | 3395 | `1edac48e24e6be6ad36a62405b7cf37e85bcefc18461a38c06861fe7e8717b0e` |
| `vllm-qwen38.service.bak-textonly-20261001-160013` | `/etc/systemd/system/vllm-qwen38.service.bak-textonly-20261001-160013` | 3185 | `9cb9819d9c61d093f98f4412a4848037a059d2beb6fac8a473e0e131fc0d0645` |
| `vllm-warmup.service.bak-20261010-152137` | `/etc/systemd/system/vllm-warmup.service.bak-20261010-152137` | 571 | `27faeb5c5f00f9ff010802ee5bd4d1ff9596817633ade1aa3cd7c678df459d93` |
| `vllm-warmup.service.bak-20261011` | `/etc/systemd/system/vllm-warmup.service.bak-20261011` | 877 | `3fb9c64187a2b35dc848eed3d7937d15bdb2a6b8d5ecd0058e01715febcba0a4` |
| `vllm-warmup.service.bak-rename-20261004-002553` | `/etc/systemd/system/vllm-warmup.service.bak-rename-20261004-002553` | 573 | `7279cc21ec37a97f77dd532c2158a1f9718881680adf7723d6e536e3778f75a4` |
| `gpu-power-limit.service.bak-20261011` | `/etc/systemd/system/gpu-power-limit.service.bak-20261011` | 377 | `20dfd2266b42d6da6e0f5412fd9e75fb1eb5b4539b717acfefa49267bf713035` |
| `gpu-fan-curve.service.bak-20261011` | `/etc/systemd/system/gpu-fan-curve.service.bak-20261011` | 589 | `5ab2231e3889ade840281fcf4ce59a192be4ffffb43a96725a60c7eae8a3374f` |
| `wb-prefix-warm.service.bak-20261011` | `/etc/systemd/system/wb-prefix-warm.service.bak-20261011` | 602 | `97e4aa8627985b6ae5a775c55ae3f667b6085119a3ab23d2068e225097558c45` |
| `wb-prefix-warm.service.bak-rename-20261004-002553` | `/etc/systemd/system/wb-prefix-warm.service.bak-rename-20261004-002553` | 604 | `cc2bec6581cee2bc7be00a84dc8302e9f9d4787a092eddf986a9b7c6f7fb7e51` |
| `vllm-warmup.sh.bak-fix-20261010-153139` | `/usr/local/bin/vllm-warmup.sh.bak-fix-20261010-153139` | 1511 | `f20c6213c853a3a5ab965107f277741ac74ec07c7a46bdf18df3092ac169dbaf` |
| `vllm-warmup.sh.bak-rename-20261004-004417` | `/usr/local/bin/vllm-warmup.sh.bak-rename-20261004-004417` | 1513 | `54e51c495a41bbb558bd5b237507dfae4e13943b8afdb97f2ba74c2b6c90b414` |
| `wb-prefix-warm.py.bak-rename-20261004-004417` | `/usr/local/bin/wb-prefix-warm.py.bak-rename-20261004-004417` | 2025 | `66f4e4a0cf8877ed13447739ae28863253c7912eb79203af4ca32a42c378fe48` |

## Notes

- Drop-ins `/etc/systemd/system/vllm-qwen38.service.d/` and `vllm-warmup.service.d/` do not exist on the host, so `deploy/systemd/dropins/` is empty.
- `/etc/systemd/system/{prometheus,grafana-server,node_exporter,nvidia_gpu_exporter}.service` are out of scope here (owned by the monitoring subtask).
- The host also keeps extra `.bak-*` copies of `vllm-qwen38.service` under `/root/` (26 files); the canonical set pulled here is the one under `/etc/systemd/system/`.
- `/root/models/Qwen3.8-27B-DFlash2-W4A16/` has no `generation_config.json` and no `recipe.yaml` (only `config.json` + `model.safetensors`), so only `config.json` was pulled.
