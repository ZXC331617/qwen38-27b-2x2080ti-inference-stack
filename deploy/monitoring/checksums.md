# C2-A 校验：原样拉取文件 sha256（本地 vs 主机）

- 主机侧：`sudo sha256sum <file>`，原始输出留存于 `deploy/raw/C2/checksums-remote.txt`
- 本机侧：`Get-FileHash -Algorithm SHA256`，对 `deploy/raw/C2/pristine/` 下**位相同**的副本计算
- 结论：**13 / 13 全部一致**，无需重拉
- 说明：`grafana.ini.sanitized`、`ldap.toml.sanitized` 是脱敏产物，值已占位，**故意**与主机不同，不参与本表比对；其未脱敏原件保存在 `deploy/raw/C2/pristine/`

| # | 主机路径 | 本地产物 | 字节 | 本地 sha256 | 主机 sha256 | 一致 |
|---|---------|---------|-----:|------------|------------|:----:|
| 1 | `/etc/prometheus/prometheus.yml` | `deploy/monitoring/prometheus.yml` | 611 | `74b7e426790ff6107efa821e6a57c24927f0cec960d15c417b97aaa62a5bb643` | `74b7e426790ff6107efa821e6a57c24927f0cec960d15c417b97aaa62a5bb643` | 一致 |
| 2 | `/etc/prometheus/prometheus.yml.bak-modellabel-20261003-221009` | `deploy/monitoring/backups/` 同名 | 583 | `ee90fd4029771839a0390f2c4021ba64d90f9d74452f869c1a0249326c169865` | `ee90fd4029771839a0390f2c4021ba64d90f9d74452f869c1a0249326c169865` | 一致 |
| 3 | `/etc/grafana/grafana.ini` | `deploy/raw/C2/pristine/grafana.ini`（原件）+ `deploy/monitoring/grafana.ini.sanitized`（脱敏） | 81706 | `b421182d029ef1813050960861de360f81649b540b3bb7fd93e3c9d20808820d` | `b421182d029ef1813050960861de360f81649b540b3bb7fd93e3c9d20808820d` | 一致 |
| 4 | `/etc/grafana/ldap.toml` | `deploy/raw/C2/pristine/ldap.toml`（原件）+ `deploy/monitoring/ldap.toml.sanitized`（脱敏） | 3137 | `d11037e9f7b2faf9bc93133639df586abe706567970be57770a9a766d2dc6729` | `d11037e9f7b2faf9bc93133639df586abe706567970be57770a9a766d2dc6729` | 一致 |
| 5 | `/etc/systemd/system/prometheus.service` | `deploy/monitoring/units/prometheus.service` | 479 | `7601a85b008d3549ecce3a5b804da77c64d02014d8335f5b1715d964b3be6656` | `7601a85b008d3549ecce3a5b804da77c64d02014d8335f5b1715d964b3be6656` | 一致 |
| 6 | `/lib/systemd/system/grafana-server.service` | `deploy/monitoring/units/grafana-server.service` | 1678 | `b567b10979cc7d60bc9fa6d0517248f3b092fbd02e64c905d96a04c7c150bddf` | `b567b10979cc7d60bc9fa6d0517248f3b092fbd02e64c905d96a04c7c150bddf` | 一致 |
| 7 | `/etc/systemd/system/node_exporter.service` | `deploy/monitoring/units/node_exporter.service` | 277 | `e099dfa1543b5a64144866b428461c95b9454dff1e25835e4e34662011777f20` | `e099dfa1543b5a64144866b428461c95b9454dff1e25835e4e34662011777f20` | 一致 |
| 8 | `/etc/systemd/system/nvidia_gpu_exporter.service` | `deploy/monitoring/units/nvidia_gpu_exporter.service` | 325 | `ab654190dc8edbcfa561237aadae338d955a8ed2336644c5f2e800dfe2b59ca0` | `ab654190dc8edbcfa561237aadae338d955a8ed2336644c5f2e800dfe2b59ca0` | 一致 |
| 9 | `/etc/grafana/provisioning/access-control/sample.yaml` | `deploy/monitoring/grafana-provisioning/access-control/sample.yaml` | 2737 | `04354df6ac647c02b024c0197427a2ad4bc10479a9aeea81c5b0557cd32ad63a` | `04354df6ac647c02b024c0197427a2ad4bc10479a9aeea81c5b0557cd32ad63a` | 一致 |
| 10 | `/etc/grafana/provisioning/alerting/sample.yaml` | `deploy/monitoring/grafana-provisioning/alerting/sample.yaml` | 10185 | `0e5e375515c2e1dd1be4c9d8a25984bc7cc2246b4a037e511e72993f1056c616` | `0e5e375515c2e1dd1be4c9d8a25984bc7cc2246b4a037e511e72993f1056c616` | 一致 |
| 11 | `/etc/grafana/provisioning/dashboards/sample.yaml` | `deploy/monitoring/grafana-provisioning/dashboards/sample.yaml` | 185 | `1cbd7490113576661c97f0c2495e19f2c6f57e9d675a3909212520c48804b2d5` | `1cbd7490113576661c97f0c2495e19f2c6f57e9d675a3909212520c48804b2d5` | 一致 |
| 12 | `/etc/grafana/provisioning/datasources/sample.yaml` | `deploy/monitoring/grafana-provisioning/datasources/sample.yaml` | 2648 | `b3b46f214838f7d0fc48c24192db4f855075dd049cea07c4f6124984d11a9b76` | `b3b46f214838f7d0fc48c24192db4f855075dd049cea07c4f6124984d11a9b76` | 一致 |
| 13 | `/etc/grafana/provisioning/plugins/sample.yaml` | `deploy/monitoring/grafana-provisioning/plugins/sample.yaml` | 219 | `dc11119192cf7b07625dafddb2af0b3790362bde14900ba8ed59dc49073f9652` | `dc11119192cf7b07625dafddb2af0b3790362bde14900ba8ed59dc49073f9652` | 一致 |

## 主机上不存在的路径（已确认，非遗漏）

- `/etc/systemd/system/grafana-server.service`：不存在；真实单元在 `/lib/systemd/system/grafana-server.service`（`systemctl show -p FragmentPath` 确认）
- `/etc/systemd/system/vllm-qwen38.service.d/`：不存在，该单元无 drop-in
- `/etc/prometheus/rules.d/`：不存在，无告警/记录规则文件
- `/var/lib/grafana/dashboards/`：不存在，provisioning 只有 5 个 `sample.yaml`（全部为 Grafana 出厂模板，注释状态，无真实数据源/仪表盘）
