# grafana.ini 脱敏说明

- 源：主机 `/etc/grafana/grafana.ini`（sha256 `b421182d029ef1813050960861de360f81649b540b3bb7fd93e3c9d20808820d`，81706 字节，未改动主机）
- 未脱敏原件：`deploy/raw/C2/pristine/grafana.ini`
- 脱敏产物：`deploy/monitoring/grafana.ini.sanitized`
- 规则：键名含 `password` / `passwd` / `secret` → `PASSWORD`；含 `token` / `api_key` → `TOKEN`。
  键名含 `url` / `file` / `name` / `hint` / `expiration` / `rotation` / `interval` / `limit` / `days` / `policy` / `enabled` / `enable` / `use_` / `refresh` / `_ttl` / `version` / `architectures` 的不脱敏（非机密）。
- 键名全部保留，值一律占位。

## 命中并替换的键（共 26 处）

| 行 | 键名 | 替换为 | 原值长度 | 主机上是否为注释 |
|---:|------|-------|--------:|:---------------:|
| 127 | `password` | `PASSWORD` | 0 | 是 |
| 304 | `intercom_secret` | `PASSWORD` | 0 | 是 |
| 330 | `admin_password` | `PASSWORD` | 5 | 是（原值为 Grafana 出厂默认值） |
| 336 | `secret_key` | `PASSWORD` | 20 | 是（原值为 Grafana 出厂示例密钥） |
| 611 | `api_key_max_seconds_to_live` | `TOKEN` | 2 | 是 |
| 665 | `client_secret` | `PASSWORD` | 11 | 是 |
| 694 | `client_secret` | `PASSWORD` | 11 | 是 |
| 722 | `client_secret` | `PASSWORD` | 18 | 是 |
| 752 | `client_secret` | `PASSWORD` | 11 | 是 |
| 767 | `client_secret` | `PASSWORD` | 18 | 是 |
| 792 | `client_secret` | `PASSWORD` | 11 | 是 |
| 821 | `client_secret` | `PASSWORD` | 11 | 是 |
| 1007 | `user_identity_client_secret` | `PASSWORD` | 0 | 是 |
| 1039 | `password` | `PASSWORD` | 0 | 是 |
| 1151 | `api_key` | `TOKEN` | 10 | 是 |
| 1247 | `ha_redis_password` | `PASSWORD` | 0 | 是 |
| 1443 | `loki_basic_auth_password` | `PASSWORD` | 8 | 是 |
| 1483 | `basic_auth_password` | `PASSWORD` | 0 | 否（唯一未注释的敏感键，值为空） |
| 1571 | `basic_auth_password` | `PASSWORD` | 0 | 是 |
| 1591 | `sso_api_token` | `TOKEN` | 2 | 是 |
| 1660 | `secret_key` | `PASSWORD` | 0 | 是 |
| 1665 | `password` | `PASSWORD` | 0 | 是 |
| 1691 | `renderer_token` | `TOKEN` | 1 | 是 |
| 1754 | `ha_engine_password` | `PASSWORD` | 2 | 是 |
| 1916 | `update_webhook_token` | `TOKEN` | 0 | 是 |
| 1932 | `gcom_api_token` | `TOKEN` | 2 | 是 |

## 现场结论

1. 该 ini 是 Grafana 11.5.1 的出厂配置：除第 1483 行 `basic_auth_password =`（空值）外，全部敏感键均处于注释状态。
2. `;admin_password = admin` 处于注释状态，说明 Grafana 仍在使用默认管理员口令 admin。这是本次拉取中最值得注意的配置风险点，已如实记录。
3. 第 336 行 `;secret_key =`（20 字符出厂示例值）同样未启用，实际签名密钥由 Grafana 首次启动时写入数据库，不在 ini 里。
4. 数据源 provisioning（`provisioning/datasources/sample.yaml`）全部为注释模板，不存在真实数据源 UID / token，故无额外占位符需要生成。
