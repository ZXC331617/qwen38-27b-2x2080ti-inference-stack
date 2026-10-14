# ldap.toml 脱敏说明

- 源：主机 `/etc/grafana/ldap.toml`（sha256 `d11037e9f7b2faf9bc93133639df586abe706567970be57770a9a766d2dc6729`，3137 字节，未改动主机）
- 未脱敏原件：`deploy/raw/C2/pristine/ldap.toml`
- 脱敏产物：`deploy/monitoring/ldap.toml.sanitized`
- 规则同 `grafana.ini`：键名含 `password` → `PASSWORD`；键名保留，值占位。

## 命中并替换的键（共 1 处）

| 行 | 键名 | 替换为 | 原值长度 |
|---:|------|-------|--------:|
| 32 | `bind_password` | `PASSWORD` | 9 |

## 现场结论

1. 该文件为 Grafana 出厂 LDAP 模板：`host = "127.0.0.1"`、`port = 389`、`search_base_dns = ["dc=grafana,dc=org"]`，`bind_password` 为出厂示例值（9 字符）。
2. 主机上没有 LDAP 服务（`ss -lntp` 未见 389/636），LDAP 认证实际未启用，该文件只是随包安装的默认模板。
3. 第 33 行是注释掉的环境变量引用示例（形如 `bind_password = 变量展开`），未参与替换，因为其值是变量引用而非明文。
