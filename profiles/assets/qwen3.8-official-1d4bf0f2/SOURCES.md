# Qwen3.8 官方文本资产（本地验证）

- 模型仓库：`Qwen/Qwen3.8-27B`
- 固定 revision：`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- 来源：`https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/`
- 仅取 `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`、`config.json`、`LICENSE`；`special_tokens_map.json` 在该 revision 返回 404，不推测其内容。
- 2026-09-24 以 `Invoke-WebRequest -NoProxy` 对固定 revision 的上述五项做 HEAD，均返回 200；`tokenizer.json` 声明长度 12,809,320 字节。实际下载后以本地哈希和解析校验为准。
- 同日用 `Invoke-WebRequest -NoProxy` 下载五项，大小和 SHA-256 固定在 `asset-manifest.json`；`tokenizer_config.json` 内嵌模板与独立 `chat_template.jinja` 逐字相同。官方 `config.json` 的文本模型类型为 `qwen3_5_text`，仓库名称仍按发布方原样记录。
- 用途：受限本地训练数据的真实 tokenizer 与官方模板验证；不下载权重，不训练、不部署，不把这些文件的许可元数据当作事故来源或外发授权。
- Python 依赖拟锁定自 PyPI：`tokenizers==0.23.2`、`Jinja2==3.1.6`、`MarkupSafe==3.0.3`。2026-09-24 已经以无代理方式查询 PyPI 版本及 Windows/Python 3.13 wheel 元数据；实际安装及传递依赖的版本以项目锁文件记录，安装前再次核对直连与哈希。
- 已用仅在该进程中清空代理环境变量并设置 `NO_PROXY=*` 的 `uv add --no-config --default-index https://pypi.org/simple` 安装；直接依赖见 `pyproject.toml`，完整传递依赖及版本见 `uv.lock`。这不等于下载模型权重或验证训练运行时。
