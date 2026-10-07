# profiles

评测台与数据构建按 `profiles/assets/<名称>/` 读取分词器与聊天模板。仓库内**不含** `tokenizer.json`（约 12.8 MB，来自官方仓库），请按 `profiles/assets/qwen3.8-official-1d4bf0f2/asset-manifest.json` 记录的 revision 与 SHA-256 自行下载并放到：

- `profiles/assets/qwen3.8-official-1d4bf0f2/tokenizer.json`
- `profiles/assets/qwen3.8-compact-v1/tokenizer.json`（与上面是同一个文件，`compact` 目录只替换聊天模板）

`SR_QWEN_PROFILE_DIR` 环境变量可指向其中任一目录（后训练构建行时用 `compact`）。
下载示例（固定 revision，校验 SHA-256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`）：

```bash
curl -L -o tokenizer.json "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json"
sha256sum tokenizer.json
```
