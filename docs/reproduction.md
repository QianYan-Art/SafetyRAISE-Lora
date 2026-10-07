# 复现与续训

目标：在新环境里，只靠本仓库 + 线上系统仓库（取提示词与协议模块）接着训练。所有命令里的 `<repo>` 指本仓库根目录；`<orin>` 指 Orin 的 SSH 目标。

## 0. 前置条件

- **训练机**：Jetson AGX Orin 64GB（JetPack 7.2.1）。QLoRA 训练 27B 需要 ~45–50 GiB 统一内存，训练前必须停掉 llama-server。
- **基座**：官方 BF16 `Qwen/Qwen3.8-27B`（合并/MTP 训练用）与 `unsloth/Qwen3.8-27B-bnb-4bit`（训练用）。Orin 直连 huggingface.co 常超时，用镜像站下载。
- **本机**：Python 3.12 + `uv`；评测台与线上环境模拟器用 `requirements-eval.txt` / `requirements-live.txt`。
- **分词器**：`profiles/assets/*/tokenizer.json` 需自行下载并校验（见 `profiles/README.md`）。
- **线上提示词与协议模块**：模拟器需要线上系统 `backend/app/report_harness` 等模块；提示词与 schema 已放在 `assets/`，但**系统代码不在本仓库**。用 `harness/tools/vendor_prod.py` 从系统仓库的 `7200e30` 提交冻结复制到 `harness/vendor/safetyraise_7200e30/`（脚本里的 `SRC` 路径改成你的系统仓库位置），校验以其 `MANIFEST.json` 的 SHA-256 为准。
- **外部模型**：教师/评审调用走 `harness/config/models.json` 与 `network.json`，密钥用环境变量，**不写进仓库**；预算与外发闸门在 `budget.json` / `outbound.json`。

> 仓库里的脚本多数以“仓库根为工作目录”运行，并把路径写成 `<repo>`（原工作目录路径已脱敏），使用前替换成你的路径；`harness/tools/*.sh` 内的 `cd <repo>` 同理。

## 1. 测试（不联网）

```bash
export PYTHONPATH="harness;harness/vendor/safetyraise_7200e30"   # Windows 用分号，Linux 用冒号
pytest harness/tests          # 评测台：假传输层
pytest harness/tests_live     # 线上环境模拟器与后训练流水线（需要冻结的线上代码；SR_QWEN_PROFILE_DIR 指向 compact 档位目录）
```

测试输出目录若放在系统临时目录会因路径限制失败，请用 `--basetemp <repo 内路径>`。

## 2. SFT（旧路径格式的 v3 谱系，Orin）

```bash
# 数据：datasets/sft-lineage-v3/<版本>/train.jsonl.gz → 解压后放到 Orin ~/work/data/
~/venvs/ft/bin/python train_sft.py --data train.jsonl --eval_data eval.jsonl --out ~/work/runs/<tag>_sft \
    --init_adapter <上一版 ckpt> --lr 1e-4 --accum 4 --epochs 1 --lora_from 32 \
    --think_weight 0.1 --report_weight 1.0 --max_len 28000 --save_every 12
```

要点：`--lora_from 32` 只在上半层加适配器；报告段与思考段分别加权；检查点含优化器与数据游标，可断点续训；SoC 超温自动保存退出。每版的数据量、学习率、更新次数见技术报告 §4.4 / §5。

## 3. 后训练数据流水线（线上环境模拟器）

```bash
# 3.1 学生在线采样（第 1 回合；服务端用 compact 档位 + 思考预算 4000；同一台服务一次只跑一个批次）
SR_LOCAL_HOST=<orin>:8080 SR_LIVE_EFFORT=compact SR_LIVE_ATTEMPT_TIMEOUT=7200 \
SR_QWEN_PROFILE_DIR=profiles/assets/qwen3.8-compact-v1 \
python -B harness/tools/live_batch.py --tag post1 --cases-file <cases.txt> --backend local --reviewer scripted \
    --retrieval sparse_half --workers 6 --stable-limit 32768 --output-reserve 7600 --max-tokens 32000 --first-call-only

# 3.2 确定性评估 + 教师最小改动修订 + 构建训练行（MiniMax 并发≤3；有缓存可重复运行）
python -B harness/tools/post_pipeline.py --tag post1 --min-traces 1 --workers 3 --final \
    --anchor-quota 16 --pair-quota 16 --unique-calls --seeded-cap 3 --min-long 6 --max-len 32768 \
    [--anchor-allowlist <严审通过的 source_call_sha256.json>] [--audit-fraction 1.0]

# 3.3 抽检：拆包 → 并行盲评 → 汇总
python harness/tools/audit_split.py <构建目录> <前缀> 3
bash   harness/tools/blind_run.sh codex|kimi|deepseek|all <前缀_a> <前缀_b> <前缀_c>
python harness/tools/audit_collect.py <构建目录> <audit.json> [--pairs-direction-only] <前缀_a> <前缀_b> <前缀_c>

# 3.4 按剔除清单派生最终集，并导出文本版
python harness/tools/finalize_rows.py <构建目录> <最终目录> <exclude.json> <review.json>
python harness/tools/export_live_records.py <最终目录> <修订记录目录> <最终目录>/records.jsonl
```

注意：

- 教师修订调用有 429（窗口型用量上限）与 529（过载）的退避重试；并发过高会触发过载。
- 校验指标升级后，旧“合格”修订记录与现行修订指令不再绑定，`post_pipeline.py` 会把它们归档到 `revisions_<tag>_stale/` 并重做。
- `SR_LIVE_ATTEMPT_TIMEOUT`：模拟器单次调用默认 1800 s（与线上一致）；本地学生并发采样时排队 + 长提示会超过，需放宽，且**不要同时对同一台服务开两个 6 路批次**。

## 4. 后训练（SimPO + 锚点），Orin

```bash
# 数据：datasets/live-post-training-v3h/rows.jsonl.gz → 解压为 ~/work/data/<name>_pairs.jsonl
bash post_chain_final.sh <初始适配器目录(如 v3f ckpt)> <TAG> <name> -- \
  --beta 2.0 --gamma 0.4 --sft_lambda 0.1 --anchor_lambda 1.0 --lr 4e-5 --accum 2 --epochs 1 \
  --report_only --max_len 32768 --lora_from 32 --longest_first 2 --save_every 4
```

链路：停 llama-server → `train_simpo.py` → 合并（`make_gguf_v3.sh`，保留 Q8_0）→ Q4_K_M → `serve_llama.sh`；状态写 `~/work/runs/<TAG>_post.status`，失败写 `FAIL:`。速度参考（~26K token 行）：锚点约 455 s、偏好对约 980 s；峰值显存 43.3 GiB（26.5K）。

## 5. 评测

```bash
export PYTHONPATH=harness SR_LOCAL_HOST=<orin>:8080
python -B -m sr_eval.cli ...                     # 旧路径评测台：见 harness/README.md
# 部署口径的在线环境评测（32768、不设输出上限，让服务端自然截断）：
python -B harness/tools/live_batch.py --tag <tag> --cases-file <dev12.txt> --backend local --reviewer scripted \
    --retrieval sparse_half --workers 6 --stable-limit 32768 --output-reserve 1 --max-tokens 32000
```

对比口径：同一批开发案、同一档位与预算；看硬门失败、评审分、思考长度、检索轮数、合法率与 37 项指标，不要只看评审分。

## 6. MTP 头重训

`train_mtp.py --mode cache`（载入终版模型 + 适配器缓存隐藏状态，约 110 分钟）→ `--mode train`（约 65 分钟）→ `merge_lora.py --mtp_overlay mtp.safetensors` → 重转 GGUF → 实测接受率并核对输出不变。MTP 只影响速度。

## 7. 踩坑清单

| 现象 | 原因 / 处理 |
| --- | --- |
| 启动即报 `MISSING_CREDENTIAL`、或连接被拒 | 评测台默认连 `:8000`（HF 服务），llama-server 在 `:8080`，需设 `SR_LOCAL_HOST` |
| HTTP 500，`Unexpected reasoning effort` | Qwen3.8 模板不认 `high`；服务端固定 `--reasoning-effort compact` |
| 采样记录 `budget_exhausted` 且无响应 | 单次调用超 1800 s（排队/长提示），设 `SR_LIVE_ATTEMPT_TIMEOUT` 并避免并发批次 |
| 训练首个更新数十分钟无输出 | 偏好对 ~26–32K token 每对约 16–22 分钟，属正常；`--longest_first` 让最长行先跑以便尽早暴露显存问题 |
| 合并时磁盘写满 | BF16 52 GB + 合并目录 52 GB + bf16 GGUF 54.6 GB 同时存在会爆；走 Q8_0 中转（`make_gguf_v3.sh`） |
| `.cmd` 启动器报找不到命令 | 参数含空格会被拆坏；任务文本走标准输入，路径用正斜杠 |
| 评审 JSON 解析不到 | 不同评审输出形态不同（数组 / 拼接的多个对象 / 偏好材料里 `passed` 是对象）；`audit_collect.py` 已兼容 |
| 校验指标改动后修订“耗尽” | 先用真实输出核对误判率；标记词表曾漏“为空”导致约 46% 误判 |
