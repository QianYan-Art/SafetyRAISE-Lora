# harness：评测台、线上环境模拟器、Orin 脚本与工具

仓库里**所有代码**都在这里。四块，互相配合：

| 目录 | 作用 | 在哪运行 |
| --- | --- | --- |
| [`sr_eval/`](sr_eval) | **旧路径评测台**（复刻 2026-09-25 的生产旧路径）：提示渲染、检索、硬门、模型评审、教师修订、SFT 样本导出 | 笔记本 |
| [`sr_eval/live/`](sr_eval/live) | **线上环境模拟器**：冻结线上 `report_harness` 协议（生成者 / 审查者 / 工具 / 窗口），采样、确定性评估、后训练数据构建、审抽检 | 笔记本（另需线上系统模块，见下） |
| [`orin/`](orin) | Orin 上的训练、合并、量化、服务和解码研究脚本 | Orin |
| [`tools/`](tools) | 笔记本侧的数据构建、采样、评测、盲评、速度探针和日志分析脚本 | 笔记本（部分调用 Orin 上的服务） |

其余：`config/`（模型与单价、预算、外发闸门、通道）· `cases/`（合成事故案件：train 240 / dev 50 / test 50，37 字段事故信息 + 专家指导意见）· `assets/`（评测台早期冻结的**旧路径**报告提示词，线上提示词在仓库根的 `assets/`）· `tests/`、`tests_live/`（单元测试，不联网）。

所有结论都是自动口径（确定性硬门 + 模型评审），**未经人工核实**。评测台本身是纯标准库实现；线上环境模拟器另需 `tokenizers` 和线上系统的冻结模块。先读 [`../docs/handover.md`](../docs/handover.md) 和 [`../docs/reproduction.md`](../docs/reproduction.md)，再回来查具体脚本。

## 运行环境

```bash
uv venv .venv && uv pip install -r requirements-eval.txt            # 旧路径评测台（pytest 等）
uv venv .venv-live && uv pip install --python .venv-live -r requirements-live.txt   # 线上环境模拟器（另需线上系统冻结模块）
export PYTHONPATH=harness                                           # 之后用 python -m sr_eval.cli …
python -B -m pytest harness/tests                                   # 旧路径：不联网
python -B -m pytest harness/tests_live                              # 线上环境模拟器：需要线上系统冻结模块
```

脚本里的路径和解释器是按开发机（Windows + Git Bash，虚拟环境在 `.venv/Scripts`）写的，换环境时改脚本开头的 `PY=`/路径变量；`<repo>`、`<orin-ssh-key>`、`<orin-user>`、`<SafetyRAISE-system-repo>` 是占位符（原路径与账号已脱敏），使用前替换成自己的。精确分词渲染（构建后训练行）需要装了 `tokenizers` 的环境（`requirements-eval.txt` 里有），所以 `.venv` 和 `.venv-live` 两个环境都会用到。线上系统的冻结模块放在 `harness/vendor/safetyraise_7200e30/`（**仓库不含**，`.gitignore` 排除）；`tools/vendor_prod.py` 是当时从线上系统仓库（只读来源，提交 `7200e30`）复制并写 SHA-256 清单的脚本，需要自己的访问权限。

## 旧路径评测台 `sr_eval/`

在**同一提示、同一检索器、同一硬门**下比较教师模型、未训练基线与微调模型。它复刻了生产旧路径（提交 `7200e308`，2026-09-25 主线）：

- `harness/assets/report_prompt.md`：冻结的生产报告提示（6 个占位符，来源与哈希见 `harness/assets/ASSETS.json`）。
- 首轮检索 6 片段 → 模型可调 `retrieve_knowledge`（≤2 轮、每轮 ≤3 片段、总计 ≤9、query ≤120 字）→ 超限后不带工具强制收束。
- 与生产一致：**没有 tool 消息历史**，检索结果靠重新渲染 system 提示带入（`sr_eval/runner.py` 顶部有说明）。
- 执行提示两种变体：`production`（逐字沿用，含“尽可能充分展开”，与提示词里的简洁要求冲突）和 `concise`（与提示词【简洁与完整要求】一致）。
- 检索器是 BM25（字二元组）+ 分块/规则两路 RRF，不是生产的稀疏+稠密；只保证被比较的模型看到同一检索结果。

模块：`prompt` 渲染 · `kb` 检索 · `llm` 客户端 + 账本 + 预算 · `policy` 外发闸门 · `runner` ReAct 循环 · `gates` 硬门与指标 · `judge` 严格评审 + 播种错误校准 · `cases` 合成案件 · `report` 汇总 · `sft` / `pairs` / `revise` / `rationale` / `retrieval_data` 训练数据构造 · `cli`。运行产物（`runs/<时间_标签>/`：`run.json`、`traces/`、`gates.json`、`judgements.json`、`summary.md`；`ledger/api-calls.jsonl` 无密钥调用账本）由评测台自己生成，被 `.gitignore` 排除，不在仓库里。

```bash
export PYTHONPATH=harness
python -B -m sr_eval.cli check                                       # 账本合计、key 状态、外发闸门
python -B -m sr_eval.cli gen-cases --model minimax --split dev --start 0 --n 6 --workers 3
python -B -m sr_eval.cli run --models hy4,dsflash --cases dev --n 6 --kb synthetic --tag bake1 --workers 2
python -B -m sr_eval.cli judge --run bake1 --judges luna,mimo --workers 2
python -B -m sr_eval.cli calibrate --run bake1 --judge luna --n 3    # 播种错误校准评审区分度
python -B -m sr_eval.cli summary --run bake1
```

`run` 另有 `--variant production|concise`、`--max-tokens`（单次输出上限，含思考，也是预算预留依据）、`--timeout`（秒，默认 1800，与生产一致）、`--effort`、`--resume <run 目录>`（跳过已有 trace 续跑）。`judge/calibrate` 默认 `--effort max`（教师与评审一律 max 档；教师只返回总结式思考，低档位只会降低质量）。学生本地模型的模板档位用 `--effort xhigh`（`local` 提供者把它作为 `chat_template_kwargs.reasoning_effort` 传给 llama-server）。要换模型或单价改 `config/models.json`。

### 硬门（任一失败即不合格，不被评审总分覆盖）

`no_report` · `truncated_output` · `tool_markup_in_report` · `tool_call_unknown_tool/arguments_not_json` · `citation_not_visible`（`[依据: …]` 必须是可见片段 id）· `article_not_in_visible_text`（法条号/国标号必须出现在可见片段）· `fabricated_quantity`（速度、钟点必须来自输入；中文数字会折算）· `sentencing_or_crime_language` · `privacy_leak`（身份证/手机/邮箱）。其余（未结论式定责、比例、未支持的数量、复述指导意见原句、元话语、英文、缺章节、重复率）为 warn，计数并交评审参考。截断（`finish_reason=length`）记硬门失败，不拿截断冒充效率优化。

### 已知局限

- 检索器与生产不同（见上）；`sanitize_markdown_output` 为简化复刻（未含表格/加粗修复）。
- 数量门只认阿拉伯数字与简单中文数字，推算出的数值会进 warn；语义层面（事实归属、因果）只能靠评审。
- JSON 回退协议（不支持原生工具的模型）是近似复刻，尚未对照生产 legacy 分支逐行核对。
- 旧路径不等于线上现行路径：线上走 `report_harness`（生成者 JSON 协议 + 独立审查者），后训练与终版评测都以线上环境模拟器为准，见 [`../docs/live-protocol.md`](../docs/live-protocol.md)。

## 线上环境模拟器 `sr_eval/live/`

在内存里装配线上冻结的角色、工具和合同（只接受合成案），不依赖数据库，不碰真实案件。主要模块：

| 模块 | 作用 |
| --- | --- |
| `env.py`、`backends.py`、`retrieval.py`、`window.py` | 环境装配、可插拔模型后端（本地 Orin 服务 / 外部教师）、检索的确定性近似、单调用训练窗口（不切分、不删减、不静默截断） |
| `metrics.py` | 37 项可解释的解读质量指标（确定性、证据保守）；首调用“全部通过”口径 |
| `samples.py`、`rows.py`、`dataset.py`、`build_post.py`、`seeded.py` | 采样轨迹 → 真实 Qwen 模板 token 样本 → 偏好对/锚点行的构建器；播种错误对 |
| `revise.py`、`audit.py`、`verify.py`、`probe.py` | 逐调用校验与教师最小改动修订、盲评抽检包、验收证据、离线 token 分布探针 |
| `scripted.py`、`demo_pairs.py`、`pytest_compat.py` | 协议验证用的合成夹具（不是教师或训练数据）、测试辅助 |

数据流水线的命令串在 [`../docs/reproduction.md`](../docs/reproduction.md) 第 3 节；数据格式见 [`../docs/data-card.md`](../docs/data-card.md)。

## `orin/`：Orin 上的脚本（整个目录复制到 `~/work/scripts/`）

解码研究脚本还会从同一目录调用 `tools/` 里的 7 个小工具，也要复制过去：`probe_multi.py`、`speed_probe2.py`、`answer_probe.py`、`answer_multi.py`、`bench_table.py`、`probe_compare.py`、`draft_log_compare.py`。C++/CUDA 诊断程序（`dump_hidden.cpp` 等）的编译命令写在各源文件的头注释里。

| 类别 | 脚本 |
| --- | --- |
| 训练 | `train_sft.py`（QLoRA SFT，只适配上半层，断点续训）· `train_simpo.py`（SimPO + 锚点，无参考模型，一次只放一条序列，梯度解耦）· `train_simpo_v3a.py`（v3a 当时用的旧版 SimPO，仅作记录）· `train_mtp.py`（MTP 头自蒸馏，逐块反传，每个 epoch 原子保存最优） |
| 训练后链 | `post_train.sh`（v3a）· `post_chain.sh`（v3a → v3b）· `post_chain_ft.sh`（v3c 起的 SFT 链）· `post_chain_pref.sh`、`post_chain_final.sh`（偏好链，v3h / v3h2 用）· `mtp_chain.sh`（MTP 头重训链：缓存隐藏状态 → 训练 → 覆盖合并 → 量化 → 起服务） |
| 合并与量化 | `merge_lora.py`（PEFT 适配器合并进官方 BF16，CPU 流式）· `make_gguf_v3.sh`（合并 → 转 GGUF → 量化，可覆盖 MTP 头） |
| 服务 | `serve_delivery.sh`（**交付入口**）· `serve_llama.sh`（通用启动，环境变量见脚本开头）· `serve_local.py`（HF 版评测服务，慢，早期用）· `bench_server.py`（对 llama-server 做真实提示的速度基准） |
| 解码研究 | `bench_sweep.sh` + `bench_configs_decode*.txt`（单路扫描）· `region_sweep.sh` + `region_configs*.txt`（思考 / 答案分区间）· `answer_sweep.sh` + `answer_configs.txt`、`answer_multi_batch.sh` + `answer_multi_configs.txt`（答案区间）· `multi_batch.sh`、`multi_run.sh` + `multi_configs.txt`（6 / 3 / 1 路并发）· `prefill_study.sh`（预填充对照，补丁 0004 的验证）· `prof_decode.sh`、`prof_multi.sh`、`prof_summary.py`（nsys 剖析） |
| 诊断 | `bw_probe.cu`（可持续读带宽）· `sampler_cost.cpp`（CPU 采样开销）· `shortlist_scatter_test.cpp`（词表子集补丁的算子序列单测）· `dump_hidden.cpp`、`eval_head_on_served.py`、`served_feature_check.sh`（MTP 头在服务模型真实特征上的一致率） |

llama.cpp 的补丁、编译方法和实测结果在 [`../inference/`](../inference)。

## `tools/`：笔记本侧脚本

多数脚本开头的注释写明了用法。会调用外部教师/评审的脚本标了 **（外部）**，需要自备接口和密钥，并先读下面的“费用与外发闸门”。

| 类别 | 脚本 |
| --- | --- |
| 数据构建 | `scale_data.sh`、`big_loop.sh`、`expert_iter.sh`、`onpolicy_v3a.sh`（SFT 数据放大 / 专家迭代，**（外部）**）· `gate_repair.sh`、`gate_repair2.sh`（硬门修复数据，**（外部）**）· `rev_pipeline.sh`（教师修订后处理，**（外部）**）· `post_pipeline.py`（线上环境采样 → 教师最小改动修订 → 构建训练行与抽检包，**（外部）**）· `finalize_rows.py`（按排除清单派生最终训练集）· `export_live_records.py`（导出与分词器无关的文本版记录）· `make_mtp_data.py`（MTP 重训序列）· `verify_pairs.py`、`spot_pairs.py`、`audit_split.py`、`audit_collect.py`（偏好对体检与抽检）· `ret_writer_minimax.py`（**（外部）**） |
| 采样 | `live_batch.py`（学生或教师在线上环境里批量跑案件，可断点续跑）· `live_prompt_probe.py`、`live_prompt_sizes.py`（只读、不调模型：渲染第 1 回合并精确计 token）· `live_zero_shot.py`（把第 1 回合 payload 直接发给 Orin 服务） |
| 评测与对比 | `auto_eval_v3.sh`、`eval_rest.sh`、`eval_test.sh`（旧路径的评测链，**（外部）**）· `judge_more.sh`、`mm_judge_chain.sh`、`minimax_fill.sh`、`rejudge_baselines.sh`、`teacher_max.sh`（评审与教师基线，**（外部）**）· `teacher_compare.py`、`snapshot_results.py`、`think_metrics.py`（汇总）· `live_compare.py`、`quant_table.py`、`auto_quant_mtp.sh`（线上环境结果对比 / 量化横评）· `final_eval.sh`、`final_eval_summary.py`（终版总评测）· `prod_retrieval_gap.py`、`check_conformance.py`（检索差距与注入一致性核对；只读使用线上系统源码，本地运行，需要自己有线上系统仓库） |
| 独立盲评 | `blind_ab.py`、`blind_ab_score.py`、`blind_agg.py`、`blind_cmp.py`、`blind_cmp_score.py`（生成盲评材料并汇总）· `blind_run.sh`、`dsh_review.patch.yml`（调用与训练评审无关的命令行评审者；只喂合成数据与代码，**（外部）**） |
| 速度与日志 | `speed_probe.py`、`speed_probe2.py`、`probe_compare.py`（单路探针与逐字对比）· `probe_multi.py`（多路并发）· `answer_probe.py`、`answer_multi.py`（答案区间）· `bench_table.py`、`region_table.py`、`multi_table.py`（出表）· `ngram_potential.py`（离线估算 n-gram 投机潜力）· `draft_vocab.py`、`draft_log_compare.py`（草稿词表子集的生成与等价性核对）· `server_log_stats.py`、`slot_concurrency.py`（llama-server 日志统计）· `fetch_decode_results.sh` |
| 其他 | `vendor_prod.py`（冻结线上系统模块）· `wait_event.sh`（阻塞到日志出现新的 EVENT/ALERT 行，给后台等待用） |

## 费用与外发闸门（必读）

- **本仓库里的外发 / 费用 / 通道配置是历史记录，不构成对新使用者的授权。** 调用任何外部服务前，自己确认授权、费用上限和数据可否外发。仓库所有者自 2026-10-06 起不再把 OpenRouter 用于后训练（软件预算上限已用尽），不要在没有确认的情况下重新启用。
- 外发闸门 `config/outbound.json`：**合成案件可外发；真实案件永不外发**；公开法规/知识库片段外发由仓库所有者在 2026-10-02 批准过。若改回 false，只能用 `--kb synthetic`（12 条虚构条款，只验证机制）。
- 密钥只来自环境变量（`OPENROUTER_API_KEY`、`MINIMAX_API_KEY` 等，见 `sr_eval/llm.py`）；也可用 `SR_SECRETS_FILE` 指向一个本地“字段: 值”文本文件（缺省 `secrets.local.txt`，已被 `.gitignore` 排除）。密钥只在进程内使用，不入账本、日志、异常文本。**不要把密钥写进仓库、提示词或训练数据。**
- OpenRouter 的 key 必须带 credit limit（服务商侧硬上限）才会发送；软件侧再按每次请求的最坏费用（输入按 1 token/字符、输出按 `--max-tokens`）预留，超过 `config/budget.json` 的 `cap_usd` 就不发送；未结算与 unknown 请求按预留计入。`cli check` 显示 key 的额度与余额。
- 通道按 `config/network.json`：OpenRouter 经环境代理（`HTTPS_PROXY`，CONNECT 隧道），MiniMax 直连，Orin 本地服务走局域网明文 HTTP；代理缺失时发送前报错，不会悄悄直连。每次请求最多在 429 / 503 时重试 1 次；网络中断记 `unknown`，不自动重试。
- 微调后的学生模型只在 Orin 本地（llama-server）上跑，不经过任何外部接口。
