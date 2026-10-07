# 数据卡

所有数据均为**原创合成**，不含真实人员、车辆、地点、号牌或案件；所有质量判断为模型评审 + 确定性检查，**人工未核实**。
本仓库的数据**不是**真实事故资料，也不是责任认定依据。

## 1. `datasets/v1-synthetic-release/`（首版发布包，哈希绑定，原样保存）

| 文件 | 内容 |
| --- | --- |
| `sft-training.zip` | 1,698 条 SFT（534 个事故组；writer 1,398 / reviewer 300），含 token 化与 messages 两种同批视图（只选一种，不叠加） |
| `rl-prompts.zip` | 534 条在线后训练输入（仅 actor 可见） |
| `rl-reward-references.zip` | 534 条奖励参考与接入合同，与 actor 隔离 |
| `heldout-evaluation.zip` | 开发/测试各 50 组，只用于评测 |
| `manifest.json` | 文件与成员 SHA-256、数量、血缘与边界 |
| `DATA-CARD.md`、`START-TRAINING.md`、`training-strategy.md` | 首版数据卡、训练入口、策略 |

要点与边界（见其自带数据卡）：总 token 2,941,164、监督 token 705,083；序列 671–4,568 token（与线上 2–3 万 token 的真实形态差约 10 倍）；正式偏好对、真实模型工具轨迹、法条引用任务均为 0；`outbound_eligible=false` 为首版治理标记，**公开发布以仓库所有者授权为准**；`START-TRAINING.md` 中的本地路径指原工作目录，仅作历史记录。**不要修改该目录内的文件**（清单中的哈希会失效）。

## 2. `harness/cases/`

评测台使用的合成事故案件（`train/ dev/ test/`，240/50/50 个 JSON）：37 字段事故信息 + 专家指导意见（按线上指导提示词合成）+ 场景元数据（`scenario`、信息量档 sparse/medium/rich、是否埋入矛盾）。字段名见 `assets/prompts/input_accident_template.json`。事实量分三档，稀疏档约 24 个非空字段、约 150 字，贴近真实输入。

## 3. `datasets/sft-lineage-v3/`

逐版监督微调数据（`train.jsonl.gz` / `eval.jsonl.gz` + 部分 `manifest.json`）：

| 目录 | 样本 | 用在 |
| --- | --- | --- |
| `v3b` | 48 | v3b |
| `v3c` | 97 | v3c |
| `v3d_new` + `v3d`（含回放 36） | 13 + 36 | v3d |
| `v3f_new` + `v3f`（含回放 46） | 40 + 46 | v3f |

行字段：`sample_id, case_id, teacher, report_start, kind(final/tool), thinking_source, judge, gains, coverage, prompt_tokens, target_tokens, input_ids, labels, target_text`。`input_ids/labels` 是 Qwen3.8 分词视图（`labels` 只覆盖目标段）；`target_text` 为报告文本；思考来源为学生原生思考（`thinking_source=student_native`），报告为教师（luna，max 档）按评审缺陷最小改动修订。**旧路径格式**（原生 tool call + 直接 Markdown），与线上 JSON 协议不同，作为“写作能力”基础使用。

> v3e（失败）、v3b 之前的试验数据与 v3h 早期 38 对旧格式偏好对已作废，不收录。

## 4. `datasets/live-post-training-v3h/`（线上环境后训练集）

由线上环境模拟器在**线上同款协议**下采样、过确定性门、教师最小改动修订、独立评审筛选得到。

| 文件 | 内容 |
| --- | --- |
| `rows.jsonl.gz` | **训练行**（token id，可直接喂 `train_simpo.py`）：`pair_id, kind, prompt_ids, chosen_ids, rejected_ids, c_start, r_start` |
| `records.jsonl.gz` | **文本版**记录（与分词器无关）：提示消息、请求参数、学生思考、选中/被拒回答、教师溯源、原始失败信号、diff |
| `provenance.jsonl` | 每行的来源：案件、源调用哈希、教师请求/响应哈希与账本 ID、种子证据 |
| `manifest.json` | 数量、各类行、窗口分布、剔除行及原因、各轮抽检摘要、`audit_policy_note`、哈希 |

### 4.1 规模与形态

26 行：16 对 `pair_revised`（学生失败回答 vs 修订通过回答，**同一提示、同一思考前缀逐 token 相同**，仅回答段不同）+ 10 条 `anchor_final`（修订版回答，只做 NLL）。提示中位 17.0K token，回答起点（思考前缀长度）中位 4,024（即 compact 档 4000 思考预算被触达），最长行 31,899 token，6 行超过 28,000。分词器 SHA-256 `0997f410…9b9f3`，模板（`compact` 档）SHA-256 `54d8f120…4fab`。

### 4.2 构造与质量门

1. 学生（v3f，compact + 预算 4000）在 110 个训练案第 1 回合采样（108 个有效首调用）；
2. 确定性评估：候选契约、来源访问（先读后引）、37 项义务一对一、`quote` 唯一、规则摘录不得作依据，加解读指标（关键事实覆盖、空字段不编造、评价字段归因、义务与字段状态一致、证据引用指向对应字段；pass/fail/unknown，unknown 单列）；
3. 修订：MiniMax-M3.1-Flash-Preview（max 档）最小改动修订，≤2 次，仍不过则丢弃；
4. 选择：每个源调用至多一行；偏好对优先取“学生真实失败 → 修订通过”；锚点仅取经 Codex 全量严审通过且复审无评审判不合格者。

### 4.3 独立评审与口径

见技术报告 §6.6。要点：偏好方向 21/21 次评审均被一致认可（两轮抽检 13 + 最终 8）；绝对质量不要求满分；锚点逐条严审（40/86 通过，最终入选 10 条）；Kimi 因配额未参与。训练在最终抽检完成前已开始，完成后按清单重启。`manifest.json` 中 `release_eligible=false`、`human_verified=false`。

### 4.4 已知问题

- **过度保守**：个别回答把案件已记载的事实（如碰撞）写成“不能据此认定”；
- **复合断言引用不完整**：一条断言含多个事实，引用只覆盖其中一部分；
- 评价性字段（事故认定原因、初查原因、主要违法行为、事故责任）有时被降级为“不确定”；
- 未覆盖：> 25K 的首提示、读片段后的多回合、修订轮、审查者角色；
- 规模很小（26 行），主要用于协议适配与 37 项解读校正。

### 4.5 重新构建 / 扩展

- 由文本版重建 token 版：用 `assets/templates/qwen3.8-compact.chat_template.jinja` 渲染 `messages` + 学生思考 + 回答，按 `answer_start` 切段，细节见 `harness/sr_eval/live/rows.py`。
- 扩展：见 [`extension-guide.md`](extension-guide.md)。

## 4b. `datasets/live-post-training-v3h2/`（第二轮后训练集）与 `datasets/mtp-retrain-data/`

- `live-post-training-v3h2/`：28 对偏好（26 `pair_revised` + 2 `pair_seeded_*`），文件同 §4（`rows.jsonl.gz`、`records.jsonl.gz`、`provenance.jsonl`、`manifest.json`）。与第一轮的调用互不重叠（构建时 `--exclude-calls`）；每个源调用至多一行；方向抽检 13/15 两位评审一致认可选中侧，2 条因 DeepSeek 判 tie 剔除（清单见 `manifest.json` 的 `excluded_rows`）。**不含锚点**；绝对质量不要求满分（见 §4.3 口径）。窗口 p50 23.8K、最长 29.6K。训练配方见技术报告 §6.8。
- `mtp-retrain-data/train.jsonl.gz`：MTP 头重训用的 60 条序列（学生在线上协议下的首回合输出：提示 + 思考 + 最终 JSON；排除了第一轮后训练用过的案件；共约 140 万 token；字段 `sample_id, case_id, kind, input_ids, labels`，`labels` 只覆盖回答段，用 compact 档位模板渲染，分词器同 §4.1）。留出集用终版模型自己在开发案上的输出（`make_mtp_data.py --eval-traces`），不入仓库。

## 5. `eval/`

`results-latest.{md,json}`（各版本 12/50 案评审快照）、`teacher-compare-{dev,test}.md`、`prod-retrieval-gap-dev.json`（线上检索与近似检索的差距）、`orin-train-logs/`（Orin 训练日志摘要）。

## 6. 不在仓库内的内容

LoRA 适配器、MTP 头、GGUF、线上系统源码与部署配置、真实案件相关材料、外部调用账本与凭据、原始采样轨迹与教师响应（体量大且含外部服务返回，保留于所有者本地归档）。
