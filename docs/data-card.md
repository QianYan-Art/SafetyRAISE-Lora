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

> v3e（失败）和 v3b 之前的试验数据已作废，不收录。v3a 与早期 v3h 的两份 38 对偏好对见下，**原件已不在**。

### v3a 的 38 对偏好对（谱系的起点，数据不在仓库）

- **用在哪**：v3a（2026-10-03），SimPO 加对 chosen 的 NLL（β=1.0、γ=0.2、chosen NLL 权重 0.2、学习率 2e-5，每次更新 2 对，共 19 次更新，约 7.3 小时），LoRA r16 只适配上半层，起点是未训练基座。训练日志见 `eval/orin-train-logs/v3_simpo.jsonl`：损失在 0.63 到 0.86 之间波动，没有明显下降（首次 0.77，末次 0.74），所以“影响很小”。它的适配器是 v3b 的初始化，之后 v3b、v3c、v3d、v3f、v3h、v3h2 一路接着训。
- **怎么来的（自己造的，不是外部数据；按顺序）**：
  1. **采轨迹（`smp1`）**：用旧路径评测台（`python -m sr_eval.cli run`，生产注入形态：首轮检索 6 片段，可调 `retrieve_knowledge`，xhigh 档）让**未训练的基座**（配置里的 `qwen27` = OpenRouter 上的 `qwen/qwen3.8-27b`，学生同源，只用于这类自采样；该通道已停用）在 80 个合成训练案上各跑 1 条完整轨迹。
  2. **采最终回答（`finalk`）**：`sr_eval.cli sample-final` 沿用 `smp1` 里已有的检索上下文，对其中 60 个案让同一个模型重复写最终回答，每案 3 份，共 180 份。
  3. **打分**：每条轨迹、每份回答都过确定性硬门（`gates`）、字段覆盖率，以及思考循环、被截断的检测；两批样本里硬门失败 18 条轨迹、47 份最终回答——到了最终步还在调检索工具而不写报告、思考陷入重复、被长度截断，正是这次要修的“尾巴”。
  4. **选对（`pairs-build --no-judge`，`harness/sr_eval/pairs.py` 阶段 B，不用逐条评审）**：同一案件、同一固定检索上下文下的多份候选里，chosen = 硬门全过、正常结束、无思考循环、至少一条引用，再按“覆盖率 − 报告字数 − 思考字符”的加权分取最高；rejected 优先取终止类失败（被截断、思考循环、最终步没写出报告），其次硬门失败，最后才是质量明显较差者；硬门失败者不会成为 chosen。
  5. **过滤质量对（`harness/tools/spot_pairs.py report|filter`）**：终止类和硬门类是客观失败，不需要评审；质量类依赖打分规则，用 luna（max）逐对验证，评审不认可或缺失的不进训练。
  6. **转 token 并上传**：用官方模板渲染成 `prompt_ids` / `chosen_ids` / `rejected_ids`（被截断的样本不补结束符），传到 Orin 给 `train_simpo_v3a.py`。
- **条数**：第 4 步得 45 对（终止类 26、硬门类 4、质量类 15），另有 61 条最佳样本；第 5 步 13 个有效质量对里 chosen 更高 8、持平 2、更低 3，均值差 +0.23（区间跨 0，说明这套打分对质量的区分力弱），只留被认可的 8 个，最终 **38 对（终止 26 + 硬门 4 + 质量 8）**。序列长度中位数约 21K、最长约 27.9K token。工程检查：提示不一致 0 对、空响应 0 对、被裁切 0 对；终止类 rejected 与 chosen 的响应长度中位数之比 1.01（没有靠压低长度取巧）。
- **格式**：每行 `{pair_id, prompt_ids, chosen_ids, rejected_ids}`，响应部分含思考与 `<|im_end|>`，提示渲染到 `<think>\n` 为止，即 `harness/orin/train_simpo_v3a.py` 读取的格式。
- **为什么不在仓库、还在不在**：当时判断它“影响很小、已被后面的版本取代”，数据卡早先就把它列为不收录。原件（`harness/sft/v3nj2/`）和原始采样轨迹 2026-10-07 归档、2026-10-08 随归档整体删除，Orin 上的副本更早已清，**现在任何地方都没有副本**。删除前记录的 SHA-256 指纹：`pairs.jsonl`（45 对）`9bbb698fdd9b6fa66d61af41ce95ad9c09f536e5806ac02f1b33fcffe5216b84`，`pairs.filtered.jsonl`（38 对）`fe13ee7256612f504a0fec02c5fdd6dffadc51f2ab5e9c79716adba67101bb85`，`best.jsonl` `b8b099f3b9dc5328d268bc37d03244079170bf333606c36650c88e8b13092e3b`；只有指纹，不能还原内容。
- **对续训的影响**：没有。交付的 v3h2 与保留的 v3d、v3f 适配器都在 Orin 上，续训从它们起步即可；只有想“从未训练基座起完整重走整条谱系”才缺这一环，重造需要重新采样（结果不会逐字相同）。

### 早期 v3h 的 38 对旧格式偏好对（作废，从未训练）

2026-10-06 为 v3f 构造的一份旧路径（非线上协议）报告段偏好对：30 对是硬门失败的修复，8 对是评审缺陷明显的修订；同一案件、同一提示、同一段原生思考，教师修复后通过硬门的报告为 chosen，原来的失败报告为 rejected，平均每条序列约 1.8 万 token。因为改走线上环境后训练而作废，被现在的 `live-post-training-v3h` 取代；原件同样随归档删除，指纹 `pairs.jsonl` `4e155548432756fdb6748c1ea10dbb5a9c5685ec0e61f13e063f61edb940e97c`。

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

`results-latest.{md,json}`（各版本 12/50 案评审快照，2026-10-06）、`teacher-compare-{dev,test}.md`、`prod-retrieval-gap-dev.json`（线上检索与近似检索的差距）、`orin-train-logs/`（Orin 训练日志摘要，含 v3h / v3h2 / MTP 头）、`final-eval-2026-10-08/`（终版总评测与解码研究结果）；索引与读法见 [`../eval/README.md`](../eval/README.md)。

## 6. 不在仓库内的内容

LoRA 适配器、MTP 头、GGUF、线上系统源码与部署配置、真实案件相关材料、外部调用账本与凭据、原始采样轨迹与教师响应（体量大且含外部服务返回，保留于所有者本地归档）。

## 7. 许可与公开范围（2026-10-08）

- **许可证**：Apache License 2.0，与基座 Qwen3.8-27B 的官方许可相同（`LICENSE`、`NOTICE`）。本节的口径由仓库所有者授权、按合理范围拟定。
- **公开的**：所有案件与数据集都是原创合成，不含真实案件或真实个人信息——首版发布包 `datasets/v1-synthetic-release/`、`harness/cases/`、`datasets/sft-lineage-v3/`、`datasets/live-post-training-v3h*/`、`datasets/mtp-retrain-data/`；以及代码、提示词与 schema、聊天模板、llama.cpp 补丁、评测与训练日志的汇总、文档。
- **不公开的**：模型权重（LoRA、MTP 头、GGUF；它们是 Qwen3.8-27B 的衍生品，同样按 Apache-2.0 授权，随训练机交接）；外部服务调用账本、原始采样轨迹、教师与评审的原始响应、盲评材料；线上系统源码与部署配置；任何真实案件材料；凭据。
- **首版发布包的 `outbound_eligible: false`**：这是首版导出时的保守治理标记，原意是“只用于本地训练，不能据此上传云训练或外部裁判接口”（见该包自带数据卡“使用限制”）。清单有哈希绑定，保持原样不改。因为包内容是 100% 原创合成、仓库所有者已把合成数据放行用于外部训练与评审（`harness/config/outbound.json`），本仓库的公开副本按 Apache-2.0 发布，这个标记对使用者不再构成限制；调用具体的外部服务时，仍应自行确认服务条款与费用。
- **知识库片段**：后训练集的提示里带着线上知识库的片段，来自公开的官方来源（法律法规、公安交管部门公开文件、国家标准等），版权属各自来源；它们的来源标注（含“由某人提供的官方 PDF”之类的备注，这是线上知识库原有的元数据）随片段原样保留，因为训练行是 token 级的、改动会使文本版与训练行对不上。Apache-2.0 不改变这些文本原有的权利状态，如有异议请联系仓库所有者删除。
