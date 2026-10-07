# 线上报告生成协议（协议级说明）

本文描述训练所对应的线上报告生成路径（SafetyRAISE `report_harness`，demo 模式，取自系统提交 `7200e30`）。只写协议与约束，不含部署基础设施信息。提示词原文与 schema 见 [`../assets/`](../assets)。

## 1. 一次运行的流程

1. **冻结快照**：37 个事故信息字段冻结；每个字段生成一条 `fact_obligation`（共 37 条，`source_refs` 指向 `accident:/<字段>`）。事故信息 ≤32000 字符时整体内联给角色，否则只给目录，要求用 `read_evidence` 读取。
2. **专家（prepare）**：按 `guidance_prompt.md` 生成指导意见 JSON（`{"guidance": ..., "knowledge": []}`，不得带知识）。
3. **首检由程序执行**，不由模型决定：`initial_retrieval(query, top_k=3)`；结果被截断则整次失败。
4. **候选循环**（最多 3 版）：
   - **生成者**每个模型回合的上下文 = `{instructions(generator.md), response_schema(CandidateReport), snapshot, prepared(guidance), candidate_version, previous_candidate, unresolved_issues, review_feedback, initial retrieval …}`；
   - 输出是 JSON（请求带 `response_format=json_object`）：要么**工具回合**（`tool_calls` JSON），要么最终 **`CandidateReport`**；
   - **审查者**在独立上下文中复核（不含生成者私有历史、专家指导、首轮片段）：输入 `candidate + candidate_digest + snapshot + unresolved_issues`，输出 `ReviewResult`，五类 `completed_checks`：facts / coverage / reasoning / citations / conciseness；
   - 全部检查通过且没有未关闭的 major/blocker → 发布；否则带着审查反馈进入下一版；**第 3 版仍不过 → `needs_review`，不产出报告**。
5. 每个角色循环有上限（模型回合、工具调用、协议修复次数、单响应大小）。

## 2. 工具（只读）

| 工具 | 参数 | 作用 |
| --- | --- | --- |
| `list_evidence` | – | 列出可读证据（事故信息字段等） |
| `read_evidence` | `evidence_ids≤10`, `cursor` | 读取证据全文 |
| `search_knowledge` | `query≤120 字`, `top_k≤3` | 检索知识库片段（只返回片段摘要） |
| `read_knowledge` | `chunk_ids≤10`, `cursor` | 读取完整来源条文 |

检索额度由 `retrieval_constraints` 动态收紧（剩余轮数/片段数），超限被拒。定义见 `assets/schemas/tool_schemas.json`。

## 3. `CandidateReport`

JSON schema 见 `assets/schemas/candidate_report.schema.json`。要点：

- `version`、`report_markdown`（业务正文，沿用旧报告模板结构）、`claims[]`、`obligation_resolutions[]`、`issue_responses[]`；
- `claims[]`：`claim_id`、`type ∈ {fact, inference, knowledge}`、`evidence_refs`、`knowledge_refs`、`quote`（必须是正文中**唯一出现**的连续片段，程序据此回填 `text_span`）；
- `obligation_resolutions[]`：**37 条，每个事故字段一条**（含空字段），`obligation_id` 形如 `fact:/<字段>`（字段名中的 `/` 转义为 `~1`），`treatment ∈ {covered, uncertain, not_relevant}`，附 `resolution` 说明；
- 空字段不得编造；评价性字段（事故认定原因、初查原因、主要违法行为、事故责任、致死原因）须归因“事故信息记载”。

## 4. 接受前的确定性检查

- 版本号 = 当前版本；`claims` 非空；
- 引用的来源必须是**本角色实际读取过**的（先读后引）；
- `knowledge` 类断言必须有 `knowledge_refs`；
- 37 条义务逐条处置；
- **规则摘录**（`#rule#` 类片段 / `source_kind=rule_excerpt`）**不能作为断言依据**：引用即判 citations 不通过，必须 `read_knowledge` 读取完整来源条文后再引。

## 5. 与训练的关系

- 训练样本以**第 1 回合**（首检片段已在提示里、无工具结果）的最终候选或工具决策为主；读片段后的后续回合会让提示暴涨 ~6K token，多数超出窗口。
- 后训练的“解读 37 项 JSON”目标：事实/评价/空缺/冲突字段各自怎样处理、关键事实不漏、空字段不编造、评价字段归因、断言的 `evidence_refs` 指向真正含该事实的字段、37 条义务处置与字段状态一致。这些由 `harness/sr_eval/live/metrics.py` 的解读指标检查。
- 线上请求若带 `reasoning.effort=high`，Qwen3.8 官方模板会报错；服务端须固定推理档位（见 [`deployment.md`](deployment.md)）。
