# 资产说明

| 路径 | 内容 | 来源 |
| --- | --- | --- |
| `prompts/report_prompt.md`、`guidance_prompt.md`、`input_generation_prompt.md`、`input_accident_template.json` | 线上报告提示词、专家指导提示词、事故信息生成提示词（视觉/多源材料 → 37 字段 JSON）、37 字段模板 | SafetyRAISE 系统提交 `7200e30` 的 `backend/config/` |
| `prompts/report_harness/generator.md`、`reviewer.md` | 线上生成者 / 独立审查者角色提示词（JSON 协议） | 同上 |
| `schemas/candidate_report.schema.json`、`tool_schemas.json` | 候选报告的 JSON schema、四个只读工具的定义 | 由系统代码导出（pydantic `model_json_schema()` 与 `tool_schemas()`） |
| `templates/qwen3.8-official.chat_template.jinja` | Qwen3.8 官方聊天模板（只认 `xhigh / medium / low`） | Qwen/Qwen3.8-27B，固定 revision 见 `profiles/assets/.../SOURCES.md` |
| `templates/qwen3.8-compact.chat_template.jinja` | 增设 `compact` 推理档位的模板（xhigh 行为不变）；放到 Orin 的 `~/work/templates/compact.jinja`，服务端 `--chat-template-file` 用它 | 本项目修改 |
| `templates/budget_msg.txt` | 思考预算耗尽时服务端注入的提示语（`--reasoning-budget-message`）；放到 Orin 的 `~/work/templates/budget_msg.txt`，`serve_llama.sh` 读取它 | 本项目 |

说明：`harness/assets/` 是评测台早期冻结的**旧路径**报告提示词（原生 tool call 形态），与 `prompts/` 下的线上 `report_harness` 提示词不同，二者都保留；后训练只以 `prompts/` 为准。
这些提示词是线上系统的一部分，复用时请遵守系统所有者的约定。
