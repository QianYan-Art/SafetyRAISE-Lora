# SafetyRAISE-Lora

SafetyRAISE 系统配套报告模型（Qwen3.8-27B + LoRA）的**训练资产仓库**：数据集、训练与评测脚本、提示词与协议资产、技术报告。
本仓库持续更新，每次更新见 [`docs/changelog.md`](docs/changelog.md)。

> **重要声明**
> - 所有训练与评测案件均为**原创合成**，不含真实人员、车辆、地点或案件；评分来自模型评审与确定性检查，**未经人工核实**。
> - 模型输出是“事故分析报告草稿”，**不是**交通事故责任认定书，也不构成法律意见。
> - 模型权重（LoRA 适配器、MTP 头、GGUF）**不在本仓库**；仓库只放数据、脚本、提示词和文档。
> - 本仓库的“线上系统”指 SafetyRAISE 的 `report_harness` 报告生成路径；系统代码本身不在仓库内，提示词与协议资产取自其 `7200e30` 提交。

## English summary

SafetyRAISE-Lora holds everything needed to continue training the report-generation model of the SafetyRAISE traffic-accident analysis system:
a QLoRA fine-tune of Qwen3.8-27B (rank-16 LoRA on the upper 32 layers) trained entirely on a single Jetson AGX Orin 64 GB, served as a merged Q4_K_M GGUF with an MTP draft head through llama.cpp.
The repository contains the synthetic datasets (v1 SFT release, v3 SFT lineage, and the live-environment post-training set), the prompt/schema assets of the production protocol, the training/evaluation scripts, and a technical report. All cases are synthetic and all quality scores are automatic (unverified by humans).

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| [`docs/technical-report.md`](docs/technical-report.md) | **技术报告**：目标、基座与方法、数据、训练、线上环境后训练、评测、部署、局限 |
| [`docs/data-card.md`](docs/data-card.md) | 数据卡：每份数据的来源、数量、字段、质量门与已知问题 |
| [`docs/reproduction.md`](docs/reproduction.md) | 复现与续训：环境、命令、Orin 上的完整链路 |
| [`docs/deployment.md`](docs/deployment.md) | 部署：GGUF 量化、llama-server 参数、compact 推理档位、窗口 32768 的截断核算、MTP |
| [`docs/live-protocol.md`](docs/live-protocol.md) | 线上报告生成协议（生成者 / 审查者 JSON 协议、工具、37 项义务）的协议级说明 |
| [`docs/extension-guide.md`](docs/extension-guide.md) | 如何扩展：更长窗口数据、新案件、视觉部分微调、两份 LoRA 融合的注意事项 |
| [`docs/changelog.md`](docs/changelog.md) | 更新记录 |
| [`datasets/`](datasets) | 数据集（见下表） |
| [`assets/`](assets) | 线上提示词模板、候选报告 JSON schema、工具定义、聊天模板（含 `compact` 档位） |
| [`harness/`](harness) | 评测台与线上环境模拟器（`sr_eval/`）、训练/合并/量化/服务脚本（`orin/`）、数据构建与审查工具（`tools/`）、测试 |
| [`eval/`](eval) | 各版本评测快照、教师对比、Orin 训练日志摘要 |

## 数据集一览

| 目录 | 内容 | 规模 |
| --- | --- | --- |
| `datasets/v1-synthetic-release/` | 首版合成 SFT/RL/评测发布包（哈希绑定，原样保存） | 1,698 条 SFT（534 组）、534 条 RL 任务与奖励参考、开发/测试各 50 组 |
| `harness/cases/` | 评测台使用的合成事故案件（37 字段事故信息 + 专家指导意见） | train 240 / dev 50 / test 50 |
| `datasets/sft-lineage-v3/` | v3b–v3f 逐版监督微调数据（学生原生思考 + 教师最小改动修订报告） | 约 100 万 token 量级，逐版清单 |
| `datasets/live-post-training-v3h/` | **线上环境后训练集**（偏好对 + 锚点，文本版与 token 版） | 26 行：16 对偏好 + 10 锚点 |
| `datasets/live-post-training-v3h2/` | 第二轮后训练集（偏好对，文本版与 token 版） | 28 对偏好 |
| `datasets/mtp-retrain-data/` | MTP 头重训用的学生线上协议输出序列 | 60 条，约 140 万 token |

数据质量与限制见数据卡；`v1-synthetic-release` 的清单中标注 `outbound_eligible: false`（首版治理策略，沿用原样），发布范围以仓库所有者的授权为准。

## 快速开始

```bash
# 评测台与测试（需要线上系统的提示词模板；见 docs/reproduction.md 的"重新提取线上代码"）
uv venv .venv && uv pip install -r requirements-eval.txt
# 在 Orin 上训练（QLoRA，bnb-4bit 基座）与合并、量化、服务：见 harness/orin/ 与 docs/reproduction.md
```

## 许可

许可证尚待仓库所有者确定（代码、数据、权重各需单独声明；基座 Qwen3.8-27B 沿用其官方许可）。在确定之前，本仓库内容按“保留所有权利”处理。

## 标识

- 仓库所有者：QianYan-Art
- 项目：SafetyRAISE 报告模型训练集合
