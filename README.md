# SafetyRAISE-Lora

SafetyRAISE 系统配套报告模型（Qwen3.8-27B + LoRA）的**训练资产仓库**：数据集、训练与评测脚本、提示词与协议资产、推理优化补丁、技术报告和接手指南。目标是让后来的人**只看这个仓库就知道怎么往下训**。模型权重（LoRA 适配器、MTP 头、GGUF）不在仓库里，随训练用的 Jetson Orin 机器一起交接，校验和见接手指南。

> **重要声明**
> - 所有训练与评测案件均为**原创合成**，不含真实人员、车辆、地点或案件；评分来自模型评审与确定性检查，**未经人工核实**。
> - 模型输出是“事故分析报告草稿”，**不是**交通事故责任认定书，也不构成法律意见。
> - 模型权重**不在本仓库**；仓库只放数据、脚本、提示词和文档。
> - 本仓库的“线上系统”指 SafetyRAISE 的 `report_harness` 报告生成路径；系统代码本身不在仓库内，提示词与协议资产取自其 `7200e30` 提交。

## 从哪里开始

| 我想… | 读这里 |
| --- | --- |
| **接手这个项目、接着训练** | [`docs/handover.md`](docs/handover.md)（现状、权重在哪、四条续训路线、已被证伪的方向、已知问题与下一步） |
| 照着命令复现数据流水线 / 训练 / 合并量化 / 评测 | [`docs/reproduction.md`](docs/reproduction.md) |
| 看做了什么、为什么、实测数字 | [`docs/technical-report.md`](docs/technical-report.md) |
| 看终版总评测和解码研究的结果 | [`eval/final-eval-2026-10-08/`](eval/final-eval-2026-10-08/README.md) |
| 部署服务、搞清窗口和截断 | [`docs/deployment.md`](docs/deployment.md)、[`inference/README.md`](inference/README.md) |
| 了解数据 | [`docs/data-card.md`](docs/data-card.md)、`datasets/` |
| 扩展（更长窗口、新案件、视觉部分、LoRA 融合） | [`docs/extension-guide.md`](docs/extension-guide.md) |

## 当前状态（2026-10-08）

模型 = Qwen3.8-27B + 后训练 LoRA（**v3h2**）合并后量化 **Q4_K_M** 的单个 GGUF，MTP 头已针对该模型重训。服务用 [`harness/orin/serve_delivery.sh`](harness/orin/serve_delivery.sh)：**compact 推理档位、思考预算 5120、KV 缓存 q8_0、每槽位 32768 × 6 槽位、MTP 草稿（上限 5，按在跑槽位数自适应）、草稿词表子集、服务端默认采样用模型官方推荐值（temperature 1.0、top_k 20、top_p 0.95）**。

终版总评测（开发 12 案，线上协议全流程，单次采样，自动口径）：发布 6、待审 5、失败 1（v3f 基线 7 / 4 / 1）；首调用通过全部确定性检查 4/12（v3f 1/12）。**主要问题**：预算 5120 在 32768 窗口下，34 次调用里有 11 次被槽位上下文截断（全是提示已超约 23K 的案件）；与教师的差距（v3f 评审配对差 −1.39、独立盲评约 −1.69 分）和硬门失败率（8/50 对教师 3/50）是最近一次实测，终版总评测里没有重测，均尚未达到既定通过线。详见接手指南第 6 节。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| [`docs/`](docs) | `handover.md` 接手指南 · `technical-report.md` 技术报告 · `reproduction.md` 复现命令 · `deployment.md` 部署 · `data-card.md` 数据卡 · `live-protocol.md` 线上协议 · `extension-guide.md` 扩展 · `changelog.md` 更新记录 |
| [`datasets/`](datasets) | 数据集（见下表） |
| [`assets/`](assets) | 线上提示词模板、候选报告 JSON schema、工具定义、聊天模板（含 `compact` 档位） |
| [`harness/`](harness) | 评测台与线上环境模拟器（`sr_eval/`）、训练/合并/量化/服务脚本（`orin/`）、数据构建、审查与分析工具（`tools/`）、合成案件（`cases/`）、测试；索引见 [`harness/README.md`](harness/README.md) |
| [`inference/`](inference) | **推理程序**：llama.cpp 补丁（草稿词表子集、按负载自适应草稿长度等）、词表子集 id 列表、实测结果与复现说明 |
| [`eval/`](eval) | 各版本评测快照、教师对比、Orin 训练日志摘要、终版总评测与解码研究；索引见 [`eval/README.md`](eval/README.md) |
| [`profiles/`](profiles) | 模型分词器/模板的来源与校验信息（分词器文件需自行下载） |

## 数据集一览

| 目录 | 内容 | 规模 |
| --- | --- | --- |
| `datasets/v1-synthetic-release/` | 首版合成 SFT/RL/评测发布包（哈希绑定，原样保存） | 1,698 条 SFT（534 组）、534 条 RL 任务与奖励参考、开发/测试各 50 组 |
| `harness/cases/` | 评测台使用的合成事故案件（37 字段事故信息 + 专家指导意见） | train 240 / dev 50 / test 50 |
| `datasets/sft-lineage-v3/` | v3b–v3f 逐版监督微调数据（学生原生思考 + 教师最小改动修订报告） | 约 100 万 token 量级，逐版清单 |
| `datasets/live-post-training-v3h/` | **线上环境后训练集**第一轮（偏好对 + 锚点，文本版与 token 版） | 26 行：16 对偏好 + 10 锚点 |
| `datasets/live-post-training-v3h2/` | 第二轮后训练集（偏好对，文本版与 token 版） | 28 对偏好 |
| `datasets/mtp-retrain-data/` | MTP 头重训用的学生线上协议输出序列 | 60 条，约 140 万 token |

数据质量与限制见数据卡；`v1-synthetic-release` 的清单中标注 `outbound_eligible: false`（首版治理策略，沿用原样），发布范围以仓库所有者的授权为准。

## 快速开始

```bash
# 笔记本侧：评测台、线上环境模拟器与测试
uv venv .venv && uv pip install -r requirements-eval.txt        # 线上环境模拟器另需 requirements-live.txt
# 需要线上系统的提示词模板与 report_harness 模块、分词器：见 docs/handover.md 第 2 节 与 docs/reproduction.md 第 0 节
# Orin 侧：把 harness/orin/ 复制到 ~/work/scripts/，按 docs/handover.md 第 3 节选一条续训路线
bash ~/work/scripts/serve_delivery.sh v3h2m-Q4_K_M               # 起交付配置的服务（监听 Orin 的局域网地址 :8080）
```

## English summary

SafetyRAISE-Lora holds everything needed to continue training the report-generation model of the SafetyRAISE traffic-accident analysis system: a QLoRA fine-tune of Qwen3.8-27B (rank-16 LoRA on the upper 32 layers) trained entirely on a single Jetson AGX Orin 64 GB, served as a merged Q4_K_M GGUF with a retrained MTP draft head through a patched llama.cpp. The repository contains the synthetic datasets (v1 SFT release, v3 SFT lineage, the live-environment post-training sets, MTP retraining sequences), the prompt/schema assets of the production protocol, the training/evaluation/serving scripts, the llama.cpp patches and decoding studies, a technical report, and a handover guide (`docs/handover.md`). Model weights are not in the repository; they travel with the Orin machine (checksums are listed in the handover guide). All cases are synthetic and all quality scores are automatic (unverified by humans).

## 许可

许可证尚待仓库所有者确定（代码、数据、权重各需单独声明；基座 Qwen3.8-27B 沿用其官方许可）。在确定之前，本仓库内容按“保留所有权利”处理。

## 标识

- 仓库所有者：QianYan-Art
- 项目：SafetyRAISE 报告模型训练集合
