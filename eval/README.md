# 评测结果与训练日志索引

本目录只放**汇总结果和训练日志摘要**，不放逐案轨迹、评审原文、案件文本和教师响应（这些含外部服务返回或案件正文，只保留在仓库所有者本地）。

**口径（所有文件通用）**：所有案件都是原创合成；评分来自模型评审者与确定性检查（硬门），**未经人工核实**；除特别注明外，每个数字都是单次采样，开发 12 案时噪声很大，只能看趋势。线上协议（`report_harness` 路径）和旧评测台路径的区别见 `docs/live-protocol.md`。

## 文件

| 路径 | 内容 | 对应的技术报告章节 |
| --- | --- | --- |
| [`results-latest.md`](results-latest.md)、[`results-latest.json`](results-latest.json) | **v3 谱系快照（2026-10-06）**：基座 / v2a / 教师 / v3b–v3f 在开发前 12 案和完整 50 案上的评审均分、硬门失败、思考量；走的是旧评测台路径（学生直接作答），不是线上协议 | §5 |
| [`teacher-compare-dev.md`](teacher-compare-dev.md)、[`teacher-compare-test.md`](teacher-compare-test.md) | 教师基线对比：默认档 vs max 档（报告字数、引用、检索轮数、思考 token、成本、硬门） | §4.3 |
| [`prod-retrieval-gap-dev.json`](prod-retrieval-gap-dev.json) | 线上检索与近似检索的差距（开发 50 案逐案的检索结果比较），说明为什么要做线上环境模拟器 | §6.1–6.2 |
| [`orin-train-logs/`](orin-train-logs) | Orin 上每次训练的日志摘要（见下） | §3、§6、§8.1 |
| [`final-eval-2026-10-08/`](final-eval-2026-10-08/README.md) | **终版总评测**（交付配置，线上协议全流程，开发 12 案）：案件状态、首调用确定性指标、窗口截断、端到端速度、服务端统计；`decode-study/` 是解码与预填充优化研究的结果 | §6.10、§8.2 |

## `orin-train-logs/` 怎么读

文件名里的 `_sft` 是 Orin 上运行目录的名字（`~/work/runs/<版本>_sft/log.jsonl`），**v3a、v3h、v3h2 实际是 SimPO 偏好后训练**，不是监督微调。每行一个 JSON 事件：

| 文件 | 训练 | 事件字段 |
| --- | --- | --- |
| `v2a_r16.jsonl` | v2a 监督微调（教师短理由替换思考，已停用） | `start` 配置；`eval` 留出损失；`update`：`loss, grad_norm, lr, tokens, sec, tok_s, peak_gib, epoch` |
| `v3b_sft.jsonl` … `v3f_sft.jsonl` | v3b–v3f 监督微调（接着上一版继续训） | 同上 |
| `v3_simpo.jsonl` | v3a：38 对偏好的 SimPO（修收尾，β=1.0、γ=0.2、学习率 2e-5） | `update`：`loss, margin, pref_acc, grad_norm, sec, peak_gib` |
| `v3h_sft.jsonl` | **v3h**：线上环境后训练第一轮（16 对偏好 + 10 锚点行，13 次更新，β=2.0、γ=0.4、`sft_lambda` 0.1、学习率 4e-5） | 另有 `a_w, a_l`（选中 / 拒绝序列的平均 token 对数概率）、`anchors`（该次更新里的锚点行数；只含锚点时 `loss`、`margin`、`pref_acc` 为 null） |
| `v3h2_sft.jsonl` | **v3h2（交付版本）**：第二轮（28 对偏好，14 次更新，β=2.0、γ=0.4、`sft_lambda` 1.0、学习率 1e-4） | 同上 |
| `mtp_v3h2_train.jsonl` | 为 v3h2 重训的 MTP 头（60 条序列，3 个 epoch；`eval` 事件里 epoch 0 是官方头在微调后主模型上的一致率，`top1_by_step` 是第 1、2、3 步的 top-1 一致率） | `train_start, eval, step, epoch, checkpoint, saved` |
| `mtp_v3h2_train_oom_attempt.txt` | 同一训练的第一次尝试：第 2 个 epoch 的 `kl_div` 显存不足而中止；修复（逐块反传、`expandable_segments`、每个 epoch 保存最优）见技术报告 §8.1 | – |
| `v3a_post.status` … `v3h2_post.status` | 训练后链（训练 → 合并 → 转换 → 量化 → 起服务）每一步的时间戳 | 行首为 `时:分:秒` |

读法提示：

- SFT 的 `eval` 事件是留出集损失（`update: 0` 是训练前的基线）。损失几乎不动（v3b、v3h）说明剂量太小，不要把它读成“已收敛”。
- SimPO 里 `margin = β·(a_w − a_l) − γ`（`a_w`、`a_l` 是选中 / 拒绝序列的平均 token 对数概率），`loss = log(1 + e^(−margin))`；`pref_acc` 是该次更新里 `a_w > a_l` 的偏好对占比。`margin` 为负说明选中侧还没有比拒绝侧高出 γ/β。每次更新只累积 2 对，`pref_acc` 只有 0、0.5、1 三种取值，噪声很大。
- `sec` 是单次更新的墙钟秒数；`peak_gib` 是 PyTorch 显存峰值。Orin 的 GPU 可见内存约 61 GiB（统一内存，系统和其他进程还要占去约 9 GiB），窗口 24–32K 时单次更新（SFT 累积 4–8 条序列，SimPO 累积 2 对）约 13–45 分钟。
- MTP 头的 `eval_top1` 是**教师强制**下的一致率，不等于线上草稿接受率（线上还受采样、草稿长度和上下文影响）；两者的关系见技术报告 §8.1 和 `final-eval-2026-10-08/README.md`。

## 想重新跑评测

- 旧评测台路径（快照里的数字）：`harness/sr_eval/`，命令见 [`docs/reproduction.md`](../docs/reproduction.md) 的评测一节。
- 线上协议路径（终版总评测）：`harness/sr_eval/live/` 加 `harness/tools/final_eval.sh`；汇总用 `final_eval_summary.py`、`server_log_stats.py`、`slot_concurrency.py`、`live_compare.py`。需要线上系统的提示词模板与 `report_harness` 模块（不在本仓库，见 [`docs/handover.md`](../docs/handover.md) 第 2 节）。
- 评审者和教师需要自己准备外部模型接口与密钥；仓库里没有密钥，调用前先读 `harness/README.md` 的“费用与外发闸门”一节和 `harness/config/outbound.json`。
