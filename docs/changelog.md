# 更新记录

| 日期 | 内容 |
| --- | --- |
| 2026-10-07 | 初始化仓库：v1 首版发布包、v3 谱系 SFT 数据、线上环境后训练集 v3h（26 行：16 对偏好 + 10 锚点）、评测台与线上环境模拟器、Orin 训练/合并/量化/服务脚本、线上提示词与 schema、compact 推理档位模板、技术报告 v0.1。v3h 训练进行中，结果待补。 |
| 2026-10-07（晚） | v3h 训练完成（13 次更新，4 h 50 min，损失几乎不动，剂量太小）；仓库所有者决定跳过快评直接做第二轮；新增第二轮数据 `live-post-training-v3h2`（28 对）与 `mtp-retrain-data`（60 条）、`mtp_chain.sh`/`auto_final.sh`/`make_mtp_data.py`/`speed_probe.py`/`live_compare.py` 等脚本；`build_post.py` 增加 `--exclude-calls`/`--anchor-allowlist`/`--audit-fraction`；技术报告补 v3h 训练结果与 §6.8；v3h2 训练中，结果待补。 |
