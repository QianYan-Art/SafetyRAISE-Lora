# 更新记录

| 日期 | 内容 |
| --- | --- |
| 2026-10-07 | 初始化仓库：v1 首版发布包、v3 谱系 SFT 数据、线上环境后训练集 v3h（26 行：16 对偏好 + 10 锚点）、评测台与线上环境模拟器、Orin 训练/合并/量化/服务脚本、线上提示词与 schema、compact 推理档位模板、技术报告 v0.1。v3h 训练进行中，结果待补。 |
| 2026-10-07（晚） | v3h 训练完成（13 次更新，4 h 50 min，损失几乎不动，剂量太小）；仓库所有者决定跳过快评直接做第二轮；新增第二轮数据 `live-post-training-v3h2`（28 对）与 `mtp-retrain-data`（60 条）、`mtp_chain.sh`/`auto_final.sh`/`make_mtp_data.py`/`speed_probe.py`/`live_compare.py` 等脚本；`build_post.py` 增加 `--exclude-calls`/`--anchor-allowlist`/`--audit-fraction`；技术报告补 v3h 训练结果与 §6.8；v3h2 训练中，结果待补。 |
| 2026-10-07（深夜） | v3h2 训练完成（14 次更新，约 6 h 20 min）；开发 12 案首回合：通过确定性检查 1/12→4/12，义务一致性字段失败 26→6（新增 schema 额外键 2/12、评价字段缺归因 4 处）；思考预算触顶统计（4000 下 58%–83% 的首调用触顶、触顶者通过率明显更低）；量化对比 Q4_K_M/Q8_0（及 Q6_K 部分），**定交付 Q4_K_M**；交付配置：KV q8_0、思考预算 5120；新增 `serve_delivery.sh`、`quant_table.py`、`auto_quant_mtp.sh`；MTP 头重训进行中。 |
