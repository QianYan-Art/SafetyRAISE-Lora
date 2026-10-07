# 结果快照(2026-10-06 03:24)

口径:luna(max) 严格评审总分(满分 25),**未经人工核实**。评审、硬门、思考指标均由 `harness/` 评测台产出;原始轨迹/评审在各 run 目录。

## 开发集前 12 案(dev_000–011)

| 系统 | 有效评审 | 均分 | 每份缺陷 | 硬门失败 | 报告字数中位 | 检索轮数 | 总思考字符中位 | 最终步思考字符中位 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 基座(云端xhigh) | 12/12 | 15.67 | 6.42 | 1/12 | 3725 | 1.92 | 49226 | 14185 |
| v2a | 12/12 | 18.92 | 4.83 | 1/12 | 1760 | 2.00 | 520 | 206 |
| teacher(luna) | 12/12 | 21.42 | 3.17 | 2/12 | 2105 | 1.83 | 2423 | 1649 |
| v3b | 12/12 | 16.08 | 6.42 | 4/12 | 3391 | 1.33 | 13075 | 6877 |
| v3c | 11/12 | 19.64 | 4.09 | 2/12 | 2281 | 1.33 | 12216 | 10687 |
| v3d | 12/12 | 19.25 | 4.08 | 1/12 | 2388 | 1.17 | 12497 | 11309 |
| v3e(失败) | 11/12 | 17.36 | 5.27 | 5/12 | 3040 | 1.17 | 6940 | 6239 |
| v3f | 12/12 | 20.42 | 3.25 | 1/12 | 1872 | 1.08 | 8660 | 7459 |

## 完整 50 个开发案(dev_000–049;基座与 v2a 只覆盖 dev_000–032)

| 系统 | 有效评审 | 均分(95% CI) | 硬门失败 | 配对差 vs teacher(95% CI) | 胜/平/负 |
| --- | --- | --- | --- | --- | --- |
| 基座(云端xhigh) | 33 | 16.18 (15.30~17.06) | 3/33 | -5.48 (-6.48~-4.48) | 1/1/31 |
| v2a | 33 | 19.27 (18.64~19.91) | 1/33 | -2.39 (-3.03~-1.76) | 3/3/27 |
| v3d | 50 | 19.58 (18.94~20.20) | 4/50 | -1.66 (-2.38~-0.98) | 9/9/32 |
| v3f | 49 | 19.82 (19.12~20.49) | 8/50 | -1.39 (-2.16~-0.65) | 13/7/29 |
| teacher(luna) | 50 | 21.24 (20.82~21.64) | 3/50 | – | – |

## 运行目录与产物位置

- 基座(云端xhigh) / `baseapi_dev33`:`harness/runs/20261003_092633_baseapi_dev33`(轨迹 33 份;`gates.json`、`judgements.json`)
- v2a / `sft_dev50`:`harness/runs/20261003_054845_sft_dev50`(轨迹 33 份;`gates.json`、`judgements.json`)
- teacher(luna) / `lunadev`:`harness/runs/20261002_105643_lunadev`(轨迹 20 份;`gates.json`、`judgements.json`)
- teacher(luna) / `lunadev2`:`harness/runs/20261003_033401_lunadev2`(轨迹 30 份;`gates.json`、`judgements.json`)
- v3b / `v3b_dev12`:`harness/runs/20261003_213114_v3b_dev12`(轨迹 12 份;`gates.json`、`judgements.json`)
- v3c / `v3c_dev12`:`harness/runs/20261004_063534_v3c_dev12`(轨迹 12 份;`gates.json`、`judgements.json`)
- v3d / `v3d_dev12`:`harness/runs/20261004_163906_v3d_dev12`(轨迹 12 份;`gates.json`、`judgements.json`)
- v3d / `v3d_dev12to49`:`harness/runs/20261004_183118_v3d_dev12to49`(轨迹 38 份;`gates.json`、`judgements.json`)
- v3e(失败) / `v3e_dev12`:`harness/runs/20261005_022542_v3e_dev12`(轨迹 12 份;`gates.json`、`judgements.json`)
- v3f / `v3f_dev12`:`harness/runs/20261005_193242_v3f_dev12`(轨迹 12 份;`gates.json`、`judgements.json`)
- v3f / `v3f_dev12to49`:`harness/runs/20261005_211430_v3f_dev12to49`(轨迹 38 份;`gates.json`、`judgements.json`)

- 训练数据:`harness/sft/{v3b,v3c,v3d,v3e_ret,v3f,v3e_gate,v3g_gate}/`(`manifest.json` 记录数量、来源、筛选统计)
- 盲评材料与 Codex/Kimi 输出:`harness/runs/blind/<名>/`(答案映射 `harness/runs/blind/answers_<名>.json`)
- 自动链日志:`harness/runs/{auto_eval_v3*.log,big_loop.log,expert_iter.log,eval_rest_*.log}`;Orin 训练日志在 Orin `~/work/runs/*_sft/log.jsonl`(已同步摘要见 `harness/reports/orin-train-logs/`)
