# 终版总评测:逐案件与逐调用明细(2026-10-08)

只有 token 计数与状态,不含任何案件文本。基线 = v3f 全流程(预算 4000、f16 KV、内置采样 0.8/40/0.95/0.05);终版 = 交付配置(预算 5120、q8_0 KV、官方采样 1.0/20/0.95/0)。

## 逐案件状态

| 案件 | 基线:状态 / 原因 | 基线:调用 / 被截断 / 最大提示 | 终版:状态 / 原因 | 终版:调用 / 被截断 / 最大提示 |
| --- | --- | --- | --- | --- |
| syn_dev_000 | needs_review / invalid_role_response | 4 / 0 / 16082 | published / – | 2 / 0 / 15817 |
| syn_dev_003 | published / – | 1 / 0 / 23808 | needs_review / invalid_role_response | 3 / 3 / 24063 |
| syn_dev_004 | published / – | 2 / 0 / 16788 | published / – | 1 / 0 / 16627 |
| syn_dev_006 | published / – | 4 / 0 / 19682 | published / – | 2 / 0 / 16033 |
| syn_dev_008 | failed / evidence_not_found | 4 / 0 / 19036 | published / – | 4 / 0 / 19449 |
| syn_dev_010 | needs_review / invalid_role_response | 1 / 0 / 22397 | needs_review / invalid_role_response | 4 / 3 / 28645 |
| syn_dev_011 | published / – | 1 / 0 / 21918 | published / – | 1 / 0 / 21918 |
| syn_dev_012 | published / – | 3 / 0 / 18933 | failed / evidence_not_found | 3 / 0 / 18979 |
| syn_dev_013 | published / – | 4 / 0 / 20082 | published / – | 4 / 0 / 19700 |
| syn_dev_018 | needs_review / invalid_role_response | 1 / 0 / 21920 | needs_review / invalid_role_response | 1 / 0 / 21920 |
| syn_dev_027 | needs_review / training_window_exceeded | 4 / 2 / 28664 | needs_review / invalid_role_response | 4 / 2 / 27959 |
| syn_dev_038 | published / – | 2 / 0 / 22198 | needs_review / invalid_role_response | 5 / 3 / 27784 |

## 终版逐调用(生成者调用)

| 案件 | 轮次 | 提示 token | 思考 token | 答案 token | 总长 | 结束原因 |
| --- | --- | --- | --- | --- | --- | --- |
| syn_dev_000 | 0 | 15656 | 5148 | 3434 | 24238 | stop |
| syn_dev_000 | 1 | 15817 | 5144 | 3683 | 24644 | stop |
| syn_dev_003 | 0 | 23808 | 5148 | 3780 | 32736 | length |
| syn_dev_003 | 1 | 23969 | 5148 | 3618 | 32735 | length |
| syn_dev_003 | 2 | 24063 | 5148 | 3527 | 32738 | length |
| syn_dev_004 | 0 | 16627 | 5144 | 4362 | 26133 | stop |
| syn_dev_006 | 0 | 15793 | 5155 | 2144 | 23092 | stop |
| syn_dev_006 | 1 | 16033 | 5147 | 3689 | 24869 | stop |
| syn_dev_008 | 0 | 17451 | 5148 | 4304 | 26903 | stop |
| syn_dev_008 | 1 | 17616 | 191 | 47 | 17854 | stop |
| syn_dev_008 | 2 | 19279 | 5148 | 3901 | 28328 | stop |
| syn_dev_008 | 3 | 19449 | 5148 | 3604 | 28201 | stop |
| syn_dev_010 | 0 | 22397 | 663 | 43 | 23103 | stop |
| syn_dev_010 | 1 | 28390 | 4348 | 2 | 32740 | length |
| syn_dev_010 | 2 | 28551 | 4188 | 2 | 32741 | length |
| syn_dev_010 | 3 | 28645 | 4094 | 2 | 32741 | length |
| syn_dev_011 | 0 | 21918 | 2380 | 2844 | 27142 | stop |
| syn_dev_012 | 0 | 18564 | 5145 | 3780 | 27489 | stop |
| syn_dev_012 | 1 | 18755 | 5148 | 4204 | 28107 | stop |
| syn_dev_012 | 2 | 18979 | 1198 | 71 | 20248 | stop |
| syn_dev_013 | 0 | 17626 | 5147 | 2255 | 25028 | stop |
| syn_dev_013 | 1 | 17866 | 5148 | 4338 | 27352 | stop |
| syn_dev_013 | 2 | 18035 | 414 | 25 | 18474 | stop |
| syn_dev_013 | 3 | 19700 | 5148 | 5194 | 30042 | stop |
| syn_dev_018 | 0 | 21920 | 5148 | 3700 | 30768 | stop |
| syn_dev_027 | 0 | 21498 | 5148 | 4257 | 30903 | stop |
| syn_dev_027 | 1 | 21684 | 670 | 109 | 22463 | stop |
| syn_dev_027 | 2 | 27865 | 4871 | 2 | 32738 | length |
| syn_dev_027 | 3 | 27959 | 4779 | 2 | 32740 | length |
| syn_dev_038 | 0 | 21960 | 366 | 23 | 22349 | stop |
| syn_dev_038 | 1 | 23623 | 317 | 334 | 24274 | stop |
| syn_dev_038 | 2 | 27529 | 5148 | 60 | 32737 | length |
| syn_dev_038 | 3 | 27690 | 5047 | 2 | 32739 | length |
| syn_dev_038 | 4 | 27784 | 4953 | 2 | 32739 | length |

共 34 次调用,以 `length`(槽位上下文用尽)结束 11 次;被截断的调用提示长度中位 27784,正常结束的中位 18755。
