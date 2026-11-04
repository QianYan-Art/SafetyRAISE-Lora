# 权重编辑实验脚本（报告实验第二版，2026-11-03 至 11-04）

方法、结果与结论见 [`docs/weight-edit.md`](../../../docs/weight-edit.md)，汇总数字见 [`eval/weight-edit-2026-11-04/`](../../../eval/weight-edit-2026-11-04/README.md)。

**放置位置**
- Orin 上的脚本原本放在 `~/work/wsp/scripts/`，数据放在 `~/work/wsp/phase0`、`~/work/wsp/phase1`；带 `chain_` 前缀的脚本按这个布局互相调用。
- 笔记本侧的分析脚本要在实验根目录下运行（相对路径 `phase0/`、`phase1/`、`results/`）。

**依赖与约定**
- 依赖仓库的 `harness/orin/`：`serve_llama.sh`、`dump_hidden.cpp`、`train_mtp.py`。
- 依赖 Orin 上 llama.cpp 自带的 `gguf-py`。
- 服务地址缺省是 Orin USB 直连口 `192.168.55.1:8080`。
- 本目录下的 `serve_llama.sh` 与 `harness/orin/` 里的同名脚本一致。`serve_delivery_exp.sh` 只比 `serve_delivery.sh` 多一个 `GGUF_DIR` 参数，用来加载不在交付目录里的模型。

## 阶段 0 / 0b：诊断（笔记本侧分析）

| 脚本 | 作用 |
| --- | --- |
| `phase0_offline.py`、`phase0_loops.py`、`phase0_selfdup.py` | 在已有的线上环境轨迹上找循环与空转（严格 24-gram 循环、自我重复、思考里起草报告的次数） |
| `phase0_analyze.py`、`phase0_compare.py`、`phase0_draftdiff.py` | 不设思考预算重放后：思考是否自然结束、各稿之间改了什么、与预算 5120 版的对照 |
| `replay_client.py`、`reap_old_client.sh` | Orin 上的重放客户端，可断点续跑；后者回收被暂停的旧客户端 |
| `make_p1.py`、`make_p2.py` | 协议变体。P1：claims 留空、思考里不起草。P2：系统提示末尾追加“抓核心”分析要求 |
| `make_base.sh`、`chain_base.sh`、`chain_p2.sh` | 把官方 Qwen3.8-27B 转成 Q4_K_M 作对照；底座与 P2 的采样链 |
| `quality.py`、`hedge.py`、`core_focus.py` | 确定性质量检查、否定式用词密度、“抓核心”粗评。只用来辅助阅读，不能代替阅读 |

## 阶段 1：取方向与转向筛选

| 脚本 | 作用 |
| --- | --- |
| `make_pairs.py`、`prep_tokens.py` | 配对数据（同案同前文，A 取 P2 写法的四、五节，B 取原协议的四、五节）及其切词 |
| `dump_layers.cpp`、`p1_capture.sh` | 在服务模型（Q4_K_M）上取每层 l_out 在指定区间的均值。设 `CVEC_RAW` 时可同时叠加控制向量 |
| `compute_dir.py` | 按案件平衡的 A−B 均值方向，留一案件一致性检查；写出 `cv_core.gguf` 和同范数的随机对照 `cv_rand.gguf` |
| `make_probe.py`、`probe.py`、`chain_steer.sh`、`chain_steer2.sh`、`score_steer.py` | 续写探针（空思考，从“## 四、致因分析”续写到“## 六、”）、筛选链与评分 |
| `chain_p2cv.sh` | 完整协议确认：P2 加控制向量 0.25，开发 12 案 × 2 采样 |

## 阶段 2：固化成权重

| 脚本 | 作用 |
| --- | --- |
| `prep_fit_tokens.py` | 拟合序列：P2 无向量的完整输出（提示 + 思考 + 正文） |
| `dump_ffn.cpp`、`edit_stats.sh`、`edit_gram.sh` | 统计 `ffn_swiglu-<il>`（`down_proj` 的输入）各神经元的均值、方差、汇聚位与极值，以及候选神经元的 Gram 矩阵 |
| `edit_select.py` | 每层选 512 个稳定常开的神经元，排除汇聚位和长尾异常 |
| `edit_solve.py` | 用岭回归求读出向量 v（使 v·a≈1），并在留出序列上评估增益 |
| `edit_build.py`、`edit_make.sh` | 解量化 17 个 `ffn_down`，加 ΔW = c·u·vᵀ，存 F16，再用 `llama-quantize` 量化为 Q4_K_M（其余张量逐字节拷贝） |
| `edit_verify.sh`、`edit_reverify.sh`、`edit_compare.py` | 数值核对：在留出序列上比较编辑偏移与控制向量偏移，看逐层余弦和幅度比 |
| `chain_wsp.sh` | 编辑模型的行为验证链（中途按仓库所有者要求停止） |

## 阶段 3：MTP 头与验收

| 脚本 | 作用 |
| --- | --- |
| `mtp_feat_gguf.sh` | 用 `dump_hidden` 在编辑后的 Q4_K_M 上（KV q8_0、FA）导出服务特征，转成 `train_mtp.py` 的缓存格式 |
| `mtp_train_export.sh` | 依次执行：训练 → 删特征 → 换头导出 → 自检 → 测速 |
| `mtp_splice.py`、`mtp_splice_check.py` | 只替换 GGUF 里的 MTP 块（blk.64.*）：临时 HF 目录 + `convert_hf_to_gguf.py --mtp`，再走 q8_0 → `llama-quantize --allow-requantize`；自检会逐张量比较 |
| `speed_single.sh`、`speed_greedy.sh` | 单路测速。前者采样温度 1.0；后者贪心解码，用来对比新旧头（同一主模型下正文一致） |
| `chain_final_probe.sh` | 在最终 GGUF 上跑续写探针（原提示、P2 提示各 12 案） |
