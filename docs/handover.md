# 接手与续训指南

写给第一次接手这个仓库和那台 Orin 的人：读完应该知道现在有什么、东西放在哪、怎么接着训、先做什么、哪些方向已经证明走不通。状态截至 2026-10-08。细节在 [`technical-report.md`](technical-report.md)，命令在 [`reproduction.md`](reproduction.md)，推理优化在 [`../inference/README.md`](../inference/README.md)。

## 0. 一页纸：现状

- **模型**：Qwen3.8-27B + 后训练 LoRA（v3h2，rank 16、只适配第 32 到 63 层）合并后量化的单个 **Q4_K_M GGUF**，MTP 草稿头已针对该模型重训。服务入口 `harness/orin/serve_delivery.sh`（compact 档位、预算 5120、KV q8_0、6 槽位 × 每槽位 40960、官方采样）。
- **质量（开发 12 案全流程，单次采样，自动口径，未经人工核实）**：发布 6、待审 5、失败 1（v3f 基线 7 / 4 / 1，差别不可分辨）；首调用通过全部确定性检查 4/12（v3f 1/12）；与教师的差距（v3f：评审配对差 −1.39，独立盲评约 −1.69 分，满分 25）和硬门失败率（v3f 8/50 对教师 3/50）是最近一次实测的数字，终版总评测里没有重测，**均尚未达到既定通过线**。详见 [`../eval/final-eval-2026-10-08/README.md`](../eval/final-eval-2026-10-08/README.md)。
- **窗口**：终版总评测在每槽位 32768 下跑，34 次调用里有 11 次被截断（全是提示已超约 23K 的案件）；2026-10-08 仓库所有者决定把交付窗口放宽到 40960（`serve_delivery.sh` 缺省，已验证 6 槽位能正常启动与生成），按评测的实际 token 数回算这 11 次都放得下，但**放宽后没有重跑评测**，见第 6 节。
- **速度**：单路解码思考区间约 16 token/s、答案区间约 31 token/s；预填充约 238 token/s（已接近这台机器的持续算力上限）；6 路并发合计约 27 到 31 token/s。端到端没有比基线快，原因未查清。
- **权重不在仓库里**，随 Orin 机器一起交接（第 1 节有路径和校验和）。

## 1. 东西在哪里

仓库只放数据、脚本、提示词和文档；以下权重和基座在 Orin 上（家目录下），**校验和用来核对交接时没有损坏或被替换**：

| Orin 路径 | 内容 | 大小 | SHA-256 |
| --- | --- | --- | --- |
| `~/models/Qwen3.8-27B/` | 官方 BF16 基座（合并、训练 MTP 头用；对应 `Qwen/Qwen3.8-27B`） | 约 52 GB | – |
| `~/models/Qwen3.8-27B-unsloth-bnb-4bit/` | 4-bit 基座（QLoRA 训练用；对应 `unsloth/Qwen3.8-27B-bnb-4bit`） | 约 21 GB | – |
| `~/models/gguf/v3h2m-Q4_K_M.gguf` | **交付模型**（v3h2 合并 + 重训 MTP 头 + Q4_K_M） | 16,810,714,624 B | `7ab181f97266cc02d61fd5bc952fcd5d7261a3ab224f75f8e1f66dc050d77f58` |
| `~/work/runs/v3h2_sft/ckpt/adapter_model.safetensors` | **v3h2 适配器**（交付版本；v3f 上两轮后训练） | 约 208 MB | `838630f594f67032f9a36f07a223a4a1845eff9c4e33c94ff0dffec02b6e8771` |
| `~/work/runs/v3f_sft/ckpt/adapter_model.safetensors` | v3f 适配器（v3d 上专家迭代；后训练的起点；硬门失败比 v3d 多） | 约 208 MB | `7fa59d73ba902312cc2840feb3868a1e95ea0d359234716b79b07cba3a226903` |
| `~/work/runs/v3d_sft/ckpt/adapter_model.safetensors` | v3d 适配器（硬门失败最少的版本） | 约 208 MB | `b83be4a9a6e3f365534d384ca1a060c35bc24d970557f7ad1ed15df34d59b95b` |
| `~/work/mtp/out_v3h2/mtp.safetensors` | 为 v3h2 重训的 MTP 头（15 个 `mtp.*` 张量，bf16） | 849,400,472 B | `7a5cb37fe0e441a4710838081f9636e689fac9b3be7ea983a5f28fc2d649a2a4` |

三个适配器的 `adapter_config.json` 完全相同（SHA-256 `57d3a3decd601657cf232ee1e574acadbd5e6a5dd3c02c1b697c21c0dd2de9cc`）。其余目录：

| Orin 路径 | 内容 |
| --- | --- |
| `~/work/scripts/` | 仓库 `harness/orin/` 的内容（整个复制过去；各脚本按这个位置互相调用），**再加上 `harness/tools/` 里的 7 个小工具**：`probe_multi.py`、`speed_probe2.py`、`answer_probe.py`、`answer_multi.py`、`bench_table.py`、`probe_compare.py`、`draft_log_compare.py`（解码研究脚本会从同一目录调用它们）。`dump_hidden`、`bw_probe`、`sampler_cost` 等 C++/CUDA 诊断程序按各自源文件头注释里的编译命令现编。Orin 家目录另有一份 `README_DELIVERY.md`（有什么、怎么启动、核对哈希），内容与本节一致 |
| `~/work/templates/` | 服务端用的 `compact.jinja` 聊天模板和思考预算提示语 `budget_msg.txt`（分别是仓库 `assets/templates/qwen3.8-compact.chat_template.jinja` 和 `assets/templates/budget_msg.txt` 的副本，内容一致，可核对 SHA-256） |
| `~/work/data/` | 训练与探针数据：`v3h2_pairs.jsonl`（第二轮后训练集）、`train.jsonl` / `eval.jsonl`（MTP 重训的训练 / 留出序列）、`v3f_train.jsonl` / `v3f_eval.jsonl`、`probe_trace_*.json` / `answer_trace_*.json`（速度探针用）、`draft_ids/`（草稿词表子集）。这些在仓库 `datasets/` 里都有（压缩版） |
| `~/work/llama.cpp/`、`~/work/llama-opt/` | llama.cpp 提交 `bed0a856…`：库存构建 / 带补丁（`inference/patches/`）的优化构建，源码和 `build/bin` |
| `~/venvs/ft/` | 训练与合并用的 Python 环境（PyTorch、bitsandbytes、peft、transformers 等） |

## 2. 前置条件

- **Orin**：Jetson AGX Orin 64GB（JetPack 7.2.1，MAXN，`sudo jetson_clocks`）。训练 27B 需要约 45 到 50 GiB 统一内存，**训练和缓存隐藏状态前必须停掉 llama-server**；合并转换峰值磁盘约 80 GB，保持 ≥95 GB 空闲。
- **笔记本侧（采样、数据流水线、评测）**：Python 3.12 + `uv`，`requirements-eval.txt` / `requirements-live.txt`；分词器 `profiles/assets/*/tokenizer.json` 需自行下载并校验（见 `profiles/README.md`）。
- **线上系统模块**：线上环境模拟器需要 SafetyRAISE 线上系统的 `report_harness` 等模块（冻结在 `7200e30` 提交，仓库不含，哈希清单与提取脚本见 `harness/tools/vendor_prod.py`）。提示词与 schema 已放在 `assets/`。
- **外部模型**（教师修订、盲评）：走 `harness/config/models.json`，密钥用环境变量，**不写进仓库**；外发闸门见 `harness/config/outbound.json`（合成案件可外发，真实案件永不外发）。

## 3. 接着训练：四条路

先判断要改什么：

| 目标 | 起点 | 做法 | 主要命令 |
| --- | --- | --- | --- |
| **A. 在 v3h2 上继续后训练**（最常见） | v3h2 适配器 + 新的线上环境失败对 | 用线上环境模拟器采样学生输出 → 确定性评估 → 教师最小改动修订 → 构建偏好对/锚点 → SimPO + chosen NLL 训练 | 数据：`reproduction.md` 第 3 节；训练：`bash post_chain_final.sh ~/work/runs/v3h2_sft/ckpt <新TAG> <数据名> -- --beta 2.0 --gamma 0.4 --sft_lambda 1.0 --lr 1e-4 --accum 2 --epochs 1 --report_only --max_len 32768 --lora_from 32 --longest_first 2 --save_every 4` |
| **B. 加监督数据（SFT）** | 某版适配器 | 学生自采样 + 教师修订报告段，思考段低权重 | `train_sft.py --init_adapter <ckpt> …`（`reproduction.md` 第 2 节）；经验：剂量与案件多样性是关键，同一批案件上加样本收益趋零 |
| **C. 只重训 MTP 头** | 任一合并后的模型 | 缓存终版模型的隐藏状态，训 15 个 `mtp.*` 张量，覆盖合并 | `bash mtp_chain.sh <适配器目录> <TAG> <训练jsonl> <留出jsonl>`（失败后 `SKIP_CACHE=1` 续跑）；**主模型一变，MTP 头就要重训**，否则接受率掉 |
| **D. 只调解码/部署** | 现有 GGUF | 草稿长度、词表子集、采样、窗口 | `inference/README.md`；`serve_delivery.sh` 的环境变量 |

每次训练完（A 或 B）的收尾顺序：**合并 → 转 Q8_0 → 重量化 Q4_K_M → 重训 MTP 头并覆盖合并 → 按交付配置起服务 → 开发集评测**。`post_chain_final.sh` 串起前半段（合并用的 MTP 覆盖默认读 `~/work/mtp/out_v3h2/mtp.safetensors`，可用环境变量 `MTP_OVERLAY` 指定；新适配器训完后这个头是失配的，应尽快用 `mtp_chain.sh` 重训）。

## 4. 评测怎么做、怎么读

- 口径：线上环境模拟器、窗口 = 服务的每槽位上下文（`final_eval.sh` 缺省 `WIN=40960`；2026-10-08 的那次总评测是 32768，`WIN=32768` 可复现，服务端要用同样的 `SLOT_CTX` 起）、`--output-reserve 1 --max-tokens 32000`（让服务端自然截断），开发 12 案；入口 `harness/tools/final_eval.sh`（交付配置）或 `live_batch.py`（`reproduction.md` 第 5 节）。对比用 `harness/tools/live_compare.py`；服务端统计用 `server_log_stats.py`、`slot_concurrency.py`、`final_eval_summary.py`。
- **每个数字都是单次采样，12 案，噪声大**；不要只看“发布了几个”，要看首调用确定性检查、37 项解读指标、被截断的调用数、思考/答案长度。
- **对比的基线必须同采样、同预算、同 KV 精度**：此前的评测都在 llama-server 内置采样（0.8 / 40 / 0.95 / 0.05）下做，交付改成了模型官方推荐（1.0 / 20 / 0.95 / 0），两边不可直接比。
- **长评测要中途看内容类指标**（各调用的结束原因、案件状态），不要只看进度和服务是否健康：终版总评测里窗口截断到收尾汇总才被发现，早看就能早停。

## 5. 已被证伪、不要再试的方向

都有实测数字，详见 `inference/README.md` 与 `eval/final-eval-2026-10-08/decode-study/`：

- 放宽 k-quant 的 MMVQ 列数（补丁 0002）：更慢。
- 预填充改走反量化 + cuBLAS（补丁 0004）：最好只快 3.7%，且与 MMQ 的数值差异大（KL 散度均值 0.20）。
- n-gram 查表式投机与 MTP 组合：答案区间反而更慢。
- 草稿置信度门限 p_min（0.5 / 0.7）：单路更慢，6 路并发合计腰斩。
- 概率式草稿采样：官方采样下答案区间 −7%。
- “训练特征（NF4 + LoRA）与服务特征（Q4_K_M）失配”：诊断过，没有失配，不需要按服务特征重训 MTP 头。

## 6. 已知问题与建议的下一步（按优先级）

1. **窗口（2026-10-08 已决定放宽到每槽位 40960，待验证）**：终版总评测在 32768 下实测 11/34 调用被槽位上下文截断（基线预算 4000 时是 2/31），提示 28K 时只剩约 4.7K，思考和答案都出不来。交付窗口已改成 40960（`serve_delivery.sh` 缺省，已验证 6 槽位能正常启动与生成），按实际 token 数回算这 11 次都放得下。**还没做的**：（1）用 `WIN=40960 bash harness/tools/final_eval.sh` 重跑开发 12 案，确认截断消失、看长上下文下速度与质量；（2）提示超过约 30.6K 的长尾（首提示 >24K 的案件读一次片段后）仍要系统层按精确 token 数拆分，或给生成者请求设 `max_tokens`（窗口 − 提示 − 余量）并在提示长时降低思考预算；（3）训练窗口只到 32768，更长窗口下的行为没有评过，必要时补更长窗口的数据（`extension-guide.md` 第 1 节）。详见 `deployment.md` 第 4 节。**这是接下来最该先做的事。**
2. **与教师的差距和硬门失败率**：差距集中在“克制”（不过度展开）和“引用相关性”；后训练集规模很小（v3h 26 行、v3h2 28 对偏好），主要用于线上协议适配和 37 项解读，不是能力的大幅提升。要继续提升，需要更多“线上环境里的真实失败对”（路 A）并配更长窗口的数据。
3. **端到端速度**：单路微基准里的提升（词表子集 +17% 等）没有在总评测的端到端里体现为更快（3.0 小时对基线 2.5 小时，工作量大 10% 到 20%）。可能的原因（均未验证）：q8_0 KV 在长上下文注意力上比 f16 慢；官方采样下思考段接受率更低；6 路并发时验证批预算 24 不是长上下文下的最优。建议先做一组长上下文（提示约 16K）的 6 路并发对照：f16 / q8_0 KV × 库存 / 优化构建 × 固定草稿 3 或 5（配置模板见 `harness/orin/multi_configs.txt`，用 `TRIM=0 bash multi_batch.sh …`）。
4. **继续优化解码的方向**：线性注意力状态快照的 ReplaySSM（估计 10% 到 15%）；MMQ 小批量（17 到 24 个 token 一档比 16 个异常贵 59 ms）的瓦片参数调优；6 路并发的 nsys 剖析；思考区间接受率（占时间约四分之三）仍是最大杠杆。
5. **采样对齐**：线上请求不带采样参数，服务端默认值决定；若线上客户端以后显式传采样参数，应与服务端默认的官方值保持一致，否则等于换了模型的采样行为。
6. **思考预算**：服务端预算是硬截断，**没有被超过**（所有留存运行里思考 token 最大 = 预算 + 提示语：5155 对 5120，4030 对 4000，见 `../eval/final-eval-2026-10-08/README.md`）；但它是实际约束——4000 下 58% 到 83% 的首调用触顶，5120 下终版总评测 34 次生成者调用里 19 次触顶（写出完整报告的 19 次调用里 18 次），真实思考长度被截断、只知下界。窗口放宽到 40960 后有空间放开预算：按最长的观测提示 28.6K 和观测最大答案 5.2K 算，预算最高约 7000。但训练行里的思考前缀是在 4000 处截断的，放大预算会改变行为，应放开预算测一次真实分布并重新评测，再定预算。
7. **更长窗口数据**：超过 28000 的提示、多回合工具往返、修订轮（见 `extension-guide.md`），配合系统层“超限拆分”。

## 7. 踩坑（完整表见 `reproduction.md` 末尾）

- 训练、缓存隐藏状态时统一内存（约 61 GiB）已被占到 45 到 50 GiB，余量很小，任何额外的重负载（编译、探针）都可能挤掉任务；合并转换磁盘峰值约 80 GB。
- 偏好优化训练的第一个更新要数十分钟（~26–32K token 一对约 16–22 分钟），属正常；`--longest_first` 让最长行先跑，尽早暴露显存问题。
- 在同一条 ssh 命令里 `pkill -f '模式'` 会匹配到命令行自身而杀掉自己的 shell；先 `pgrep -f` 取 PID 再按 PID 杀。
- 编辑正在运行的 bash 脚本会错位；写成新文件再 `mv`。已打开的配置文件只在末尾追加。
- 增量构建时编辑源文件的同时后台在 make，会编出旧内容却带新时间戳；用二进制行为或日志判断构建是否含改动。

## 8. 边界与未决事项

- 所有案件为原创合成；评分来自模型评审与确定性检查，**未经人工核实**；模型输出是事故分析报告草稿，不是责任认定书。
- 真实案件材料永不进入本仓库、永不外发；权重不公开；凭据只用环境变量。
- **许可与公开范围已定（2026-10-08）**：Apache-2.0，与基座 Qwen3.8-27B 相同（根目录 `LICENSE`、`NOTICE`，README“许可与公开范围”一节，数据卡第 7 节）；`v1-synthetic-release` 清单里的 `outbound_eligible=false` 保持原样，按合成数据公开。仓库历史在 2026-10-08 重写过（去掉旧注释里的个人称呼，文件内容不变），**旧克隆请重新克隆**。
- 未决：线上 `report_harness` 的端点校验仅认固定域名，接入本地模型需要系统层改动；系统层的硬门失败重试或云端回退；窗口放宽到 40960 后要不要重跑总评测（建议重跑，见第 6 节第 1 条）。
