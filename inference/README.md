# 推理程序（Orin 上的交付形态）

本目录收拢 **模型推理用到的程序文件**：llama.cpp 的改动补丁、启动与压测脚本、草稿词表子集工具。权重（LoRA、MTP 头、GGUF）不在仓库里。

## 交付形态

- 设备：Jetson AGX Orin 64GB（JetPack 7.2.1，SM87，MAXN + `jetson_clocks`）
- 引擎：llama.cpp，固定提交 `bed0a856606ee4a24a164066f73d2379447033f5`（`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`，FA 量化组合默认 `q4_0/q8_0/f16/bf16`），加本目录的补丁
- 模型：v3h2 合并后的 Q4_K_M 单个 GGUF（含重训的 MTP 头），不挂 LoRA；GGUF 本身与补丁无关，库存 llama.cpp 也能加载
- 服务参数：compact 推理档位 + 思考预算 5120 + 6 槽位 × 每槽位 40960 上下文（2026-10-08 由 32768 放宽，窗口取舍见 `docs/deployment.md` §4）+ KV 缓存 q8_0 + MTP 草稿（`--spec-type draft-mtp`，草稿上限 5）
- 服务端默认采样：模型官方推荐值（`generation_config.json`：temperature 1.0、top_k 20、top_p 0.95、min_p 0）。线上请求不带采样参数，所以由服务端默认值决定；请求里带了则以请求为准。**此前 llama-server 用的是内置默认 0.8 / 40 / 0.95 / 0.05**，历次评测都在这个默认下做的，见技术报告。
- 入口：`harness/orin/serve_delivery.sh <GGUF 基名>`：优化构建 + 草稿词表子集（`SR_MTP_DRAFT_IDS`）+ `SPEC_N_MAX=5` + `SR_SPEC_BATCH_TOKENS=24`（按在跑槽位数自适应草稿长度：6 路时每路 3 个、≤4 路时 5 个）+ 官方采样；`SR_BASELINE=1` 回到库存构建、不用子集和自适应。

## 目录

| 路径 | 内容 |
| --- | --- |
| （Orin 上的位置） | `harness/orin/` 的脚本整个放在 `~/work/scripts/`，再加 `harness/tools/` 里的 `probe_multi.py`、`speed_probe2.py`、`answer_probe.py`、`answer_multi.py`、`bench_table.py`、`probe_compare.py`、`draft_log_compare.py`；下表里 `harness/orin/`、`harness/tools/` 的脚本在 Orin 上都在这一个目录 |
| `inference/patches/` | 对 llama.cpp 的补丁（对上游提交 `bed0a85` 干净应用），见下 |
| `inference/draft-vocab/` | MTP 草稿词表子集的 token id 列表与覆盖率报告（不是权重） |
| `harness/orin/serve_llama.sh` | 通用启动脚本。环境变量见脚本头注释：`LLAMA_DIR`、`SPEC_N_MAX`、`SPEC_P_MIN`、`SPEC_TYPES`（逗号分隔，顺序即优先级）、`NGRAM_N/NGRAM_M`、`SR_TEMP/SR_TOP_K/SR_TOP_P/SR_MIN_P`（服务端默认采样）、`KV_TYPE`、`EXTRA_ARGS`、`WRAP`、`DRY_RUN`；补丁相关的 `SR_MTP_DRAFT_IDS`、`SR_SPEC_BATCH_TOKENS`、`SR_MMVQ_MAX_NE11`、`SR_CUBLAS_MIN_NE11` 直接设在环境里 |
| `harness/orin/serve_delivery.sh` | 交付配置启动入口 |
| `harness/orin/prefill_study.sh` | 批大小曲线（一次前向处理 n 个 token 的耗时）、预填充对照（MMQ 与反量化 + cuBLAS）、KL 散度数值核对、nsys 预填充剖析 |
| `harness/orin/region_sweep.sh`、`region_configs*.txt` | **分区间**对照：每个配置先测思考区间、再测答案区间（见下“为什么分区间”） |
| `harness/orin/multi_batch.sh`、`multi_configs.txt`、`multi_run.sh` | 多槽位并发吞吐（同一提示 N 路同时生成） |
| `harness/orin/answer_multi_batch.sh`、`answer_multi_configs.txt`、`answer_sweep.sh`、`answer_configs.txt` | 答案区间的并发与单路对照 |
| `harness/orin/bench_sweep.sh`、`bench_configs_decode*.txt` | 单路逐配置对照（温度 0 与默认采样各测一遍） |
| `harness/orin/prof_decode.sh`、`prof_multi.sh`、`prof_summary.py` | 用 nsys 抓解码段（单路 / 6 路）并汇总到内核类别 |
| `harness/orin/shortlist_scatter_test.cpp` | 草稿词表子集那条算子链的单元测试（CPU + CUDA，真实词表尺寸） |
| `harness/orin/bw_probe.cu`、`sampler_cost.cpp` | Orin 实际可持续读带宽探针；CPU 采样单次开销 |
| `harness/orin/dump_hidden.cpp`、`eval_head_on_served.py`、`served_feature_check.sh` | 诊断“训练特征与服务特征是否失配”：用 llama.cpp 导出服务模型的末层隐藏状态，再用 HF 里的 MTP 头算教师强制一致率 |
| `harness/orin/mtp_chain.sh`、`train_mtp.py`、`make_gguf_v3.sh` | MTP 头重训链、合并转换量化 |
| `harness/tools/speed_probe2.py`、`probe_multi.py`、`answer_probe.py`、`answer_multi.py` | 单路 / 多路 / 答案区间速度探针 |
| `harness/tools/probe_compare.py`、`draft_log_compare.py`、`bench_table.py`、`region_table.py`、`multi_table.py` | 文本逐字比对、草稿候选逐步比对、各类对照表 |
| `harness/tools/draft_vocab.py` | 词表子集生成 |
| `harness/tools/slot_concurrency.py`、`server_log_stats.py` | 从服务日志统计并发槽位数分布、每次调用的预填充/解码速度与草稿接受率 |
| `harness/tools/ngram_potential.py` | 离线估算 n-gram 查表式投机在真实输出上的潜力 |
| `harness/tools/fetch_decode_results.sh` | 把 Orin 上的实验结果取回本机 |

## 补丁

### 0001-mtp-draft-vocab-shortlist.patch（草稿词表子集，借鉴 ninfer 的 proposal head）— 采用

MTP 草稿的每一步要读 MTP 块（0.263 GB）加整个输出头（Q6_K，1.043 GB），**80% 的字节花在输出头上**。草稿只需要“猜下一个最可能的 token”，不必在全部 248320 个词里选，所以只用按词频排序取出的 K 行：

- 模型载入后，把 `output` 里指定 token 的行（任意量化类型的行互相独立，直接按字节复制）拷成一个小张量；MTP 草稿图只算这 K 行的 logits，再散射回完整词表的位置（其余填负无穷），后面的采样器（后端或 CPU）不用改。
- 验证用的主模型头仍是完整词表，所以**不改变输出**，只省草稿阶段读输出头的字节；猜不到子集外的 token 只是少一次命中（K=32768 时领域覆盖 99.83%）。
- 由环境变量 `SR_MTP_DRAFT_IDS` 指向 int32 小端的 id 文件启用，不设就和原版完全一样；**GGUF 文件本身不变，库存 llama.cpp 照样能加载**。

### 0002-orin-mmvq-batch-limit-env.patch（实验开关）— 实测无收益，默认关

上游把 k-quant 在 Orin（算力 8.7）上的 MMVQ 限制在 1 列，更多列走 MMQ。补丁加环境变量 `SR_MMVQ_MAX_NE11` 覆盖这个上限，用来实测验证批的最优分流。**结论：放宽反而更慢**（见下表），上游的取值是对的，保留补丁只为复现这个负结果。

### 0003-adaptive-draft-length.patch（按负载自适应草稿长度）— 采用

一次验证的批次是“每个在跑槽位 ×（1 + 草稿长度）”个 token。6 槽位、草稿 5 时约 36 个 token，已经是**算力受限**，被拒的草稿白占算力。补丁做三件事：

- MTP 草稿循环遵守每槽位的 `n_max`（原实现总是画满再截断）；
- 服务端按在跑槽位数限制草稿长度：环境变量 `SR_SPEC_BATCH_TOKENS=N`（缺省 0=关）时，每槽位草稿上限为 `max(1, N / 在跑槽位数 - 1)`；单路仍用 `--spec-draft-n-max`。交付用 N=24：6 路时每路 3 个、5 路 3 个、≤4 路 5 个；
- 把“每个位置的草稿接受率”（`acc per pos`）从跟踪日志级别提到默认日志级别，方便按数据选草稿长度。

### 0004-prefill-cublas-switch.patch（预填充可选走反量化 + cuBLAS）— 实测无收益，默认关

上游对 Orin 的量化权重矩阵乘一律走 MMQ（int8 张量核心）。补丁加环境变量 `SR_CUBLAS_MIN_NE11=N`：批次列数 ≥ N（即预填充）的**量化权重**矩阵乘改走“反量化成 bf16 + cuBLAS”（bf16 输入、fp32 累加；`SR_CUBLAS_QUANT_PREC=f16` 可换回上游的 fp16 选择做对照）。浮点权重的走法、解码和草稿验证批都不受影响。**结论：不快**（见下表），保留只为复现。

### 应用与构建

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout bed0a856606ee4a24a164066f73d2379447033f5
patch -p1 < <本仓库>/inference/patches/0001-mtp-draft-vocab-shortlist.patch
patch -p1 < <本仓库>/inference/patches/0002-orin-mmvq-batch-limit-env.patch
patch -p1 < <本仓库>/inference/patches/0003-adaptive-draft-length.patch
patch -p1 < <本仓库>/inference/patches/0004-prefill-cublas-switch.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target llama-server llama-bench llama-cli test-backend-ops
```

（Orin 上直连 huggingface.co 不通，构建时 UI 资源下载会超时，对服务端无影响。**增量构建要小心**：编辑源文件的同时后台正在 make，会编出旧内容却带着新时间戳——判断构建是否包含改动要看二进制的行为或日志，不能只比时间戳。）

## 草稿词表子集

`harness/tools/draft_vocab.py` 用线上协议下学生模型自己输出的 token 做领域计数，加 ninfer 公开的通用词频（`tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64`，Apache-2.0，**不随本仓库分发**，自行下载）做先验，按“领域频率 + 0.1 × 先验频率”排序取前 K 个，特殊 token 一律纳入；按案件 5 折交叉验证，报告“模型实际输出的 token 落在子集内”的比例。`draft-vocab/` 里是 Qwen3.8 词表（248320 行）的 K=16384/32768/65536 三份 id 列表，换分词器/词表必须重新生成。

## 验证（如实）

### 为什么分区间

一次调用的两段差别很大：**思考**约 4 到 5K token、占约四分之三的时间，草稿接受率只有四成上下；**答案**是结构化 JSON、大量照抄证据原文，接受率约 95%。常规探针只生成思考开头的几百个 token，既低估答案段、也测不出长上下文，所以用三类探针：批大小曲线（`llama-bench`）、区间探针（思考区间 = 完整提示从头生成 1500 token；答案区间 = 完整提示 + 学生真实的思考文本之后续写 1200 token）、N 路并发探针。全部在交付配置下（Q4_K_M、KV q8_0、预算 5120、6 槽位 × 32768；窗口放宽到 40960 之前的测量，解码速度与窗口基本无关）。交付形态下 62% 的时间 6 路同时在生成、31% 是 5 路，有预填充在进行的时间约占 33%（`slot_concurrency.py` 对早先全流程服务日志的统计；终版总评测期间实测 6 路同时在生成占 44%、有预填充在进行占 21%，见 `eval/final-eval-2026-10-08/README.md`）。

### 机器上限与批大小曲线

可持续读带宽最高 162.6 GB/s（`bw_probe`）。一次前向处理 n 个 token 的耗时（MMQ 默认路径，ms）：

| n | 1 | 6 | 8 | 12 | 16 | 24 | 32 | 36 | 48 | 64 | 512 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ms | 137 | 150 | 151 | 170 | 173 | 232 | 268 | 293 | 319 | 366 | 2146 |

8 个 token 以内的验证批几乎不比单 token 贵；之后每多一个 token 约贵 3 到 7 ms，6 路并发的验证批（路数 ×（1+草稿长度））落在算力受限的区间。预填充约 238 token/s；nsys 看预填充：MMQ 占 71.9%、线性注意力内核 12.6%、元素/拷贝 8.2%，GPU 空档只有 0.6%，已接近这台机器的持续算力上限。

### 词表子集（补丁 0001）：思考与答案区间都快 17%，输出逐位不变

| 配置（单路，默认采样，种子 11） | 思考 token/s | 接受率 | 答案 token/s | 接受率 |
| --- | --- | --- | --- | --- |
| 库存构建，草稿 5 | 13.48 | 0.381 | 26.11 | 0.951 |
| 词表子集 K=32768，草稿 5 | 15.81 | 0.381 | 30.52 | 0.951 |

两种构建的生成文本 SHA 相同、草稿数与接受数相同（思考 982/2580，答案 990/1041）。6 路并发时合计 25.0 → 27.5 token/s（+10%，并发时算力受限，收益小）。

### 草稿长度与并发（官方采样，词表子集，思考开头段）

| 配置 | 路数 | 合计 token/s | 每路平均 |
| --- | --- | --- | --- |
| 库存，草稿 5 | 6 | 25.0 | 4.87 |
| 草稿 5 | 6 | 27.5 | 4.81 |
| 草稿 3 | 6 | **31.2** | 5.82 |
| 草稿 2 / 草稿 1 | 6 | 29.8 / 29.5 | 5.34 / 5.38 |
| 草稿 8 | 6 | 25.1 | 5.25 |
| 验证批预算 16 / 24 | 6 | 30.2 / 29.1 | 5.25 / 5.79 |
| 置信度门限 0.5 / 0.7 | 6 | 15.3 / 13.6 | 2.82 / 2.42 |
| 草稿 5 | 3 | 20.3 | 7.16 |
| 验证批预算 16（每路 4 个） | 3 | 24.6 | 8.76 |
| 草稿 5 | 1 | 14.4 | 14.5 |
| 草稿 5，答案区间（提示约 2 万 token） | 6 | 54.0 | 9.15（接受率 99.5%） |

交付取 `SPEC_N_MAX=5` + `SR_SPEC_BATCH_TOKENS=24`：6 路时每路 3 个草稿（6 路思考段最优），≤4 路时 5 个（单路与少路沿用 5，答案段接受率高时长草稿划算）。草稿长度 3 与 5 的差别（约 6%）小于单样本噪声，所以取了覆盖面最广的预算式自适应。这些并发数字用约 5K token 的提示，真实调用的上下文是 16 到 25K，并发时长上下文的注意力与线性注意力状态快照还会多出开销（没有用 nsys 拆开看）。

### 试了没采用的

| 做法 | 结果 |
| --- | --- |
| 放宽 MMVQ 到 8 列（补丁 0002） | 更慢：6 列 209 ms（默认 150 ms）、8 列 261 ms；只有 2 列略好（134 对 145 ms） |
| 预填充走反量化 + cuBLAS（补丁 0004） | bf16 ubatch 512：209 token/s（MMQ 238）；ubatch 2048：247.6，只比 MMQ 最好值快 3.7%；fp16 更差。与 MMQ 的 KL 散度均值 0.20（中位数 0.0024，尾部很长），数值差异比预期大，原因没查 |
| n-gram 查表在前、MTP 兜底 | 离线估算（`ngram_potential.py`）答案段 56% 的步有草稿、整体每步推进 3.4 个 token；实测答案段反而更慢（23.4 对 29.9 token/s），因为答案段 MTP 本身接受率已有 93% 以上 |
| 置信度门限 p_min 0.5 / 0.7 | 单路思考区间更慢（12.6 对 15.8），6 路并发合计几乎腰斩（15.3 / 13.6）；原因未查（猜测是逐步逐槽位变化的草稿长度打断图复用） |
| 概率式草稿采样 | 默认采样下思考区间 +12%、答案区间 +2%；官方采样下思考区间 +2.4%、答案区间 −7%（接受率 87% 对 95%） |
| 单路草稿长度 8 | 答案区间 +19%、思考区间 −12%，按一次调用的思考与答案 token 数加权后 5 更好 |
| 草稿长度 3（单路） | 答案区间只有 22.4 token/s（5 为 30.6） |

### 训练特征与服务特征没有失配

怀疑过：新头训练用的是 NF4 + LoRA 的特征，而服务的是合并后的 Q4_K_M。用 `dump_hidden` 导出服务模型的末层隐藏状态，再用 HF 里的头算教师强制一致率（留出集前 3 条，约 2.46 万个监督 token）：新头 0.926 / 0.878 / 0.844，官方头 0.922 / 0.853 / 0.798，不低于训练时的数字（新头 0.907 / 0.857 / 0.819），所以不需要按服务特征重训。思考区间接受率偏低主要是采样温度和文本本身难预测。
