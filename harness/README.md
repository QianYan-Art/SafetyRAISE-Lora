# 评测台(harness)

在**同一提示、同一检索器、同一硬门**下比较教师模型、未训练基线与微调模型,为蒸馏/微调提供自动验收口径。
所有结论均为自动口径(确定性硬门 + 模型评审),**未经人工核实**。纯标准库实现,无额外依赖。

## 它复刻了什么(生产旧路径,TS_analysis_report 2026-09-25 主线 7200e308)
- `assets/report_prompt.md`:冻结的生产报告提示(6 个占位符,来源与哈希见 `assets/ASSETS.json`)。
- 首轮检索 6 片段 → 模型可调 `retrieve_knowledge`(≤2 轮、每轮 ≤3 片段、总计 ≤9、query ≤120 字)→ 超限后不带工具强制收束。
- 与生产一致:**没有 tool 消息历史**,检索结果靠重新渲染 system 提示带入(`sr_eval/runner.py` 顶部有说明)。
- 执行提示两种变体:`production`(逐字沿用,含"尽可能充分展开",与提示词里的简洁要求冲突)和 `concise`(与提示词【简洁与完整要求】一致)。
- 检索器是 BM25(字二元组)+ 分块/规则两路 RRF,不是生产的稀疏+稠密;只保证被比较的模型看到同一检索结果。

## 目录
- `sr_eval/`:`prompt` 渲染 · `kb` 检索 · `llm` 客户端+账本+预算 · `policy` 外发闸门 · `runner` ReAct 循环 · `gates` 硬门/指标 · `judge` 严格评审+播种错误校准 · `cases` 合成案件 · `report` 汇总 · `cli`
- `config/`:`models.json`(模型与单价)、`budget.json`(预算)、`outbound.json`(外发闸门)
- `cases/<split>/`:合成案件(37 字段事故信息 + 生产指导意见提示生成的指导意见)
- `runs/<时间_标签>/`:`run.json`、`traces/`(完整请求/响应/思考)、`gates.json`、`judgements.json`、`summary.md`
- `ledger/api-calls.jsonl`:无密钥调用账本(预留/结算/unknown,所有 run 共用,预算按它重放)
- `tests/`:`pytest harness/tests`(假传输层,不联网)

## 用法(在 `<repo>`)
```powershell
$env:PYTHONPATH = "harness"
.venv\Scripts\python.exe -B -m pytest harness/tests
.venv\Scripts\python.exe -B -m sr_eval.cli check                      # 账本合计、OpenRouter key 状态、外发闸门
.venv\Scripts\python.exe -B -m sr_eval.cli gen-cases --model minimax --split dev --start 0 --n 6 --workers 3
.venv\Scripts\python.exe -B -m sr_eval.cli run --models hy4,dsflash --cases dev --n 6 --kb synthetic --tag bake1 --workers 2
.venv\Scripts\python.exe -B -m sr_eval.cli judge --run bake1 --judges luna,mimo --workers 2
.venv\Scripts\python.exe -B -m sr_eval.cli calibrate --run bake1 --judge luna --n 3   # 播种错误校准评审区分度
.venv\Scripts\python.exe -B -m sr_eval.cli summary --run bake1
```

`run` 另有 `--variant production|concise`、`--max-tokens`(单次输出上限,含思考,也是预算预留依据)、`--timeout`(秒,默认 1800,与生产一致)、`--effort`、`--resume <run 目录>`(跳过已有 trace 续跑)。`judge/calibrate` 默认 `--effort max`(仓库所有者 2026-10-03:luna 与 MiniMax 一律 max;luna 只返回总结式思考,低档位只会降低质量)。学生本地模型的模板档位用 `--effort xhigh`(`local` 提供者把它作为 `chat_template_kwargs.reasoning_effort` 传给 llama-server)。

## 费用与外发闸门(必读)
- OpenRouter 用专用 key:读取顺序 环境变量 `OPENROUTER_API_KEY` → `补充.txt` 的 `openrouter_key:` 行。密钥只在进程内使用,不入账本/日志/异常。
- **key 必须带 credit limit**(服务商侧硬上限)才会发送;没有上限的 key 直接拒绝(docs/api-authorization.md 要求可证明的费用上界)。软件侧再按每次请求的**最坏费用**(输入按 1 token/字符、输出按 `--max-tokens`)预留,超过 `config/budget.json` 的 `cap_usd` 就不发送;未结算与 unknown 请求按预留计入。
- 通道按 `config/network.json`:**OpenRouter 经环境代理(`HTTPS_PROXY`,CONNECT 隧道),MiniMax 直连**(仓库所有者 2026-10-02 指示,已记入 docs/api-authorization.md);代理缺失时发送前报错,不会悄悄直连。`gpt-6-luna` 直连会 403(地区限制),经代理可用。每次请求最多在 429/503 时重试 1 次;网络中断记 `unknown`,不自动重试。
- 外发闸门 `config/outbound.json`:合成案件可外发;公开法规/知识库片段外发已由仓库所有者 2026-10-02 批准(`public_statutes_external_send=true`);真实案件永不外发。若改回 false,只能用 `--kb synthetic`(12 条虚构条款,只验证机制)。
- `cli check` 会显示 key 的 credit limit 与账户余额;**key 无 limit 时评测台拒绝发送**(换专用 key,或由仓库所有者明确变更 docs/api-authorization.md 的边界)。
- 截断(`finish_reason=length`)记硬门失败,不拿截断冒充效率优化。

## 硬门(任一失败即不合格,不被评审总分覆盖)
`no_report` · `truncated_output` · `tool_markup_in_report` · `tool_call_unknown_tool/arguments_not_json` · `citation_not_visible`(`[依据: …]` 必须是可见片段 id)· `article_not_in_visible_text`(法条号/国标号必须出现在可见片段)· `fabricated_quantity`(速度、钟点必须来自输入;中文数字会折算)· `sentencing_or_crime_language` · `privacy_leak`(身份证/手机/邮箱)。
其余(未结论式定责、比例、未支持的数量、复述指导意见原句、元话语、英文、缺章节、重复率)为 warn,计数并交评审参考。

## 已知局限
- 检索器与生产不同(见上);`sanitize_markdown_output` 为简化复刻(未含表格/加粗修复)。
- 数量门只认阿拉伯数字与简单中文数字,推算出的数值会进 warn;语义层面(事实归属、因果)只能靠评审。
- JSON 回退协议(不支持原生工具的模型)是近似复刻,尚未对照生产 legacy 分支逐行核对。
- 模式 H(harness `/report-runs` 的 CandidateReport JSON + 独立审查角色)尚未实现。
