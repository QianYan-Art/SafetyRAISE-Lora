# 从这里开始训练

本目录是首版完整数据交付入口，不是继续制作的空脚手架。
先做 SFT，再用同版 RL 任务进行在线采样；根据实际模型错误迭代数据。
训练策略见同目录 `training-strategy.md`，限制见 `DATA-CARD.md`。

## 文件

| 文件 | 使用方式 |
|---|---|
| `sft-training.zip` | 现有 1,698 条 SFT，冻结 v22 字节；先用这一包 |
| `rl-prompts.zip` | 534 个完整报告任务，只含模型可见输入及血缘 |
| `rl-reward-references.zip` | 534 个参考目标和奖励合同，只由独立奖励进程读取 |
| `heldout-evaluation.zip` | 开发、测试各 50 组，含参考答案，不用于训练 |
| `manifest.json` | 来源、文件与 ZIP 成员哈希及真实能力边界 |

## SFT 选择

解包 `sft-training.zip` 后，仅选
`releases/local-synthetic-multitask-sft-20260930-v30/qwen-tokenized/train.jsonl`。
每行用 `input_ids`、`attention_mask`、`labels`；保留 `record_id` 与 `case_group_id`
用于追踪。沿用已固定基座与文本资产时，不再套模板、分词或追加 EOS，
也不要预先手动移动 labels；训练器的 causal shift 需在第一批实际验证。
pad 的 label 应为 `-100`，attention 为 0。

换 tokenizer 或模板时，改用同目录 `messages/train.jsonl` 重新导出，
尊重 `supervised_message_indexes`，不把 system/user/tool 观察作为目标。
这两种视图不能混在一次训练中，也不能与历史版本叠加。
原始训练清单的模型资产引用与血缘仍在 SFT 档案内；固定文本资产与哈希见训练策略。

## 后训练接入

`rl-prompts.zip` 中 `rl/prompts.jsonl` 每行是一项通用任务。
actor 只渲染 `messages`，`task_id` / `case_group_id` / `reward_reference_id`
用于路由和审计。`rl/lineage.jsonl` 只用于核对来源，不作为额外 prompt。

奖励进程按 `task_id` 查 `reward/references.jsonl`，并检查同案 ID、
`input_sha256` 与 `reference_sha256`。
`reward/reward-contract.json` 是硬门与评分合同，不是已经实现的奖励函数。
适配训练器后先在训练侧小样本验证 rollout、奖励区分度与更新，再扩大；
不能只因 534 条任务已导出就跳过奖励校准。

哈希编码固定如下，不能把两种输入哈希混用：

- `input_sha256`：对原 system/user 两条 messages 执行
  `json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"`，
  以 UTF-8 编码后计算 SHA-256；这里的 LF 换行属于哈希字节。
- `rl/lineage.jsonl` 的 `source_visible_input_sha256`：仅 user content 的 UTF-8 字节哈希，
  与源 SFT lineage 的 `visible_input_sha256` 交叉核对。
- `reference_sha256`：原 assistant 正文的 UTF-8 字节哈希，不做空白归一化。

优先实验路线为 SFT 加响应蒸馏，然后比较 GSPO 与保留的 SFT；
DPO 在实际同输入候选成对后作离线对照。PPO、token 级 OPD、特征蒸馏与逐层结构改造
不是首轮必选项。

## 首次训练前的短检查

1. 对照 manifest 核验哈希，只加载本版，不加载候选和隔离目录。
2. 使用 16–32 条训练侧样本核验 collator、mask、EOS、梯度与保存/重载。
3. 开发集只用于选配置，测试集在冻结候选后验收；两者不做校准或蒸馏。
4. 首轮保存 0.5 与 1 epoch 检查点，按严重错误和分项能力选择。
5. 没有法规正文时不让模型编法条；未执行工具时不宣称已有工具轨迹。

## 工作区只读复核

在原工作目录 `C:\tmp\internal\dataset` 执行：

```powershell
$env:PYTHONPATH = 'src'
& .venv\Scripts\python.exe -B -m safetyraise_dataset.jobs.first_training_release --workspace . --verify
```

此命令复核数据包，不训练模型、不访问 API。更改数据、奖励合同或策略时发布新版本，
不要覆盖这个冻结首版。
