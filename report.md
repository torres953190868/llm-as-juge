# 基于 MT-Bench 的 LLM-as-a-Judge 长度偏差与位置偏差试点实验研究

作者：Hongyu Zhou  
学号：2575016  
Programme：MRes Computer Science  
项目：`llm-judge-bias`

## 摘要

本研究关注大语言模型作为自动裁判时是否会受到非内容因素影响。实验围绕两个问题展开：第一，在语义基本相同的情况下，更长的回答是否更容易被 LLM judge 选中；第二，同一组回答在 A/B 展示位置互换后，A 位置是否更容易获胜。实验使用 FastChat MT-Bench 问题与模型回答作为数据来源，构建了两条独立流程：长度偏差实验和位置偏差实验。

长度偏差实验先筛选适合进行回答加长的 MT-Bench 样本，再使用 DeepSeek 生成加长回答，并通过 manipulation check 过滤掉语义变化、引入新事实或质量提升的样本。最终保留 21 个问题，生成 84 条 A/B trial，并由 Gemini、DeepSeek 和小米 Mimo 三个 judge 模型进行评判，共得到 252 条 parsed judgment。结果显示，长回答总体胜率为 45.4%，net length preference 为 -7.9%，bootstrap 95% CI 为 [-31.0%, 16.7%]，未观察到稳定的总体长度偏好。然而，长回答放在 A 位置时胜率为 58.8%，放在 B 位置时胜率为 30.4%，显示出明显的位置敏感性。

位置偏差实验独立比较 `gpt-4` 与 `gpt-3.5-turbo` 的原始回答，并交换 A/B 位置。最终生成 152 条 position-bias trial，由三个 judge 模型评判后得到 456 条 parsed judgment。结果显示 judge 明显偏好 `gpt-4` 回答，`gpt-4` decisive win rate 为 80.5%；但 A 位置 decisive win rate 仅为 53.3%，二项检验 p = 0.1876，未形成强位置偏差证据。整体而言，当前实验更适合作为 pilot study：它说明 LLM judge 在质量差异较小时可能更容易受展示位置影响，但尚不足以支持普遍性的长度偏差或位置偏差结论。

## 1. 研究背景

LLM-as-a-Judge 是当前大语言模型评估中常见的方法。它让一个强模型阅读用户问题和候选回答，然后判断哪一个回答更好。与人工评估相比，这种方法成本更低、速度更快，也更容易批量运行。但是，如果 judge 模型本身受到回答位置、回答长度、格式或措辞等非内容因素影响，那么自动评估结果就可能产生系统性偏差。

本项目研究两个常见风险：

- **长度偏差**：在内容基本一致时，更长、更详细的回答是否更容易被 judge 选中。
- **位置偏差**：同一组回答中，放在 A 位置的回答是否更容易被 judge 选中。

这两个问题需要分开研究。长度实验中的 `long_A` 和 `long_B` 是为了控制长回答的位置，不应直接当作独立位置偏差结论。真正的位置偏差需要在两个原始回答之间交换 A/B 位置，并观察同一来源回答是否仍然获胜，或者同一展示位置是否反复获胜。

## 2. 数据与实验环境

实验数据来自 FastChat MT-Bench。项目中使用两个数据根目录：

- 问题文件：`FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl`
- 模型回答文件：`FastChat/data/mt_bench/model_answer/*.jsonl`

长度偏差实验使用 `gpt-4.jsonl` 中的原始回答作为待加长对象。位置偏差实验使用 `gpt-4.jsonl` 与 `gpt-3.5-turbo.jsonl` 中的原始回答进行 A/B 交换。

实验环境为 Windows PowerShell，Python 3.11 或更新版本。项目脚本主要使用 Python 标准库。API key 通过 `.env` 文件提供，避免写入代码或提交到版本库。

主要模型配置为：回答加长使用 `deepseek-v4-pro`；manipulation check 使用 `deepseek-v4-pro`；length-bias 和 position-bias judge 使用 `gemini-3-flash-preview`、`deepseek-v4-pro` 和 `mimo-v2-pro`。

## 3. 实验一：长度偏差实验

### 3.1 实验目的

长度偏差实验的目标是在尽量保持回答语义不变的前提下，仅改变回答长度，然后观察 judge 是否更倾向选择长回答。

核心设计是将同一条原始回答构造为两个版本：

- 原始回答：较短版本。
- Padded answer：通过模型加长后的版本。

随后将两者组成 A/B 对照，并交换位置：

- `long_A`：长回答放在 A，短回答放在 B。
- `long_B`：短回答放在 A，长回答放在 B。

如果长回答无论放 A 还是 B 都更容易胜出，才更接近长度偏差证据。如果长回答只在 A 位置明显胜出，则更可能说明位置与长度之间存在交互。

### 3.2 样本筛选

首先运行筛选脚本：

```powershell
python 01_screen_length_bias_eligibility.py --dry-run
python 01_screen_length_bias_eligibility.py
```

筛选规则较保守。脚本自动排除 coding、extraction、math、reasoning 等任务，因为这些任务通常对格式、推理链或数值精度敏感，加长回答可能改变答案质量。脚本还排除回答过短、包含代码块、严格输出格式、严格长度限制等样本。

关键阈值：

| 条件 | 阈值 |
|---|---:|
| 回答总词数 | 至少 100 words |
| 每个回答 turn | 至少 60 words |

筛选结果如下：

| 阶段 | 数量 |
|---|---:|
| MT-Bench screened rows | 80 |
| eligible rows | 28 |
| excluded rows | 46 |
| manual review rows | 6 |

eligible 样本主要来自 writing、roleplay、STEM 和 humanities 类别。

### 3.3 生成加长回答

对 eligible 样本调用 DeepSeek 生成 padded answer：

```powershell
python 02_pad_answers_deepseek.py --input length_bias_eligible_samples.jsonl --input-format jsonl --dry-run
python 02_pad_answers_deepseek.py --input length_bias_eligible_samples.jsonl --input-format jsonl
```

加长配置如下：

| 参数 | 值 |
|---|---:|
| padding model | `deepseek-v4-pro` |
| minimum length ratio | 1.3 |
| maximum length ratio | 2.0 |
| maximum attempts | 3 |
| max tokens | 4096 |

本次运行中，28 条 eligible 样本全部生成 padded answer，`failed_deepseek_padding.jsonl` 行数为 0。通过 manipulation check 后保留的 21 条样本中，加长比例统计如下：

| 指标 | 数值 |
|---|---:|
| min length ratio | 1.36 |
| max length ratio | 1.81 |
| mean length ratio | 1.52 |
| median length ratio | 1.50 |

### 3.4 Manipulation Check

加长回答不能直接进入 length-bias 实验，因为模型在加长时可能引入新事实、改善结构或提升回答质量。如果 padded answer 不只是变长，而是变得更好，那么后续 judge 选择长回答就不能归因于长度。

因此，实验生成 manipulation-check 任务：

```powershell
python 03_prepare_manipulation_check_trials.py --dry-run --limit 2
python 03_prepare_manipulation_check_trials.py
```

随后调用 judge 执行检查：

```powershell
python 04_run_manipulation_check_judge.py --dry-run --limit 1
python 04_run_manipulation_check_judge.py
```

检查字段为 `semantic_equivalence`、`new_facts`、`structure_improvement` 和 `quality_improvement`。严格通过要求是语义等价为 true，且无新增事实、无结构改进、无质量改进。

严格过滤命令：

```powershell
python 05_filter_manipulation_check_results.py --dry-run
python 05_filter_manipulation_check_results.py
```

manipulation check 结果：

| 结果 | 数量 |
|---|---:|
| padded rows checked | 28 |
| passed strict check | 21 |
| excluded after check | 7 |

7 条被排除的原因包括语义不完全等价、新增事实、结构改善或质量改善。过滤后保留样本的类别分布如下：

| 类别 | 数量 |
|---|---:|
| writing | 5 |
| roleplay | 4 |
| stem | 5 |
| humanities | 7 |

### 3.5 构造 Length-Bias A/B Trials

使用通过 manipulation check 的 21 条样本构造 A/B trial：

```powershell
python 06_prepare_length_bias_trials.py --input mt_bench_questions_answers_padded_deepseek_checked.jsonl --dry-run
python 06_prepare_length_bias_trials.py --input mt_bench_questions_answers_padded_deepseek_checked.jsonl
```

每条样本生成四种 trial：

| 因素 | 条件 |
|---|---|
| 长回答位置 | `long_A`, `long_B` |
| judge prompt | `standard_anti_length`, `neutral_no_length` |

因此 trial 数量为：

```text
21 questions x 2 prompts x 2 positions = 84 trials
```

其中 `standard_anti_length` 会明确提醒 judge 不要受回答长度影响；`neutral_no_length` 只提醒 judge 避免位置偏差，不额外强调长度。

### 3.6 Judge 运行

运行 judge：

```powershell
python 07_run_length_bias_judge.py --dry-run --limit 1
python 07_run_length_bias_judge.py
```

使用三个 judge 模型，每个 trial 被三个模型分别评判：

```text
84 trials x 3 judges = 252 parsed judgments
```

随后分析：

```powershell
python 08_analyze_length_bias_results.py --dry-run
python 08_analyze_length_bias_results.py
```

输出文件包括 `length_bias_summary.json`、`length_bias_summary.txt` 和 `length_bias_summary.svg`。

### 3.7 Length-Bias 实验结果

总体结果：

| 指标 | 数值 |
|---|---:|
| total judgments | 252 |
| long wins | 98 |
| short wins | 118 |
| ties | 36 |
| invalid | 0 |
| long decisive win rate | 45.4% |
| tie rate | 14.3% |
| net length preference | -7.9% |
| bootstrap 95% CI | [-31.0%, 16.7%] |

结果显示，长回答总体没有明显优势。长回答胜率低于短回答，net length preference 为 -7.9%，但 bootstrap 置信区间跨过 0，因此当前 pilot 不能支持“LLM judge 稳定偏好长回答”的结论。

按 judge 分组：

| Judge | Total | Long | Short | Tie | Long win rate | Net |
|---|---:|---:|---:|---:|---:|---:|
| `deepseek-v4-pro` | 84 | 33 | 30 | 21 | 52.4% | +3.6% |
| `gemini-3-flash-preview` | 84 | 25 | 58 | 1 | 30.1% | -39.3% |
| `mimo-v2-pro` | 84 | 40 | 30 | 14 | 57.1% | +11.9% |

不同 judge 的方向并不一致。Gemini 明显偏向短回答，而 DeepSeek 和 Mimo 略偏向长回答。这说明 judge 模型之间存在较大差异，不能只用单一 judge 得出泛化结论。

按长回答位置分组：

| 条件 | Total | Long wins | Short wins | Ties | Long win rate | Net |
|---|---:|---:|---:|---:|---:|---:|
| `long_A` | 126 | 67 | 47 | 12 | 58.8% | +15.9% |
| `long_B` | 126 | 31 | 71 | 24 | 30.4% | -31.7% |

成对 swapped-position 统计：

| 指标 | 数值 |
|---|---:|
| complete pairs | 126 |
| paired mean length preference | -7.9% |
| paired 95% CI | [-31.0%, 16.7%] |
| position delta A minus B | 47.6% |
| position delta 95% CI | [27.8%, 66.7%] |

这一结果很关键。虽然总体没有稳定长度偏好，但长回答放在 A 时明显更容易获胜，放在 B 时明显更不容易获胜。因此，实验一更准确的发现不是“长回答一定更受偏好”，而是“长度操控下出现明显位置敏感性”。这也说明如果不交换 A/B 位置，长度偏差实验很容易被位置因素污染。

## 4. 实验二：位置偏差实验

### 4.1 实验目的

位置偏差实验独立于长度偏差实验。它不使用 padded answer，而是直接比较两个来源模型的原始回答，并交换 A/B 展示位置。如果 judge 偏好某个展示位置，那么同一个位置在交换后仍会反复获胜；如果 judge 主要看回答质量，则同一来源模型应该在交换位置后仍然获胜。

本实验默认比较：

| 角色 | 模型 |
|---|---|
| source model A | `gpt-4` |
| source model B | `gpt-3.5-turbo` |

### 4.2 构造 Position-Bias Trials

推荐入口：

```powershell
python run_position_bias_experiment.py --dry-run --question-limit 2
python run_position_bias_experiment.py
```

底层流程为：

```powershell
python 09_prepare_position_bias_trials.py
python 10_run_position_bias_judge.py
python 11_analyze_position_bias_results.py
```

默认排除 question IDs `105`, `107`, `128`, `136`，因为这些问题在 pilot 中反复导致部分 judge 返回空内容。最终覆盖 76 个问题：

| 类别 | 问题数 |
|---|---:|
| writing | 10 |
| roleplay | 10 |
| reasoning | 8 |
| math | 10 |
| coding | 9 |
| extraction | 9 |
| stem | 10 |
| humanities | 10 |

每个问题生成两个 swapped trials：

- `model_a_A`：`gpt-4` 回答放 A。
- `model_a_B`：`gpt-4` 回答放 B。

因此：

```text
76 questions x 2 positions = 152 trials
152 trials x 3 judges = 456 parsed judgments
```

### 4.3 Position-Bias 实验结果

总体结果：

| 指标 | 数值 |
|---|---:|
| total judgments | 456 |
| `gpt-4` wins | 338 |
| `gpt-3.5-turbo` wins | 82 |
| position A wins | 224 |
| position B wins | 196 |
| ties | 36 |
| invalid | 0 |
| `gpt-4` decisive win rate | 80.5% |
| position A decisive win rate | 53.3% |
| position A vs B binomial p | 0.1876 |

按 judge 分组：

| Judge | Total | `gpt-4` win rate | Position A win rate | p-value |
|---|---:|---:|---:|---:|
| `deepseek-v4-pro` | 152 | 75.9% | 52.6% | 0.6084 |
| `gemini-3-flash-preview` | 152 | 81.2% | 54.9% | 0.2786 |
| `mimo-v2-pro` | 152 | 84.2% | 52.5% | 0.6110 |

位置偏差的核心结果是：A 位置胜率为 53.3%，高于 50%，但幅度较小，且 p = 0.1876，没有达到常见显著性标准。因此，本实验不能强烈支持 judge 存在独立位置偏差。

相反，source model preference 非常明显。`gpt-4` decisive win rate 为 80.5%，说明三个 judge 大多能够识别并偏好 GPT-4 来源回答，即使回答位置被交换。

成对 swapped-pair 结果：

| 类型 | 数量 |
|---|---:|
| total pairs | 228 |
| source_model_a_both | 157 |
| source_model_b_both | 28 |
| position_A_both | 18 |
| position_B_both | 5 |
| tie_both | 16 |
| mixed | 4 |
| position_A_consistent_rate | 7.9% |

`source_model_a_both` 表示交换位置后仍然是 `gpt-4` 来源回答获胜，这比 `position_A_both` 多得多。因此，实验二的主要信号是来源模型质量差异，而不是展示位置偏差。

## 5. 讨论

### 5.1 为什么实验一出现明显位置敏感性，而实验二没有强位置偏差

实验一和实验二的比较对象不同，因此位置效应的表现也不同。

实验一比较的是同一个原始回答和其 padded version。经过 manipulation check 后，两者语义和质量被尽量控制为接近。在这种难以判断优劣的场景中，judge 更容易受到展示位置、回答长度或 prompt 细节等非内容因素影响。因此，长回答放 A 时胜率明显上升，放 B 时胜率明显下降。

实验二比较的是 `gpt-4` 与 `gpt-3.5-turbo` 的原始回答。两者通常存在更明显的质量差异，judge 更容易根据内容质量选择 `gpt-4` 回答。强质量信号会压过较弱的位置信号，所以 A 位置只显示出轻微优势，并没有形成显著位置偏差。

因此，实验一中的位置效应应解释为“长度操控下的位置敏感性”，不是独立位置偏差结论。实验二才是专门的位置偏差实验，而它当前没有显示强证据。

### 5.2 对 Length Bias 的解释

当前 length-bias 总体结果没有支持稳定的长回答偏好。三个 judge 的方向不一致：Gemini 明显偏短，DeepSeek 和 Mimo 略偏长。这可能说明不同 judge 的评价偏好不同，也可能说明当前样本量不足以稳定估计总体效应。

不过，long_A 与 long_B 的差异非常大，说明在长度实验中必须做位置交换。如果只把长回答固定放在 A，可能会错误地得出“长回答更好”的结论；如果只把长回答固定放在 B，可能会得出相反结论。

### 5.3 对 Position Bias 的解释

独立 position-bias 实验中，A 位置胜率为 53.3%，方向上略高于 B，但统计证据较弱。更强的现象是 judge 对 `gpt-4` 来源回答的偏好。这说明当候选回答质量差异明显时，judge 的内容判断可以压过位置影响。

## 6. 局限性

本研究是 pilot study，仍有以下限制：

1. Length-bias 样本量较小。筛选和 manipulation check 后只剩 21 个问题，不能代表完整 MT-Bench。
2. Length-bias 类别覆盖有限，只包含 writing、roleplay、STEM 和 humanities。
3. Padding 操作由模型生成，虽然经过 manipulation check，但仍可能存在微小语义或风格变化。
4. Manipulation check 本身也由 LLM judge 完成，可能继承 judge 模型的不稳定性。
5. 不同 judge 模型结果差异较大，尤其 Gemini 与 DeepSeek/Mimo 在 length-bias 方向上不一致。
6. Position-bias 实验中 `gpt-4` 和 `gpt-3.5-turbo` 的质量差异较大，这有利于观察 source-model preference，但可能掩盖较弱的位置偏差。
7. 本实验主要使用 word count 衡量长度，没有使用 tokenizer-level token count。
8. 当前结论依赖本次 API 模型版本和运行时间，未来模型更新可能改变结果。

## 7. 可复现性

项目提供了完整脚本和中间结果文件。`Readme.md` 记录推荐运行流程，`REPRODUCIBILITY.md` 记录可复现性边界。关键实验产物包括 `length_bias_screening_summary.json`、`parsed_manipulation_check_judgments.jsonl`、`length_bias_trials.jsonl`、`parsed_length_bias_judgments.jsonl`、`length_bias_summary.txt`、`position_bias_trials.jsonl`、`parsed_position_bias_judgments.jsonl` 和 `position_bias_summary.txt`。

每次修改脚本后，可以先运行语法检查、单元测试和各阶段 dry-run。dry-run 不会调用付费 API；真正运行 padding 或 judge 阶段时会产生 API 调用成本。

## 8. 结论

本研究构建并运行了两个 LLM-as-a-Judge 偏差实验。长度偏差实验没有发现稳定的总体长回答偏好：长回答 decisive win rate 为 45.4%，net length preference 为 -7.9%，且置信区间跨过 0。然而，在长度实验中发现明显的位置敏感性：长回答放 A 时胜率为 58.8%，放 B 时胜率为 30.4%，position delta 为 47.6%。这说明长度偏差实验必须显式控制 A/B 位置，否则可能得出误导性结论。

独立位置偏差实验中，judge 明显偏好 `gpt-4` 来源回答，`gpt-4` decisive win rate 为 80.5%。A 位置 decisive win rate 为 53.3%，p = 0.1876，未显示强独立位置偏差。换言之，当回答质量差异明显时，judge 更主要根据内容质量而不是展示位置作出选择。

总体而言，本实验可以作为研一期末作业中的试点实证研究。它的主要贡献不是给出最终普遍结论，而是展示了一个可复现的实验框架，并说明 LLM-as-a-Judge 偏差研究需要将长度、位置和来源模型质量分开控制。
