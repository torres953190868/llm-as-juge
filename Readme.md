# llm-as-judge bias 实验

这个项目是用来研究一个问题：

> 让大模型当裁判时，它会不会偏心？

这里主要看两种偏心：

- **位置偏差**：同样两个回答，放在 A 位置会不会更容易赢？
- **长度偏差**：内容差不多时，更长的回答会不会更容易被裁判选中？

目前用的数据主要来自 FastChat 的 MT-Bench。

## 这个项目里有什么

根目录下主要是我自己写的实验脚本：

- `00_export_mt_bench_pairs.py`：早期导出 MT-Bench 问题和回答的辅助脚本。
- `01_screen_length_bias_eligibility.py`：先筛题，排除不适合做“加长回答”的题。
- `02_pad_answers_deepseek.py`：用 DeepSeek 把原回答加长，但尽量不改变意思。
- `03_prepare_length_bias_trials.py`：把原回答和加长回答组成 A/B 对照实验。
- `04_run_length_bias_judge.py`：调用 judge 模型，让它在 A/B 之间做选择。
- `05_analyze_length_bias_results.py`：统计结果，看长回答、短回答、A 位置、B 位置各赢了多少，也会对 `long_A` / `long_B` 做成对分析。
- `06_prepare_position_bias_trials.py`：从两个原始 MT-Bench 模型回答生成纯位置偏差 swapped A/B trials，默认 `gpt-4` vs `gpt-3.5-turbo`。
- `07_prepare_manipulation_check_trials.py`：为 padded answer 生成语义等价、新事实、结构改善、质量改善检查任务。
- `08_analyze_position_bias_results.py`：分析 position-bias parsed judgments，分开报告来源模型偏好和 A/B 位置偏好。
- `99_run_length_bias_experiment.py`：目前只串起 `prepare -> judge -> analyze`，还没有把 `screen` 和 `pad` 串进去。

简单说，长度偏差主流程是：

```text
筛题 -> 加长回答 -> 做 A/B 对照 -> 让模型当裁判 -> 统计偏差
```

位置偏差是另一条独立流程：

```text
选两组原始模型回答 -> 互换 A/B 位置 -> 让模型当裁判 -> 统计来源模型偏好和 A/B 位置偏好
```

不要把 length-bias 里的 `long_A` / `long_B` 控制直接当成最终 position-bias 结论；它主要是长度实验里的位置控制。

## 需要自己准备的数据

这个 GitHub 仓库不会上传 FastChat 和实验生成的数据文件。

你需要在项目根目录放一个 `FastChat/` 文件夹，并确保下面这些路径存在：

```text
FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
FastChat/data/mt_bench/model_answer/gpt-4.jsonl
```

注意：这个项目里有两个 MT-Bench 数据根目录，不要混用：

- 题目在 `FastChat/fastchat/llm_judge/data/mt_bench/`
- 下载的模型回答在 `FastChat/data/mt_bench/`

## 环境准备

建议用 Python 3.11 或更新版本。

在 PowerShell 里可以这样做：

```powershell
python -m venv llm-judge-env
.\llm-judge-env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

目前根目录脚本只用 Python 标准库，所以 `requirements.txt` 里暂时没有第三方依赖。

## API key 怎么放

不要把真实 API key 写进代码，也不要上传到 GitHub。

先复制一份环境变量模板：

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 里填自己的 key，例如：

```text
DEEPSEEK_API_KEY=你的key
GEMINI_API_KEY=你的key
XIAOMI_API_KEY=你的key
OPENAI_API_KEY=你的key
```

`.env` 已经被 `.gitignore` 忽略，不会被上传。

## 推荐运行顺序

下面是 length-bias pilot 的完整手动流程。`99_run_length_bias_experiment.py` 目前只串起 `03_prepare -> 04_judge -> 05_analyze`，不包含前面的筛题和加长步骤。

### 1. 先筛题

先 dry-run 看看会筛出多少题：

```powershell
python 01_screen_length_bias_eligibility.py --dry-run
```

确认没问题后，真正写出筛选结果：

```powershell
python 01_screen_length_bias_eligibility.py
```

这一步会生成：

- `length_bias_screened_samples.jsonl`
- `length_bias_eligible_samples.jsonl`
- `length_bias_screening_excluded_samples.jsonl`
- `length_bias_screening_summary.json`

### 2. 给回答加长

建议显式使用筛出来的 eligible 样本：

```powershell
python 02_pad_answers_deepseek.py --input length_bias_eligible_samples.jsonl --input-format jsonl --dry-run
```

确认样本能正常读取后，再真正调用 API：

```powershell
python 02_pad_answers_deepseek.py --input length_bias_eligible_samples.jsonl --input-format jsonl
```

这一步会调用 DeepSeek API，可能花钱。

### 3. 生成 A/B 实验题

先检查：

```powershell
python 03_prepare_length_bias_trials.py --dry-run
```

再真正生成：

```powershell
python 03_prepare_length_bias_trials.py
```

这一步会生成 `length_bias_trials.jsonl`。

每个样本会生成几种对照：

- 长回答放 A，短回答放 B
- 短回答放 A，长回答放 B
- 使用不同 judge prompt 条件

这样后面才能分开看“长度偏差”和“位置偏差”。

### 4. 让模型当裁判

先 dry-run 看第一条 trial：

```powershell
python 04_run_length_bias_judge.py --dry-run --limit 1 --deepseek 1 --gemini 0 --xiaomi 0
```

真正运行时要明确指定你想用哪些 judge。比如只跑 DeepSeek：

```powershell
python 04_run_length_bias_judge.py --deepseek 1 --gemini 0 --xiaomi 0
```

或者用配置文件：

```powershell
python 04_run_length_bias_judge.py --judge-config judge_config.example.json
```

这一步会真实调用 API，可能花钱。

### 5. 分析结果

先 dry-run：

```powershell
python 05_analyze_length_bias_results.py --dry-run
```

再写出结果：

```powershell
python 05_analyze_length_bias_results.py
```

会生成：

- `length_bias_summary.json`
- `length_bias_summary.txt`
- `length_bias_summary.svg`

### 6. 做 manipulation check

长度偏差实验的关键风险是：padded answer 可能不只是变长，还可能变得更清楚、更完整或新增事实。正式写结论前，建议先生成 manipulation-check 任务：

```powershell
python 07_prepare_manipulation_check_trials.py --dry-run --limit 2
python 07_prepare_manipulation_check_trials.py
```

这一步会生成 `manipulation_check_trials.jsonl`，用于检查：

- padded answer 是否保持语义等价
- 是否新增事实或例子
- 是否改善结构
- 是否改善整体质量

只有通过检查的 padded answer 才适合进入更强的最终 length-bias 结论。

### 7. 单独跑 position-bias 实验

position-bias 不使用 padded answer，而是比较两组原始模型回答的 A/B 位置互换。默认是 `gpt-4` vs `gpt-3.5-turbo`：

```powershell
python 06_prepare_position_bias_trials.py --dry-run --limit 2
python 06_prepare_position_bias_trials.py
```

然后复用 judge runner，但显式指定 position-bias 的输入和输出文件：

```powershell
python 04_run_length_bias_judge.py --trials position_bias_trials.jsonl --raw-output raw_position_bias_judgments.jsonl --parsed-output parsed_position_bias_judgments.jsonl --deepseek 1 --gemini 0 --xiaomi 0
```

最后分析 position-bias 结果：

```powershell
python 08_analyze_position_bias_results.py --dry-run
python 08_analyze_position_bias_results.py
```

会生成：

- `position_bias_summary.json`
- `position_bias_summary.txt`

### 8. 使用 99 入口串起后半段

如果已经有 `mt_bench_questions_answers_padded_deepseek.jsonl`，可以用 `99_run_length_bias_experiment.py` 串起 trial 准备、judge 和分析：

```powershell
python 99_run_length_bias_experiment.py --dry-run --limit 1 --deepseek 1 --gemini 0 --xiaomi 0
```

真正运行时去掉 `--dry-run`。注意：`--judge-model` 只影响 DeepSeek judge 的模型名；是否启用 DeepSeek/Gemini/Xiaomi 仍然由 `--deepseek`、`--gemini`、`--xiaomi` 控制。

## 每次改代码后怎么快速检查

最小检查可以这样跑：

```powershell
python -m py_compile "length_bias_common.py" "length_bias_samples.py" "99_run_length_bias_experiment.py"
python -m unittest test_bias_framework.py
python 01_screen_length_bias_eligibility.py --dry-run
python 03_prepare_length_bias_trials.py --dry-run
python 06_prepare_position_bias_trials.py --dry-run --limit 2
python 07_prepare_manipulation_check_trials.py --dry-run --limit 2
python 05_analyze_length_bias_results.py --dry-run
```

如果只是改 README，不需要跑完整实验。

## 哪些文件不会上传

下面这些是本地文件，不应该进 GitHub：

- `.env`
- `llm-judge-env/`
- `FastChat/`
- `__pycache__/`
- `raw_*.jsonl`
- `tmp_*.jsonl`
- `length_bias_trials.jsonl`
- `parsed_length_bias_judgments.jsonl`
- `length_bias_summary.*`
- `position_bias_trials.jsonl`
- `parsed_position_bias_judgments.jsonl`
- `position_bias_summary.*`
- `manipulation_check_trials.jsonl`
- 其他实验生成的 JSONL/TXT/SVG 结果

原因很简单：

- `.env` 里有密钥。
- `llm-judge-env/` 是本机虚拟环境。
- `FastChat/` 是外部数据和源码。
- raw/parsed 结果文件可能很大，而且应该按实验版本重新生成。

## 现在这个项目还需要改进什么

当前项目已经能跑 pilot，但还不是最终版。

后面最重要的改进是：

- 把 `01_screen -> 02_pad -> 03_prepare -> 04_judge -> 05_analyze` 全部串成一个统一入口。
- 扩展 manipulation check 的运行和分析闭环，而不只是生成检查任务。
- 给 position-bias 实验补齐真实 judge 结果和结果文件。
- 把最终实验配置和结果版本记录得更清楚。

## Current pilot limitations

The current length-bias results are pilot evidence, not final claims.

- Current attrition is `80 -> 28 -> 17`: 80 screened MT-Bench rows, 28 eligible rows, and 17 questions with parsed judge results.
- The current parsed result shape is `204 rows = 17 questions x 2 prompts x 2 positions x 3 judges`.
- Category coverage is limited after screening and padding. Current analyzed questions cover only a subset of MT-Bench categories, mainly humanities, roleplay, and STEM.
- Position bias should be treated as a separate experiment from length bias. The length-bias pipeline swaps `long_A` / `long_B` for control, but final position-bias claims need their own dedicated design and analysis.
- A stronger final length-bias run needs a manipulation check showing that padded answers preserve meaning while changing length enough to test the intended mechanism.
- Dry-run commands do not call paid APIs. Paid API usage starts when padding or judging scripts are run without `--dry-run`.
