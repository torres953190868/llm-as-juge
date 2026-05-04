# llm-as-judge bias 实验

这个项目用来研究一个问题：

> 让大模型当裁判时，它会不会偏心？

目前主要看两种偏心：

- **位置偏差**：同样两个回答，放在 A 位置会不会更容易赢？
- **长度偏差**：内容差不多时，更长的回答会不会更容易被裁判选中？

当前数据主要来自 FastChat 的 MT-Bench。

## 这个项目里有什么

根目录下主要是实验脚本：

- `00_export_mt_bench_pairs.py`：早期导出 MT-Bench 问题和回答的辅助脚本。
- `01_screen_length_bias_eligibility.py`：筛出适合做“加长回答”的样本。
- `02_pad_answers_deepseek.py`：用 DeepSeek 把原回答加长，并检查长度比例。
- `03_prepare_manipulation_check_trials.py`：为 padded answer 生成 manipulation-check 任务。
- `04_run_manipulation_check_judge.py`：调用 judge 模型检查 padded answer 是否仍适合作为长度操控。
- `05_filter_manipulation_check_results.py`：只保留通过严格 manipulation check 的 padded rows。
- `06_prepare_length_bias_trials.py`：把原回答和通过检查的加长回答组成 A/B 对照实验。
- `07_run_length_bias_judge.py`：调用 judge 模型，让它在 A/B 之间做选择。
- `08_analyze_length_bias_results.py`：统计 length-bias 结果，包括长/短回答胜率、A/B 位置和成对分析。
- `09_prepare_position_bias_trials.py`：从两个原始 MT-Bench 模型回答生成独立 position-bias swapped A/B trials。
- `10_analyze_position_bias_results.py`：分析 position-bias parsed judgments。
- `11_run_position_bias_experiment.py`：串起独立 position-bias 的 prepare、judge、analyze 流程。
- `99_run_length_bias_experiment.py`：串起 length-bias 后半段或 checked-all 严谨流程。

正式 length-bias 流程是：

```text
筛题 -> 加长回答 -> manipulation check -> 过滤样本 -> 做 A/B 对照 -> 让模型当裁判 -> 统计偏差
```

position-bias 是另一条独立流程：

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
FastChat/data/mt_bench/model_answer/gpt-3.5-turbo.jsonl
```

注意：这个项目里有两个 MT-Bench 数据根目录，不要混用：

- 题目在 `FastChat/fastchat/llm_judge/data/mt_bench/`
- 下载的模型回答在 `FastChat/data/mt_bench/`

## 环境准备

建议用 Python 3.11 或更新版本。

```powershell
python -m venv llm-judge-env
.\llm-judge-env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

目前根目录脚本只用 Python 标准库，所以 `requirements.txt` 里暂时没有第三方依赖。

## API key 怎么放

不要把真实 API key 写进代码，也不要上传到 GitHub。

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 里填自己的 key：

```text
DEEPSEEK_API_KEY=你的key
GEMINI_API_KEY=你的key
XIAOMI_API_KEY=你的key
OPENAI_API_KEY=你的key
```

`.env` 已经被 `.gitignore` 忽略，不会被上传。

## 推荐运行顺序

下面是 length-bias 的严谨手动流程。`99_run_length_bias_experiment.py --stage checked-all` 可以串起 `03 -> 04 -> 05 -> 06 -> 07 -> 08`，但 `01` 和 `02` 仍然需要先单独运行。

### 1. 先筛题

```powershell
python 01_screen_length_bias_eligibility.py --dry-run
python 01_screen_length_bias_eligibility.py
```

这一步会生成：

- `length_bias_screened_samples.jsonl`
- `length_bias_eligible_samples.jsonl`
- `length_bias_screening_excluded_samples.jsonl`
- `length_bias_screening_summary.json`

### 2. 给回答加长

先 dry-run，确认 eligible 样本能正常读取：

```powershell
python 02_pad_answers_deepseek.py --input length_bias_eligible_samples.jsonl --input-format jsonl --dry-run
```

再真正调用 API：

```powershell
python 02_pad_answers_deepseek.py --input length_bias_eligible_samples.jsonl --input-format jsonl
```

这一步会生成：

- `mt_bench_questions_answers_padded_deepseek.jsonl`
- `mt_bench_questions_answers_padded_deepseek.txt`
- `raw_deepseek_padding_responses.jsonl`
- `failed_deepseek_padding.jsonl`

这一步会调用 DeepSeek API，可能花钱。

### 3. 生成 manipulation-check 任务

```powershell
python 03_prepare_manipulation_check_trials.py --dry-run --limit 2
python 03_prepare_manipulation_check_trials.py
```

这一步会生成 `manipulation_check_trials.jsonl`。

### 4. 跑 manipulation-check judge

先 dry-run：

```powershell
python 04_run_manipulation_check_judge.py --dry-run --limit 1 --deepseek 1 --gemini 0 --xiaomi 0
```

真正运行时明确指定 judge：

```powershell
python 04_run_manipulation_check_judge.py --deepseek 1 --gemini 0 --xiaomi 0
```

这一步会生成：

- `raw_manipulation_check_judgments.jsonl`
- `parsed_manipulation_check_judgments.jsonl`

这一步会真实调用 API，可能花钱。

### 5. 过滤通过 manipulation check 的样本

```powershell
python 05_filter_manipulation_check_results.py --dry-run
python 05_filter_manipulation_check_results.py
```

严格通过标准是：

```text
semantic_equivalence = true
new_facts = false
structure_improvement = false
quality_improvement = false
```

如果有多个 manipulation-check judge，必须所有 judge 都通过。输出：

- `mt_bench_questions_answers_padded_deepseek_checked.jsonl`
- `manipulation_check_excluded_samples.jsonl`

### 6. 生成 length-bias A/B 实验题

使用 checked padded rows：

```powershell
python 06_prepare_length_bias_trials.py --input mt_bench_questions_answers_padded_deepseek_checked.jsonl --dry-run
python 06_prepare_length_bias_trials.py --input mt_bench_questions_answers_padded_deepseek_checked.jsonl
```

这一步会生成：

- `length_bias_trials.jsonl`
- `excluded_length_bias_samples.jsonl`

每个样本会生成几种对照：

- 长回答放 A，短回答放 B
- 短回答放 A，长回答放 B
- 使用不同 judge prompt 条件

### 7. 让模型当裁判

先 dry-run 看第一条 trial：

```powershell
python 07_run_length_bias_judge.py --dry-run --limit 1 --deepseek 1 --gemini 0 --xiaomi 0
```

真正运行时明确指定 judge：

```powershell
python 07_run_length_bias_judge.py --deepseek 1 --gemini 0 --xiaomi 0
```

或者用配置文件。注意：`--judge-config` 会替代内置 judge flags，请复制并修改自己的配置后再跑：

```powershell
python 07_run_length_bias_judge.py --judge-config judge_config.example.json
```

这一步会生成：

- `raw_length_bias_judgments.jsonl`
- `parsed_length_bias_judgments.jsonl`

这一步会真实调用 API，可能花钱。

### 8. 分析 length-bias 结果

如果只跑 DeepSeek，分析时也带同样的 judge flags：

```powershell
python 08_analyze_length_bias_results.py --dry-run --deepseek 1 --gemini 0 --xiaomi 0
python 08_analyze_length_bias_results.py --deepseek 1 --gemini 0 --xiaomi 0
```

会生成：

- `length_bias_summary.json`
- `length_bias_summary.txt`
- `length_bias_summary.svg`

### 9. 单独跑 position-bias 实验

position-bias 不使用 padded answer，而是比较两组原始模型回答的 A/B 位置互换。默认是 `gpt-4` vs `gpt-3.5-turbo`：

推荐使用独立入口：

```powershell
python 11_run_position_bias_experiment.py --dry-run --question-limit 2
python 11_run_position_bias_experiment.py
```

默认 judge 是 `deepseek-v4-flash`，也就是 `--deepseek 1 --gemini 0 --xiaomi 0`。`--dry-run` 只打印将执行的命令，不写文件，也不调用 API；真正运行到 judge 阶段时会调用 API，可能花钱。

如果要调试底层步骤，可以手动运行：

```powershell
python 09_prepare_position_bias_trials.py --dry-run --limit 2
python 09_prepare_position_bias_trials.py
```

然后复用 length judge runner，但显式指定 position-bias 的输入和输出文件：

```powershell
python 07_run_length_bias_judge.py --trials position_bias_trials.jsonl --raw-output raw_position_bias_judgments.jsonl --parsed-output parsed_position_bias_judgments.jsonl --deepseek 1 --gemini 0 --xiaomi 0
```

最后分析 position-bias 结果：

```powershell
python 10_analyze_position_bias_results.py --dry-run
python 10_analyze_position_bias_results.py
```

会生成：

- `position_bias_summary.json`
- `position_bias_summary.txt`

## 使用 99 入口

如果已经有 `mt_bench_questions_answers_padded_deepseek.jsonl`，可以用 checked-all 串起严谨后半段：

```powershell
python 99_run_length_bias_experiment.py --stage checked-all --dry-run --limit 1 --deepseek 1 --gemini 0 --xiaomi 0
```

`checked-all --dry-run` 只打印将执行的命令，不写中间文件，也不调用 API。真正运行时去掉 `--dry-run`。

旧的默认入口仍然只串起 length-bias 的 `06_prepare -> 07_judge -> 08_analyze`：

```powershell
python 99_run_length_bias_experiment.py --dry-run --limit 1 --deepseek 1 --gemini 0 --xiaomi 0
```

## 每次改代码后怎么快速检查

最小检查可以这样跑：

```powershell
python -m py_compile "length_bias_manipulation_judge.py" "03_prepare_manipulation_check_trials.py" "04_run_manipulation_check_judge.py" "05_filter_manipulation_check_results.py" "06_prepare_length_bias_trials.py" "07_run_length_bias_judge.py" "08_analyze_length_bias_results.py" "09_prepare_position_bias_trials.py" "10_analyze_position_bias_results.py" "11_run_position_bias_experiment.py" "99_run_length_bias_experiment.py"
python -m unittest test_bias_framework.py
python 01_screen_length_bias_eligibility.py --dry-run
python 03_prepare_manipulation_check_trials.py --dry-run --limit 2
python 04_run_manipulation_check_judge.py --dry-run --limit 1 --deepseek 1 --gemini 0 --xiaomi 0
python 06_prepare_length_bias_trials.py --dry-run
python 09_prepare_position_bias_trials.py --dry-run --limit 2
python 11_run_position_bias_experiment.py --dry-run --question-limit 2
python 99_run_length_bias_experiment.py --stage checked-all --dry-run --limit 1 --deepseek 1 --gemini 0 --xiaomi 0
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
- `mt_bench_questions_answers_padded_deepseek_checked.jsonl`
- `length_bias_trials.jsonl`
- `parsed_length_bias_judgments.jsonl`
- `length_bias_summary.*`
- `position_bias_trials.jsonl`
- `parsed_position_bias_judgments.jsonl`
- `position_bias_summary.*`
- `manipulation_check_trials.jsonl`
- `parsed_manipulation_check_judgments.jsonl`
- `manipulation_check_excluded_samples.jsonl`
- 其他实验生成的 JSONL/TXT/SVG 结果

原因很简单：

- `.env` 里有密钥。
- `llm-judge-env/` 是本机虚拟环境。
- `FastChat/` 是外部数据和源码。
- raw/parsed 结果文件可能很大，而且应该按实验版本重新生成。

## Current pilot limitations

The current length-bias results are pilot evidence, not final claims.

- Current attrition is `80 -> 28 -> 17`: 80 screened MT-Bench rows, 28 eligible rows, and 17 questions with parsed judge results.
- The current parsed result shape is `204 rows = 17 questions x 2 prompts x 2 positions x 3 judges`.
- Category coverage is limited after screening and padding. Current analyzed questions cover only a subset of MT-Bench categories, mainly humanities, roleplay, and STEM.
- Position bias should be treated as a separate experiment from length bias. The length-bias pipeline swaps `long_A` / `long_B` for control, but final position-bias claims need their own dedicated design and analysis.
- Final length-bias claims require manipulation check evidence that padded answers preserve meaning while changing length enough to test the intended mechanism.
- Dry-run commands do not call paid APIs. Paid API usage starts when padding or judging scripts are run without `--dry-run`.
