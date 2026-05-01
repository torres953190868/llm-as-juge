# llm-as-judge bias 实验

这个项目是用来研究一个问题：

> 让大模型当裁判时，它会不会偏心？

这里主要看两种偏心：

- **位置偏差**：同样两个回答，放在 A 位置会不会更容易赢？
- **长度偏差**：内容差不多时，更长的回答会不会更容易被裁判选中？

目前用的数据主要来自 FastChat 的 MT-Bench。

## 这个项目里有什么

根目录下主要是我自己写的实验脚本：

- `screen_length_bias_eligibility.py`：先筛题，排除不适合做“加长回答”的题。
- `pad_answers_deepseek.py`：用 DeepSeek 把原回答加长，但尽量不改变意思。
- `prepare_length_bias_trials.py`：把原回答和加长回答组成 A/B 对照实验。
- `run_length_bias_judge.py`：调用 judge 模型，让它在 A/B 之间做选择。
- `analyze_length_bias_results.py`：统计结果，看长回答、短回答、A 位置、B 位置各赢了多少。
- `run_length_bias_experiment.py`：目前只串起 `prepare -> judge -> analyze`，还没有把 `screen` 和 `pad` 串进去。

简单说，流程是：

```text
筛题 -> 加长回答 -> 做 A/B 对照 -> 让模型当裁判 -> 统计偏差
```

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

### 1. 先筛题

先 dry-run 看看会筛出多少题：

```powershell
python screen_length_bias_eligibility.py --dry-run
```

确认没问题后，真正写出筛选结果：

```powershell
python screen_length_bias_eligibility.py
```

这一步会生成：

- `length_bias_screened_samples.jsonl`
- `length_bias_eligible_samples.jsonl`
- `length_bias_screening_excluded_samples.jsonl`
- `length_bias_screening_summary.json`

### 2. 给回答加长

建议显式使用筛出来的 eligible 样本：

```powershell
python pad_answers_deepseek.py --input length_bias_eligible_samples.jsonl --input-format jsonl --dry-run
```

确认样本能正常读取后，再真正调用 API：

```powershell
python pad_answers_deepseek.py --input length_bias_eligible_samples.jsonl --input-format jsonl
```

这一步会调用 DeepSeek API，可能花钱。

### 3. 生成 A/B 实验题

先检查：

```powershell
python prepare_length_bias_trials.py --dry-run
```

再真正生成：

```powershell
python prepare_length_bias_trials.py
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
python run_length_bias_judge.py --dry-run --limit 1 --deepseek 1 --gemini 0 --xiaomi 0
```

真正运行时要明确指定你想用哪些 judge。比如只跑 DeepSeek：

```powershell
python run_length_bias_judge.py --deepseek 1 --gemini 0 --xiaomi 0
```

或者用配置文件：

```powershell
python run_length_bias_judge.py --judge-config judge_config.example.json
```

这一步会真实调用 API，可能花钱。

### 5. 分析结果

先 dry-run：

```powershell
python analyze_length_bias_results.py --dry-run
```

再写出结果：

```powershell
python analyze_length_bias_results.py
```

会生成：

- `length_bias_summary.json`
- `length_bias_summary.txt`
- `length_bias_summary.svg`

## 每次改代码后怎么快速检查

最小检查可以这样跑：

```powershell
python -m py_compile "length_bias_common.py" "length_bias_samples.py"
python screen_length_bias_eligibility.py --dry-run
python prepare_length_bias_trials.py --dry-run
python analyze_length_bias_results.py --dry-run
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
- 其他实验生成的 JSONL/TXT/SVG 结果

原因很简单：

- `.env` 里有密钥。
- `llm-judge-env/` 是本机虚拟环境。
- `FastChat/` 是外部数据和源码。
- raw/parsed 结果文件可能很大，而且应该按实验版本重新生成。

## 现在这个项目还需要改进什么

当前项目已经能跑 pilot，但还不是最终版。

后面最重要的改进是：

- 把 `screen -> pad -> prepare -> judge -> analyze` 全部串成一个统一入口。
- 在分析里增加“同一题交换 A/B 后的成对统计”。
- 加几个最小测试，防止后面改脚本时把核心逻辑改坏。
- 把最终实验配置和结果版本记录得更清楚。
