# -_- coding: utf-8 -_-

## llm-judge-env 是虚拟环境

## FastChat是数据集

## MT-bench Human Annotation Dataset：

针对 80 个 MT-Bench 问题上的 6 个模型回答。
主要用途是验证“LLM 作为裁判”靠不靠谱，也就是比较 GPT-4 裁判和人工裁判的一致性。

1. 题目层
   FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
   内容: 80 个 MT-Bench 问题
   字段: question_id, category, turns

2. 模型回答层
   FastChat/fastchat/llm_judge/data/mt_bench/model_answer/\*.jsonl
   内容: 各个模型对同一批题目的回答
   字段: question_id, model_id, choices, answer_id, tstamp

3. judgment 层
   A. GPT-4 judgment
   FastChat/fastchat/llm_judge/data/mt_bench/model_judgment/gpt-4_single.jsonl
   或
   FastChat/fastchat/llm_judge/data/mt_bench/model_judgment/gpt-4_pair.jsonl

   B. Human judgment
   不在仓库本地默认提供
   来自 Hugging Face: lmsys/mt_bench_human_judgments

可以这样理解：
question.jsonl
= 试卷题目

model_answer/\*.jsonl
= 不同模型的答卷

gpt-4_single.jsonl / gpt-4_pair.jsonl
= AI 阅卷结果

mt_bench_human_judgments
= 人工阅卷结果

## 实验设计
