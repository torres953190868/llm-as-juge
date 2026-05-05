import importlib
import json
import tempfile
import unittest
from pathlib import Path

from length_bias_manipulation import CHECK_CRITERIA, make_manipulation_trial
from length_bias_manipulation import sample_sha256
from length_bias_metadata import run_metadata, sanitize_judge_config
from length_bias_position import build_position_trials
from length_bias_statistics import build_statistical_summary
from length_bias_judge import DEEPSEEK_MODEL, GEMINI_MODEL, XIAOMI_MODEL
from length_bias_judge_client import anthropic_messages_payload, extract_judge_content


class BiasFrameworkTests(unittest.TestCase):
    def test_position_trials_use_original_answer_fields(self):
        questions = {
            1: {
                "question_id": 1,
                "category": "writing",
                "question_turns": ["Question"],
            }
        }
        answers_a = {
            1: {
                "answer_id": "a1",
                "answer_turns": ["Answer from model A"],
                "answer": "Answer from model A",
            }
        }
        answers_b = {
            1: {
                "answer_id": "b1",
                "answer_turns": ["Answer from model B"],
                "answer": "Answer from model B",
            }
        }

        trials, skipped = build_position_trials(
            questions, answers_a, answers_b, "model-a", "model-b"
        )

        self.assertEqual([], skipped)
        self.assertEqual(2, len(trials))
        self.assertEqual({"model_a_A", "model_a_B"}, {row["condition"] for row in trials})
        self.assertTrue(all(row["bias_type"] == "position" for row in trials))
        self.assertTrue(all("padded_answer" not in row for row in trials))

    def test_position_swapped_pair_analysis_classifies_bias_patterns(self):
        analyzer = importlib.import_module("11_analyze_position_bias_results")

        def row(question_id, condition, winner, model_a_won):
            return {
                "bias_type": "position",
                "judge_model": "judge",
                "prompt_condition": "neutral_no_length",
                "question_id": question_id,
                "source_model_a": "model-a",
                "source_model_b": "model-b",
                "condition": condition,
                "winner": winner,
                "model_a_won": model_a_won,
            }

        summary = analyzer.summarize_swapped_pairs(
            [
                row(1, "model_a_A", "A", True),
                row(1, "model_a_B", "A", False),
                row(2, "model_a_A", "A", True),
                row(2, "model_a_B", "B", True),
                row(3, "model_a_A", "B", False),
                row(3, "model_a_B", "B", True),
            ]
        )

        self.assertEqual(3, summary["total_pairs"])
        self.assertEqual(1, summary["position_A_both"])
        self.assertEqual(1, summary["position_B_both"])
        self.assertEqual(1, summary["source_model_a_both"])

    def test_position_summary_includes_category_and_binomial_test(self):
        analyzer = importlib.import_module("11_analyze_position_bias_results")
        rows = [
            {
                "bias_type": "position",
                "judge_model": "judge",
                "category": "writing",
                "winner": winner,
                "model_a_won": model_a_won,
            }
            for winner, model_a_won in (
                ("A", True),
                ("A", False),
                ("A", True),
                ("B", False),
            )
        ]

        summary = analyzer.summarize(rows)

        self.assertEqual(4, summary["overall"]["position_binomial_n"])
        self.assertEqual(3, summary["overall"]["position_binomial_k"])
        self.assertIsNotNone(summary["overall"]["position_a_vs_b_binomial_p"])
        self.assertIn("writing", summary["by_category"])
        self.assertIn("judge", summary["swapped_pair_analysis_by_judge"])

    def test_position_orchestrator_builds_safe_default_dry_run_commands(self):
        orchestrator = importlib.import_module("run_position_bias_experiment")
        args = orchestrator.build_arg_parser().parse_args(
            ["--dry-run", "--question-limit", "2"]
        )

        cmds = orchestrator.selected_commands(args)
        rendered = [" ".join(cmd) for cmd in cmds]

        self.assertEqual(3, len(cmds))
        self.assertIn("09_prepare_position_bias_trials.py", rendered[0])
        self.assertIn("--limit 2", rendered[0])
        self.assertIn("10_run_position_bias_judge.py", rendered[1])
        self.assertIn("--trials position_bias_trials.jsonl", rendered[1])
        self.assertIn("--raw-output raw_position_bias_judgments.jsonl", rendered[1])
        self.assertIn("--parsed-output parsed_position_bias_judgments.jsonl", rendered[1])
        self.assertIn("--deepseek 1", rendered[1])
        self.assertIn("--gemini 1", rendered[1])
        self.assertIn("--opencode-go 0", rendered[1])
        self.assertIn("--xiaomi 1", rendered[1])
        self.assertIn("11_analyze_position_bias_results.py", rendered[2])

    def test_default_judge_configs_keep_gemini_and_use_official_deepseek_xiaomi(self):
        runner = importlib.import_module("07_run_length_bias_judge")
        args = runner.build_arg_parser().parse_args(["--dry-run"])

        configs = runner.load_judge_configs(args)
        judge_models = [config["judge_model"] for config in configs]
        gemini = configs[0]
        deepseek = configs[1]
        xiaomi = configs[2]

        self.assertEqual([GEMINI_MODEL, DEEPSEEK_MODEL, XIAOMI_MODEL], judge_models)
        self.assertEqual("GEMINI_API_KEY", gemini["api_key_env"])
        self.assertEqual("deepseek", deepseek["provider"])
        self.assertEqual("DEEPSEEK_API_KEY", deepseek["api_key_env"])
        self.assertEqual("xiaomi", xiaomi["provider"])
        self.assertEqual("XIAOMI_API_KEY", xiaomi["api_key_env"])
        self.assertEqual(
            "https://token-plan-cn.xiaomimimo.com/v1/chat/completions",
            xiaomi["endpoint"],
        )

    def test_minimax_messages_payload_and_parser(self):
        payload = anthropic_messages_payload(
            {
                "model": "minimax-m2.7",
                "messages": [
                    {"role": "system", "content": "Judge carefully."},
                    {"role": "user", "content": "Pick A or B."},
                ],
                "temperature": 0.0,
                "max_tokens": 128,
                "stream": False,
            }
        )
        content = extract_judge_content(
            {"api_mode": "anthropic_messages"},
            {"content": [{"type": "text", "text": "Reasoning.\n[[A]]"}]},
        )

        self.assertEqual("Judge carefully.", payload["system"])
        self.assertEqual([{"role": "user", "content": "Pick A or B."}], payload["messages"])
        self.assertEqual("Reasoning.\n[[A]]", content)

    def test_manipulation_trial_has_required_criteria_and_hashes(self):
        row = {
            "question_id": 7,
            "category": "humanities",
            "question": "Explain the topic.",
            "original_answer": "Original answer.",
            "padded_answer": "Original answer. In other words, original answer.",
            "length_ratio": 1.5,
        }

        trial = make_manipulation_trial(row, "inputhash", {"stage": "test"})

        criteria_names = {item["name"] for item in CHECK_CRITERIA}
        self.assertEqual(
            {
                "semantic_equivalence",
                "new_facts",
                "structure_improvement",
                "quality_improvement",
            },
            criteria_names,
        )
        self.assertEqual("manipulation_check", trial["bias_type"])
        self.assertEqual(64, len(trial["system_prompt_sha256"]))
        self.assertEqual(64, len(trial["user_prompt_sha256"]))
        self.assertEqual(64, len(trial["prompt_sha256"]))
        self.assertEqual(64, len(trial["sample_sha256"]))
        self.assertTrue(trial["trial_id"].startswith("q7_manipulation_"))
        self.assertIn("generated_at", trial)

    def test_manipulation_check_parser_uses_strict_pass_policy(self):
        runner = importlib.import_module("04_run_manipulation_check_judge")
        parsed = runner.parse_check_result(
            '{"semantic_equivalence": true, "new_facts": false, '
            '"structure_improvement": false, "quality_improvement": false, '
            '"explanation": "same meaning"}'
        )

        self.assertTrue(parsed["manipulation_passed"])
        self.assertFalse(
            runner.strict_passed(
                {
                    "semantic_equivalence": True,
                    "new_facts": False,
                    "structure_improvement": True,
                    "quality_improvement": False,
                }
            )
        )

    def test_filter_requires_all_judges_to_pass_strict_check(self):
        filter_mod = importlib.import_module("05_filter_manipulation_check_results")
        row = {
            "question_id": 7,
            "category": "humanities",
            "original_answer": "Original answer.",
            "padded_answer": "Original answer. In other words, original answer.",
            "length_ratio": 1.5,
        }
        base_check = {
            "question_id": 7,
            "sample_sha256": sample_sha256(row),
            "parse_status": "parsed",
            "semantic_equivalence": True,
            "new_facts": False,
            "structure_improvement": False,
            "quality_improvement": False,
            "manipulation_passed": True,
        }
        checks = [
            {**base_check, "judge_model": "judge-a"},
            {**base_check, "judge_model": "judge-b", "quality_improvement": True},
        ]

        accepted, excluded, required_judges, duplicates = filter_mod.filter_rows(
            [row], checks, {"stage": "test"}
        )

        self.assertEqual(["judge-a", "judge-b"], required_judges)
        self.assertEqual(set(), duplicates)
        self.assertEqual([], accepted)
        self.assertEqual(1, len(excluded))
        self.assertEqual("failed_strict_check", excluded[0]["excluded_reason"])

    def test_run_metadata_has_generated_at_and_sanitizes_secrets(self):
        metadata = run_metadata(input_path="missing.jsonl")
        sanitized = sanitize_judge_config(
            {
                "model": "judge",
                "api_key": "secret",
                "Authorization": "Bearer secret",
                "api_key_env": "OPENAI_API_KEY",
            }
        )

        self.assertIn("generated_at", metadata)
        self.assertIn("created_at", metadata)
        self.assertNotIn("api_key", sanitized)
        self.assertNotIn("Authorization", sanitized)
        self.assertEqual("OPENAI_API_KEY", sanitized["api_key_env"])

    def test_padding_output_compatibility_allows_same_input_and_config(self):
        from length_bias_padding_io import load_compatible_completed_ids

        metadata = self.padding_metadata()
        row = self.padding_row(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "padded.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            completed_ids = load_compatible_completed_ids(str(path), metadata)

        self.assertEqual({1}, completed_ids)

    def test_padding_output_compatibility_rejects_different_input_sha(self):
        from length_bias_padding_io import load_compatible_completed_ids

        metadata = self.padding_metadata(input_sha256="new-sha")
        row = self.padding_row(1, input_sha256="old-sha")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "padded.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Use --overwrite"):
                load_compatible_completed_ids(str(path), metadata)

    def test_padding_output_compatibility_rejects_different_model(self):
        from length_bias_padding_io import load_compatible_completed_ids

        metadata = self.padding_metadata(model="deepseek-v4-pro")
        row = self.padding_row(1, model="deepseek-v4-flash")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "padded.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "deepseek-v4-flash"):
                load_compatible_completed_ids(str(path), metadata)

    def test_padding_overwrite_skips_output_compatibility_check(self):
        runner = importlib.import_module("02_pad_answers_deepseek")
        metadata = self.padding_metadata(model="deepseek-v4-pro")
        row = self.padding_row(1, model="deepseek-v4-flash")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "padded.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            args = runner.build_arg_parser().parse_args(
                ["--overwrite", "--output-jsonl", str(path)]
            )

            completed_ids = runner.completed_ids_for_run(args, metadata)

        self.assertEqual(set(), completed_ids)

    def padding_metadata(
        self,
        input_path="length_bias_eligible_samples.jsonl",
        input_sha256="same-sha",
        model="deepseek-v4-pro",
        min_ratio=1.3,
        max_ratio=2.0,
        prompt_version="deepseek_padding_v5_retry_direction",
    ):
        return {
            "input_path": input_path,
            "input_sha256": input_sha256,
            "padding_config": {
                "model": model,
                "min_ratio": min_ratio,
                "max_ratio": max_ratio,
                "prompt_version": prompt_version,
            },
        }

    def padding_row(self, question_id, **metadata_overrides):
        metadata = self.padding_metadata(**metadata_overrides)
        return {
            "question_id": question_id,
            "padded_answer": "Padded answer.",
            "run_metadata": metadata,
        }

    def test_length_statistics_report_factorial_shape(self):
        rows = []
        for prompt in ("standard_anti_length", "neutral_no_length"):
            for condition, long_won in (("long_A", True), ("long_B", False)):
                rows.append(
                    {
                        "question_id": 1,
                        "judge_model": "judge",
                        "prompt_condition": prompt,
                        "condition": condition,
                        "winner": "A",
                        "long_answer_won": long_won,
                    }
                )

        summary = build_statistical_summary(rows, seed=1, iterations=10)

        self.assertEqual(4, summary["data_shape"]["rows"])
        self.assertTrue(summary["data_shape"]["is_full_factorial"])
        self.assertEqual(1, summary["question_cluster"]["clusters"])
        self.assertEqual(2, summary["swapped_paired"]["complete_pairs"])

    def test_length_analysis_deepseek_alias_includes_legacy_flash(self):
        analyzer = importlib.import_module("08_analyze_length_bias_results")
        args = analyzer.build_arg_parser().parse_args(
            [
                "--deepseek", "1",
                "--gemini", "0",
                "--opencode-go", "0",
                "--xiaomi", "0",
            ]
        )
        models = analyzer.selected_judge_models(args)
        rows = [
            {
                "bias_type": "length",
                "judge_model": "deepseek-v4-flash",
                "winner": "A",
                "long_answer_won": True,
            },
            {
                "bias_type": "length",
                "judge_model": "gemini-3-flash-preview",
                "winner": "B",
                "long_answer_won": False,
            },
        ]

        filtered = analyzer.filter_rows_by_judge(rows, models)
        available = analyzer.available_judge_models(rows)

        self.assertEqual(["deepseek-v4-flash"], [row["judge_model"] for row in filtered])
        self.assertEqual([], analyzer.missing_judge_models(models, available))

    def test_length_statistics_include_judge_prompt_cluster_ci(self):
        rows = [
            {
                "question_id": question_id,
                "judge_model": "judge",
                "prompt_condition": "neutral_no_length",
                "condition": "long_A",
                "winner": "A" if long_won else "B",
                "long_answer_won": long_won,
            }
            for question_id, long_won in ((1, True), (2, False))
        ]

        summary = build_statistical_summary(rows, seed=1, iterations=10)
        group = summary["by_judge_prompt"]["judge::neutral_no_length"]

        self.assertEqual(2, group["clusters"])
        self.assertEqual(2, group["valid_observations"])
        self.assertIn("lower", group["bootstrap_95_ci"])
        self.assertIn("upper", group["bootstrap_95_ci"])

    def test_length_svg_mentions_ci_and_complete_pair_legend(self):
        from length_bias_plotting import render_academic_svg

        counts = {
            "total": 4,
            "long_wins": 2,
            "short_wins": 1,
            "ties": 1,
            "invalid": 0,
        }
        summary = {
            "filtered_row_count": 4,
            "filtered_judge_models": ["judge"],
            "by_judge": {"judge": counts},
            "by_judge_prompt": {"judge::neutral_no_length": counts},
            "statistical_analysis": {
                "data_shape": {"questions": 2},
                "question_cluster": {
                    "bootstrap_95_ci": {"seed": 1, "iterations": 10}
                },
                "by_judge_prompt": {
                    "judge::neutral_no_length": {
                        "mean_net_length_preference": 0.25,
                        "bootstrap_95_ci": {"lower": 0.0, "upper": 0.5},
                    }
                },
            },
            "swapped_pair_analysis": {
                "by_judge": {
                    "judge": {
                        "total_pairs": 2,
                        "long_both_positions": 1,
                        "short_both_positions": 1,
                        "position_A_both": 0,
                        "position_B_both": 0,
                        "tie_both": 0,
                        "mixed_or_partial_tie": 0,
                        "invalid_pair": 0,
                        "missing_pair": 0,
                    }
                }
            },
        }

        svg = render_academic_svg(summary)

        self.assertIn("95% CI", svg)
        self.assertIn("Short both", svg)
        self.assertIn("B both", svg)


if __name__ == "__main__":
    unittest.main()
