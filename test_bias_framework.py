import importlib
import unittest

from length_bias_manipulation import CHECK_CRITERIA, make_manipulation_trial
from length_bias_manipulation import sample_sha256
from length_bias_metadata import run_metadata, sanitize_judge_config
from length_bias_position import build_position_trials
from length_bias_statistics import build_statistical_summary


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
        analyzer = importlib.import_module("10_analyze_position_bias_results")

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
        analyzer = importlib.import_module("10_analyze_position_bias_results")
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
        orchestrator = importlib.import_module("11_run_position_bias_experiment")
        args = orchestrator.build_arg_parser().parse_args(
            ["--dry-run", "--question-limit", "2"]
        )

        cmds = orchestrator.selected_commands(args)
        rendered = [" ".join(cmd) for cmd in cmds]

        self.assertEqual(3, len(cmds))
        self.assertIn("09_prepare_position_bias_trials.py", rendered[0])
        self.assertIn("--limit 2", rendered[0])
        self.assertIn("07_run_length_bias_judge.py", rendered[1])
        self.assertIn("--trials position_bias_trials.jsonl", rendered[1])
        self.assertIn("--raw-output raw_position_bias_judgments.jsonl", rendered[1])
        self.assertIn("--parsed-output parsed_position_bias_judgments.jsonl", rendered[1])
        self.assertIn("--deepseek 1", rendered[1])
        self.assertIn("--gemini 0", rendered[1])
        self.assertIn("--xiaomi 0", rendered[1])
        self.assertIn("10_analyze_position_bias_results.py", rendered[2])

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


if __name__ == "__main__":
    unittest.main()
