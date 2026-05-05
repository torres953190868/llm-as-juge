import importlib
import unittest


class ManipulationCheckRunnerTests(unittest.TestCase):
    def test_default_judge_is_official_deepseek_with_parallel_three(self):
        runner = importlib.import_module("04_run_manipulation_check_judge")
        args = runner.build_arg_parser().parse_args(["--dry-run"])

        configs = runner.load_judge_configs(args)

        self.assertEqual(3, args.parallel)
        self.assertEqual(15, args.progress_interval)
        self.assertEqual(["deepseek-v4-pro"], [config["judge_model"] for config in configs])
        self.assertEqual("deepseek", configs[0]["provider"])
        self.assertEqual("DEEPSEEK_API_KEY", configs[0]["api_key_env"])
        self.assertEqual(
            "https://api.deepseek.com/chat/completions",
            configs[0]["endpoint"],
        )


if __name__ == "__main__":
    unittest.main()
