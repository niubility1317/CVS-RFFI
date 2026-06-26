import json
import os
import pathlib
import subprocess
import sys
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import codex_automation_fallback as caf  # noqa: E402


class AutomationFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(os.environ.get("TEMP", ".")) / "codex_automation_fallback_tests"
        if self.tmp.exists():
            for path in sorted(self.tmp.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        self.tmp.mkdir(parents=True, exist_ok=True)

    def write_text(self, rel, text):
        path = self.tmp / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_update_toml_preserves_prompt_model_and_sets_one_shot(self):
        toml = self.write_text(
            "automation.toml",
            textwrap.dedent(
                '''
                version = 1
                id = "cv-sincnet-post-run-log-analysis-and-tuning"
                kind = "cron"
                name = "Optimizer"
                prompt = "keep me"
                status = "PAUSED"
                rrule = "FREQ=DAILY;COUNT=1"
                model = "gpt-5.5"
                reasoning_effort = "xhigh"
                execution_environment = "local"
                cwds = ["E:\\\\old"]
                updated_at = 1
                '''
            ).strip()
            + "\n",
        )

        caf.update_automation_toml(
            toml,
            status="ACTIVE",
            rrule="FREQ=MINUTELY;COUNT=1",
            cwd=r"E:\type10-7",
            updated_at=1234567890,
        )

        text = toml.read_text(encoding="utf-8")
        data = caf.load_toml(toml)
        self.assertIn('prompt = "keep me"', text)
        self.assertEqual(data["status"], "ACTIVE")
        self.assertEqual(data["rrule"], "FREQ=MINUTELY;COUNT=1")
        self.assertEqual(data["execution_environment"], "local")
        self.assertEqual(data["cwds"], [r"E:\type10-7"])
        self.assertEqual(data["model"], "gpt-5.5")
        self.assertEqual(data["reasoning_effort"], "xhigh")
        self.assertEqual(data["updated_at"], 1234567890)

    def test_prepare_handoff_marks_active_requested_without_consuming(self):
        handoff = self.write_text(
            "handoff.json",
            json.dumps(
                {
                    "event_key": "centralized-idle:20260528_212504:9bcff8bc6ab7",
                    "optimizer_status": "UI_RUNTIME_BLOCKED",
                    "optimizer_thread_or_run_id": None,
                    "consumed_at": None,
                },
                indent=2,
            ),
        )

        result = caf.prepare_handoff_for_trigger(handoff, "fallback-run")

        data = json.loads(handoff.read_text(encoding="utf-8"))
        self.assertEqual(result["optimizer_status"], "ACTIVE_REQUESTED")
        self.assertEqual(data["optimizer_status"], "ACTIVE_REQUESTED")
        self.assertIsNone(data["optimizer_thread_or_run_id"])
        self.assertIsNone(data["consumed_at"])
        self.assertEqual(data["fallback_trigger_attempt"]["mode"], "fallback-run")

    def test_build_codex_exec_command_uses_single_prompt_argument(self):
        cmd = caf.build_codex_exec_command(
            node_path="node",
            codex_js=r"C:\codex\codex.js",
            cwd=r"E:\type10-7",
            model="gpt-5.4-mini",
            sandbox="read-only",
            output_last_message=r"C:\tmp\last.txt",
            prompt="Return exactly OK.",
        )

        self.assertEqual(cmd[-1], "Return exactly OK.")
        self.assertIn("--output-last-message", cmd)
        self.assertIn("gpt-5.4-mini", cmd)
        self.assertIn("read-only", cmd)

    def test_full_prompt_uses_original_optimizer_without_smoke_stop_rule(self):
        prompt = caf.build_full_prompt(
            {"id": "auto-id", "prompt": "ORIGINAL PROMPT"},
            pathlib.Path(r"E:\type10-7\handoff.json"),
            "event:key",
        )

        self.assertIn("Automation ID: auto-id", prompt)
        self.assertIn("ORIGINAL PROMPT", prompt)
        self.assertIn("event:key", prompt)
        self.assertNotIn("trigger-path smoke run", prompt)
        self.assertNotIn("Do not SSH/SCP/N607", prompt)

    def test_dispatch_with_fake_codex_writes_result_and_session_id(self):
        fake = self.write_text(
            "fake_codex.py",
            textwrap.dedent(
                '''
                import pathlib
                import sys
                args = sys.argv[1:]
                out = pathlib.Path(args[args.index("--output-last-message") + 1])
                out.write_text("FAKE_OK\\n", encoding="utf-8")
                print("session id: fake-session-123")
                print(args[-1])
                '''
            ).strip()
            + "\n",
        )
        report = self.tmp / "dispatch.json"

        result = caf.dispatch_codex_exec(
            command=[sys.executable, str(fake), "exec", "--output-last-message", str(self.tmp / "last.txt"), "PROMPT"],
            stdout_path=self.tmp / "stdout.txt",
            stderr_path=self.tmp / "stderr.txt",
            timeout_seconds=20,
        )
        report.write_text(json.dumps(result, indent=2), encoding="utf-8")

        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["session_id"], "fake-session-123")
        self.assertEqual(result["last_message"], "FAKE_OK")


if __name__ == "__main__":
    unittest.main()
