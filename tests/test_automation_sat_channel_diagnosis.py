import unittest
from pathlib import Path


AUTOMATION_TOML = Path(
    "E:/codex/home/automations/"
    "cv-sincnet-n607-centralized-only-optimizer-v4/automation.toml"
)


class AutomationSatChannelDiagnosisTest(unittest.TestCase):
    def test_centralized_optimizer_prompt_contains_sat_channel_diagnosis(self):
        if not AUTOMATION_TOML.exists():
            self.skipTest(f"automation file not found: {AUTOMATION_TOML}")

        prompt_text = AUTOMATION_TOML.read_text(encoding="utf-8-sig")
        required_tokens = [
            "RIEI+Sat > CVS-RFFI > DRFIT+Sat",
            "concat SAT CE-only",
            "lambda_sat_cons=0",
            "z_id clean-sat",
            "GroupCE/Fishr min_domains=4",
            "satellite scenario",
            "satellite-aware checkpoint ranking",
            "branch drift",
        ]

        missing = [token for token in required_tokens if token not in prompt_text]
        self.assertFalse(missing, f"missing diagnosis tokens: {missing}")


if __name__ == "__main__":
    unittest.main()
