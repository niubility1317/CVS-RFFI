import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "conversation_index.py"


def load_module():
    spec = importlib.util.spec_from_file_location("conversation_index", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConversationIndexTest(unittest.TestCase):
    def test_build_index_filters_to_type10_project_and_searches_keywords(self):
        ci = load_module()
        with tempfile.TemporaryDirectory(prefix="conversation_index_test_", dir=ROOT) as tmp:
            base = Path(tmp)
            codex_home = base / ".codex"
            summaries = codex_home / "memories" / "rollout_summaries"
            sessions = codex_home / "sessions" / "2026" / "05" / "25"
            summaries.mkdir(parents=True)
            sessions.mkdir(parents=True)

            project_session = sessions / "rollout-2026-05-25T18-41-47-thread-project.jsonl"
            project_session.write_text(
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "thread-project",
                            "cwd": r"E:\type10-7",
                            "timestamp": "2026-05-25T10:41:47.906Z",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            (summaries / "2026-05-25T10-41-47-project.md").write_text(
                "\n".join(
                    [
                        "thread_id: thread-project",
                        "updated_at: 2026-05-25T10:46:18+00:00",
                        f"rollout_path: {project_session}",
                        r"cwd: \\?\E:\type10-7",
                        "",
                        "# CVS-RFFI concat satellite comparison was monitored",
                        "",
                        "Outcome: success",
                        "",
                        "Reusable knowledge:",
                        "- concat satellite channel augmentation strict replication",
                    ]
                ),
                encoding="utf-8",
            )

            (summaries / "2026-05-25T10-20-08-other.md").write_text(
                "\n".join(
                    [
                        "thread_id: thread-other",
                        "updated_at: 2026-05-25T10:26:15+00:00",
                        r"rollout_path: C:\Users\lh594\.codex\sessions\other.jsonl",
                        r"cwd: C:\Users\lh594\Documents\Codex\2026-05-25\new-chat",
                        "",
                        "# Composer context prompt was restored",
                    ]
                ),
                encoding="utf-8",
            )

            output_dir = base / "conversation_index"
            entries = ci.build_index(
                project_root=Path(r"E:\type10-7"),
                codex_home=codex_home,
                output_dir=output_dir,
            )

            self.assertEqual([entry.thread_id for entry in entries], ["thread-project"])
            self.assertTrue((output_dir / "type10_7_conversations.json").exists())
            self.assertTrue((output_dir / "type10_7_conversations.md").exists())

            index = ci.load_index(output_dir / "type10_7_conversations.json")
            results = ci.search_entries(index, "concat satellite", limit=5)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].entry.thread_id, "thread-project")
            self.assertGreater(results[0].score, 0)

    def test_session_scan_keeps_project_cwd_when_no_summary_exists(self):
        ci = load_module()
        with tempfile.TemporaryDirectory(prefix="conversation_index_test_", dir=ROOT) as tmp:
            base = Path(tmp)
            codex_home = base / ".codex"
            session_dir = codex_home / "sessions" / "2026" / "05" / "26"
            session_dir.mkdir(parents=True)
            session_file = session_dir / "rollout-2026-05-26T11-00-00-thread-session.jsonl"
            session_file.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "session_meta",
                                "payload": {
                                    "id": "thread-session",
                                    "cwd": r"\\?\E:\type10-7\code",
                                    "timestamp": "2026-05-26T11:00:00.000Z",
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response_item",
                                "payload": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "input_text",
                                            "text": "\n".join(
                                                [
                                                    r"# AGENTS.md instructions for E:\type10-7",
                                                    "<INSTRUCTIONS>",
                                                    "project context",
                                                    "</INSTRUCTIONS>",
                                                    "<environment_context>",
                                                    r"<cwd>E:\type10-7</cwd>",
                                                    "</environment_context>",
                                                    "check federated launcher",
                                                ]
                                            ),
                                        }
                                    ],
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            entries = ci.build_index(
                project_root=Path(r"E:\type10-7"),
                codex_home=codex_home,
                output_dir=base / "conversation_index",
            )

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].source_kind, "session")
            self.assertEqual(entries[0].thread_id, "thread-session")
            self.assertEqual(entries[0].title, "check federated launcher")
            self.assertIn("federated launcher", entries[0].summary)


if __name__ == "__main__":
    unittest.main()
