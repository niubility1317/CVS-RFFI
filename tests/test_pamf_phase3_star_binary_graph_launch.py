"""Synthetic immutable-launch and truth-isolation checks; no real subprocesses."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class BinaryGraphLaunchTests(unittest.TestCase):
    def launch(self):
        try:
            return importlib.import_module("experiments.pamf_phase3_open_world.star_binary_graph_launch")
        except ModuleNotFoundError as exc:
            self.fail(f"binary graph launcher is not implemented: {exc}")

    def fixture(self, base):
        m = self.launch()
        root = base / "runs" / m.RUN
        root.parent.mkdir()
        release = base / "releases" / (m.RUN + "-release")
        release.mkdir(parents=True)
        old_source = root.parent / m.SOURCE_RUN / "source"
        old_source.mkdir(parents=True)
        (old_source / "complete.json").write_text(json.dumps({"source_unknown_rejection": 0.1}), encoding="utf-8")
        pool = base / "public-pool"
        config = dict(run_id=m.RUN, code_commit="a" * 40, source_root=str(old_source),
                      U_public_root=str(pool / "discovery/public"), U_capsule_id="u-capsule", U_split_id="u-split",
                      Q_public_root=str(pool / "evaluation/public"), Q_capsule_id="q-capsule", Q_split_id="q-split",
                      output_root=str(root / "predict"), batch_size=128,
                      dura_checkpoint=str(base / "frozen-g.pt"), dura_release=str(base / "frozen-g-release"))
        runtime = dict(run_id=m.RUN, code_commit="a" * 40, run_root=str(root), release_root=str(release),
                       python="verified-python", launch_owner="synthetic-owner", allowed_gpus=list(range(8)),
                       minimum_free_mb=4096, max_added_workers_per_gpu=2, predict_config=config,
                       U_truth=str(base / "private/U.npz"), Q_truth=str(base / "private/Q.npz"),
                       targets_path=str(release / "configs/targets.json"))
        path = base / "runtime.json"
        path.write_text(json.dumps(runtime), encoding="utf-8")
        return m, root, runtime, path

    @staticmethod
    def run_stub():
        return types.SimpleNamespace(validate_config=lambda c: c, STATUS="STAR_BINARY_GRAPH_PREDICTIONS_COMPLETE")

    @staticmethod
    def prepare(root, runtime):
        (root / "logs").mkdir(parents=True)
        (root / "configs").mkdir()
        (root / "configs/predict.json").write_text(json.dumps(runtime["predict_config"]), encoding="utf-8")

    def test_runtime_is_fixed_and_predictor_cannot_receive_truth_or_replace_source(self):
        with tempfile.TemporaryDirectory() as directory:
            m, root, runtime, _ = self.fixture(Path(directory))
            with patch.dict(sys.modules, {m.__package__ + ".star_binary_graph_run": self.run_stub()}):
                self.assertEqual(m.validate(runtime), runtime)
                changes = [
                    lambda r: r.update(extra=True),
                    lambda r: r.update(run_id="another-run"),
                    lambda r: r.update(code_commit="invalid"),
                    lambda r: r.update(allowed_gpus=[0, 1]),
                    lambda r: r.update(max_added_workers_per_gpu=3),
                    lambda r: r["predict_config"].update(truth=r["Q_truth"]),
                    lambda r: r["predict_config"].update(code_commit="b" * 40),
                    lambda r: r["predict_config"].update(source_root=str(root / "new-source")),
                    lambda r: r["predict_config"].update(output_root=str(root / "other")),
                    lambda r: r["predict_config"].update(U_public_root=r["U_truth"]),
                    lambda r: r.update(run_root=r["predict_config"]["U_public_root"]),
                ]
                for change in changes:
                    with self.subTest(change=changes.index(change)):
                        candidate = copy.deepcopy(runtime)
                        change(candidate)
                        with self.assertRaises(ValueError):
                            m.validate(candidate)

    def test_start_copies_exact_config_and_records_background_driver_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            m, root, runtime, path = self.fixture(Path(directory))
            captured = []
            def spawn(argv, **kwargs):
                captured.append((argv, kwargs))
                return types.SimpleNamespace(pid=12345)
            with patch.dict(sys.modules, {m.__package__ + ".star_binary_graph_run": self.run_stub()}), \
                    patch.object(m.subprocess, "Popen", side_effect=spawn), \
                    patch.object(m, "__name__", "__main__"):
                m.start(path)
            copied = json.loads((root / "configs/predict.json").read_text(encoding="utf-8"))
            self.assertEqual(copied, runtime["predict_config"])
            self.assertEqual(copied["source_root"], str(root.parent / m.SOURCE_RUN / "source"))
            launch = json.loads((root / "launch.json").read_text(encoding="utf-8"))
            self.assertEqual(launch["pid"], 12345)
            self.assertEqual(launch["code_commit"], runtime["code_commit"])
            self.assertEqual(captured[0][0], [runtime["python"], "-B", "-u", "-m", m.__package__ + ".star_binary_graph_launch", "driver", "--runtime", str(path)])
            self.assertEqual(captured[0][1]["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertTrue(captured[0][1]["start_new_session"])
            self.assertTrue((root / "logs/driver.log").is_file())

    def test_existing_output_and_pipeline_state_are_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            m, root, runtime, path = self.fixture(Path(directory))
            self.prepare(root, runtime)
            marker = root / "pipeline_state.json"
            marker.write_bytes(b'{"preserved":"original-pid"}')
            before = marker.read_bytes()
            with patch.dict(sys.modules, {m.__package__ + ".star_binary_graph_run": self.run_stub()}), \
                    patch.object(m, "_job", side_effect=AssertionError("must not launch")):
                with self.assertRaises(FileExistsError):
                    m.start(path)
                with self.assertRaises(FileExistsError):
                    m.driver(path)
            self.assertEqual(marker.read_bytes(), before)

    def completed_job(self, m, root, runtime, calls, passed=False, fail=None, score_changes=None, predict_changes=None):
        def job(r, state, path, name, command, gpu):
            calls.append((name, command, gpu))
            state["jobs"][name] = dict(pid=23450 + len(calls), argv=command, gpu=gpu, status="COMPLETE")
            folder = root / name
            folder.mkdir()
            (folder / "partial.txt").write_text("keep", encoding="utf-8")
            if name == fail:
                return False
            marker = dict(run_id=m.RUN, code_commit=runtime["code_commit"])
            if name == "predict":
                marker.update(status="STAR_BINARY_GRAPH_PREDICTIONS_COMPLETE", registered_count=0, trusted_count=0)
                marker.update(predict_changes or {})
            else:
                marker.update(status="STAR_BINARY_GRAPH_SCORES_COMPLETE", primary_state="DA1_REG1",
                              primary_pass=passed, next_step="READY_FOR_SEPARATE_CONFIRMATION" if passed else "NO_PROMOTION")
                marker.update(score_changes or {})
            (folder / "complete.json").write_text(json.dumps(marker), encoding="utf-8")
            return True
        return job

    def test_single_gpu_prediction_precedes_cpu_score_even_with_zero_registry_and_failed_science(self):
        for passed in (False, True):
            with self.subTest(passed=passed), tempfile.TemporaryDirectory() as directory:
                m, root, runtime, path = self.fixture(Path(directory))
                self.prepare(root, runtime)
                calls = []
                with patch.dict(sys.modules, {m.__package__ + ".star_binary_graph_run": self.run_stub()}), \
                        patch.object(m.subprocess, "check_output", return_value="7, 8192\n2, 12000\n4, 3000\n"), \
                        patch.object(m, "_job", side_effect=self.completed_job(m, root, runtime, calls, passed)):
                    m.driver(path)
                self.assertEqual([c[0] for c in calls], ["predict", "score"])
                self.assertEqual([c[2] for c in calls], [2, None])
                self.assertEqual(calls[0][1], [runtime["python"], "-B", "-u", "-m", m.__package__ + ".star_binary_graph_run",
                                             "run", "--config", str(root / "configs/predict.json"), "--device", "cuda:0"])
                self.assertEqual(calls[1][1], [runtime["python"], "-B", "-u", "-m", m.__package__ + ".star_binary_graph_score",
                                             "--predictions-root", str(root / "predict"), "--U-truth", runtime["U_truth"],
                                             "--Q-truth", runtime["Q_truth"], "--targets", runtime["targets_path"], "--output", str(root / "score")])
                self.assertNotIn(runtime["U_truth"], calls[0][1])
                self.assertNotIn(runtime["Q_truth"], calls[0][1])
                state = json.loads((root / "pipeline_state.json").read_text(encoding="utf-8"))
                self.assertEqual(state["status"], "ANALYZED_DEVELOPMENT")
                self.assertEqual(state["primary_pass"], passed)
                self.assertFalse(state["final_phase3_complete"])
                self.assertEqual(state["primary_state"], "DA1_REG1")
                self.assertEqual(state["jobs"]["predict"]["pid"], 23451)
                self.assertEqual((root.parent / m.SOURCE_RUN / "source/complete.json").read_text(encoding="utf-8"),
                                 '{"source_unknown_rejection": 0.1}')

    def test_technical_failure_is_preserved_without_retry_or_following_stage(self):
        for fail in ("predict", "score", "gpu"):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                m, root, runtime, path = self.fixture(Path(directory))
                self.prepare(root, runtime)
                calls = []
                with patch.dict(sys.modules, {m.__package__ + ".star_binary_graph_run": self.run_stub()}), \
                        patch.object(m, "_gpu_free", return_value=[] if fail == "gpu" else [5]), \
                        patch.object(m, "_job", side_effect=self.completed_job(m, root, runtime, calls, fail=fail)):
                    m.driver(path)
                self.assertEqual([c[0] for c in calls], [] if fail == "gpu" else ["predict"] if fail == "predict" else ["predict", "score"])
                state = json.loads((root / "pipeline_state.json").read_text(encoding="utf-8"))
                self.assertEqual(state["status"], "FAILED_PRESERVED")
                if fail != "gpu":
                    self.assertEqual((root / fail / "partial.txt").read_text(encoding="utf-8"), "keep")

    def test_completion_identity_failures_cannot_be_reported_as_analyzed(self):
        cases = [("score", {"code_commit": "b" * 40}), ("score", {"run_id": "wrong"}),
                 ("score", {"primary_state": "DA0_REG0"}),
                 ("score", {"primary_pass": 0}), ("score", {"next_step": ""}),
                 ("predict", {"status": "INCOMPLETE"})]
        for role, change in cases:
            with self.subTest(role=role, change=change), tempfile.TemporaryDirectory() as directory:
                m, root, runtime, path = self.fixture(Path(directory))
                self.prepare(root, runtime)
                calls = []
                kwargs = {role + "_changes": change}
                with patch.dict(sys.modules, {m.__package__ + ".star_binary_graph_run": self.run_stub()}), \
                        patch.object(m, "_gpu_free", return_value=[0]), \
                        patch.object(m, "_job", side_effect=self.completed_job(m, root, runtime, calls, **kwargs)):
                    m.driver(path)
                state = json.loads((root / "pipeline_state.json").read_text(encoding="utf-8"))
                self.assertEqual(state["status"], "FAILED_PRESERVED")
                if role == "predict":
                    self.assertEqual([c[0] for c in calls], ["predict"])

    def test_cli_dispatches_start_and_driver_with_runtime(self):
        m = self.launch()
        for mode in ("start", "driver"):
            with self.subTest(mode=mode), patch.object(sys, "argv", [m.__name__, mode, "--runtime", "runtime.json"]), \
                    patch.object(m, mode, side_effect=lambda path: (mode, path)):
                self.assertEqual(m.main(), (mode, "runtime.json"))


if __name__ == "__main__":
    unittest.main()
