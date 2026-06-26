import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


class SpaceborneFewShotTest(unittest.TestCase):
    def test_sfe_enrolls_new_class_without_moving_source_prototypes(self):
        from cvsrffi.spaceborne_fewshot import build_prototype_set, run_sfe_enrollment

        source = build_prototype_set(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0, 1]),
        )
        support_x = torch.tensor([[-1.0, 0.0], [-0.9, 0.1]])
        support_y = torch.tensor([2, 2])
        query_x = torch.tensor(
            [
                [0.95, 0.05],   # old class 0
                [-0.95, 0.02],  # new class 2
                [0.05, -0.95],  # unknown relative to enrolled bank
            ]
        )
        query_y = torch.tensor([0, 2, -1])

        result = run_sfe_enrollment(
            source,
            support_x,
            support_y,
            query_x,
            query_y,
            unknown_threshold=0.70,
        )

        self.assertTrue(torch.allclose(result.prototype_set.vectors[:2], source.vectors))
        self.assertEqual(result.predicted_labels.tolist(), [0, 2, -1])
        self.assertAlmostEqual(result.metrics["coverage"], 2 / 3)
        self.assertAlmostEqual(result.metrics["full_accuracy"], 1.0)
        self.assertAlmostEqual(result.metrics["new_class_accuracy"], 1.0)
        self.assertAlmostEqual(result.metrics["old_class_accuracy"], 1.0)
        self.assertAlmostEqual(result.metrics["unknown_rejection_rate"], 1.0)

    def test_ftrc_shrinkage_keeps_low_k_target_prototype_anchored_to_source(self):
        from cvsrffi.spaceborne_fewshot import build_prototype_set, shrink_target_prototypes

        source = build_prototype_set(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0, 1]),
        )
        support_x = torch.tensor([[0.0, 1.0]])
        support_y = torch.tensor([0])

        adapted = shrink_target_prototypes(
            source,
            support_x,
            support_y,
            kappa=3.0,
            drift_by_label={0: 1.0},
        )

        label0 = adapted.index_of(0)
        target_mean = torch.tensor([0.0, 1.0])
        source_vec = source.vectors[source.index_of(0)]
        adapted_vec = adapted.vectors[label0]

        self.assertGreater(torch.dot(adapted_vec, source_vec).item(), torch.dot(adapted_vec, target_mean).item())
        self.assertEqual(adapted.counts[label0].item(), 1)
        self.assertEqual(adapted.metadata["shrinkage"][0]["rho"], 0.2)

    def test_prediction_metrics_keep_full_denominator_and_accepted_accuracy_separate(self):
        from cvsrffi.spaceborne_fewshot import compute_open_set_metrics

        metrics = compute_open_set_metrics(
            true_labels=torch.tensor([0, 1, 2, -1]),
            predicted_labels=torch.tensor([0, -1, 1, -1]),
            accepted=torch.tensor([True, False, True, False]),
            old_labels={0, 1},
            new_labels={2},
        )

        self.assertAlmostEqual(metrics["coverage"], 0.5)
        self.assertAlmostEqual(metrics["full_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["accepted_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["old_class_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["new_class_accuracy"], 0.0)
        self.assertAlmostEqual(metrics["unknown_rejection_rate"], 1.0)

    def test_combined_gate_rejects_ambiguous_and_far_open_set_samples(self):
        from cvsrffi.spaceborne_fewshot import OpenSetGateConfig, build_prototype_set, predict_with_prototypes

        gate = OpenSetGateConfig(
            mode="combined",
            min_cosine=0.20,
            min_margin=0.05,
            max_mahalanobis=3.0,
            openmax_tail_size=4,
            openmax_quantile=1.0,
            openmax_min_threshold=0.05,
        )
        prototypes = build_prototype_set(
            torch.tensor(
                [
                    [1.0, 0.00, 0.0],
                    [0.99, 0.02, 0.0],
                    [0.00, 1.0, 0.0],
                    [0.02, 0.99, 0.0],
                ]
            ),
            torch.tensor([0, 0, 1, 1]),
            gate_config=gate,
        )

        pred = predict_with_prototypes(
            torch.tensor(
                [
                    [1.0, 0.01, 0.0],  # accepted class 0
                    [0.5, 0.5, 0.0],   # low margin between class 0/1
                    [0.1, 0.0, 1.0],   # high Mahalanobis/off-tail relative to class 0
                ]
            ),
            prototypes,
            gate_config=gate,
        )

        self.assertEqual(pred.predicted_labels.tolist()[0], 0)
        self.assertFalse(bool(pred.accepted[1]))
        self.assertFalse(bool(pred.accepted[2]))
        self.assertIn("low_margin", pred.gate_reasons[1])
        self.assertTrue("high_mahalanobis" in pred.gate_reasons[2] or "openmax_tail" in pred.gate_reasons[2])

    def test_openmax_tail_gate_rejects_far_cosine_distance_without_query_calibration(self):
        from cvsrffi.spaceborne_fewshot import OpenSetGateConfig, build_prototype_set, predict_with_prototypes

        gate = OpenSetGateConfig(mode="openmax", min_cosine=0.0, openmax_tail_size=2, openmax_quantile=1.0, openmax_min_threshold=0.05)
        prototypes = build_prototype_set(
            torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]),
            torch.tensor([0, 0, 1, 1]),
            gate_config=gate,
        )

        pred = predict_with_prototypes(torch.tensor([[-1.0, 0.0]]), prototypes, gate_config=gate)

        self.assertEqual(pred.predicted_labels.tolist(), [-1])
        self.assertEqual(pred.gate_reasons[0], "openmax_tail")

    def test_stage2_protocol_guard_blocks_target_new_leakage_and_unknown_query_calibration(self):
        from cvsrffi.spaceborne_fewshot import validate_stage2_protocol

        with self.assertRaisesRegex(ValueError, "target-new support"):
            validate_stage2_protocol("Stage2-B", use_target_new_support=True)
        with self.assertRaisesRegex(ValueError, "unknown query"):
            validate_stage2_protocol("Stage2-C", use_unknown_query_for_threshold_calibration=True)

        summary = validate_stage2_protocol(
            "Stage2-C",
            use_target_old_support=True,
            use_target_new_support=True,
            use_unknown_query_for_threshold_calibration=False,
        )
        self.assertEqual(summary["stage"], "Stage2-C")
        self.assertTrue(summary["target_new_support_allowed"])

    def test_oa_mse_head_scores_masked_subspace_and_preserves_defer_decision(self):
        from cvsrffi.spaceborne_fewshot import ClassState, OrbitAdaptiveMSEHead, predict_with_oa_mse_head

        states = {
            0: ClassState(
                class_id=0,
                group="old",
                prototype=torch.tensor([1.0, 0.0, 0.0]),
                mask=torch.tensor([1.0, 1.0, 0.0]),
                subspace=torch.tensor([[0.0], [1.0], [0.0]]),
                covariance_diag=torch.ones(3) * 0.25,
                thresholds={"max_residual": 0.20, "max_mahalanobis": 6.0, "min_margin": 0.05},
                evt_params={"tail_threshold": 0.25},
            ),
            1: ClassState(
                class_id=1,
                group="old",
                prototype=torch.tensor([0.0, 1.0, 0.0]),
                mask=torch.tensor([1.0, 1.0, 0.0]),
                subspace=torch.zeros(3, 0),
                covariance_diag=torch.ones(3) * 0.25,
                thresholds={"max_residual": 0.20, "max_mahalanobis": 6.0, "min_margin": 0.05},
                evt_params={"tail_threshold": 0.25},
            ),
        }
        head = OrbitAdaptiveMSEHead(dim=3, class_states=states, beta_residual=0.5, eta_mahalanobis=0.05)

        scores = head.compute_class_scores(torch.tensor([[1.0, 0.4, 0.0], [0.0, 0.0, 1.0]]))
        self.assertLess(scores[0]["residual"][0].item(), scores[0]["residual"][1].item())

        pred = predict_with_oa_mse_head(
            torch.tensor([[1.0, 0.4, 0.0], [1.0, 0.0, 0.0]]),
            head,
            quality_scores=torch.tensor([1.0, 0.1]),
            quality_threshold=0.5,
        )

        self.assertEqual(pred.predicted_labels.tolist()[0], 0)
        self.assertEqual(pred.decisions[0], "accept")
        self.assertFalse(bool(pred.accepted[1]))
        self.assertEqual(pred.decisions[1], "defer")
        self.assertIn("low_quality", pred.gate_reasons[1])

    def test_oa_mse_head_outputs_seen_new_reject_and_uncertain_decisions(self):
        from cvsrffi.spaceborne_fewshot import ClassState, OrbitAdaptiveMSEHead, predict_with_oa_mse_head

        thresholds = {
            "max_residual": 0.001,
            "reject_residual": 0.20,
            "max_mahalanobis": 1.0e6,
            "min_margin": -1.0,
        }
        states = {
            0: ClassState(
                class_id=0,
                group="old",
                prototype=torch.tensor([1.0, 0.0, 0.0]),
                mask=torch.ones(3),
                subspace=torch.zeros(3, 0),
                covariance_diag=torch.ones(3),
                thresholds=dict(thresholds),
                evt_params={"tail_threshold": 0.001},
            ),
            2: ClassState(
                class_id=2,
                group="seen_new",
                prototype=torch.tensor([0.0, 1.0, 0.0]),
                mask=torch.ones(3),
                subspace=torch.zeros(3, 0),
                covariance_diag=torch.ones(3),
                thresholds=dict(thresholds),
                evt_params={"tail_threshold": 0.001},
            ),
        }
        head = OrbitAdaptiveMSEHead(dim=3, class_states=states, beta_residual=0.5, eta_mahalanobis=0.01)

        pred = predict_with_oa_mse_head(
            torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [0.98, 0.20, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            head,
        )

        self.assertEqual(pred.predicted_labels.tolist()[0], 2)
        self.assertTrue(bool(pred.accepted[0]))
        self.assertEqual(pred.decisions[0], "accept")
        self.assertEqual(pred.decisions[1], "uncertain")
        self.assertEqual(pred.decisions[2], "reject")

    def test_oa_mse_seen_new_evidence_gate_separates_unknown_like_seen_new(self):
        from cvsrffi.spaceborne_fewshot import ClassState, OrbitAdaptiveMSEHead, predict_with_oa_mse_head

        states = {
            0: ClassState(
                class_id=0,
                group="old",
                prototype=torch.tensor([1.0, 0.0, 0.0]),
                mask=torch.ones(3),
                subspace=torch.zeros(3, 0),
                covariance_diag=torch.ones(3),
                thresholds={"max_residual": 1.0, "max_mahalanobis": 1.0e6, "min_margin": -1.0},
                evt_params={"tail_threshold": 1.0},
            ),
            2: ClassState(
                class_id=2,
                group="seen_new",
                prototype=torch.tensor([0.0, 1.0, 0.0]),
                mask=torch.ones(3),
                subspace=torch.zeros(3, 0),
                covariance_diag=torch.ones(3),
                thresholds={
                    "max_residual": 1.0,
                    "max_mahalanobis": 1.0e6,
                    "min_margin": -1.0,
                    "min_seen_new_support_affinity": 0.95,
                    "max_seen_new_support_residual": 0.05,
                    "min_seen_new_evidence": 0.90,
                },
                evt_params={"tail_threshold": 1.0},
            ),
        }
        head = OrbitAdaptiveMSEHead(dim=3, class_states=states, beta_residual=0.0, eta_mahalanobis=0.0)

        pred = predict_with_oa_mse_head(
            torch.tensor(
                [
                    [0.0, 1.0, 0.0],   # supported seen-new identity
                    [0.0, 0.80, 0.60],  # would be pulled into seen-new by score only
                ]
            ),
            head,
        )

        self.assertEqual(pred.predicted_labels.tolist(), [2, -1])
        self.assertEqual(pred.decisions, ["accept", "reject"])
        self.assertEqual(pred.gate_reasons[1], "seen_new_evidence_reject")
        self.assertIsNotNone(pred.seen_new_evidence)
        self.assertGreater(pred.seen_new_evidence[0].item(), pred.seen_new_evidence[1].item())

    def test_oa_mse_seen_new_anchor_gate_rejects_support_midpoint_unknown(self):
        from cvsrffi.spaceborne_fewshot import ClassState, OrbitAdaptiveMSEHead, predict_with_oa_mse_head

        states = {
            0: ClassState(
                class_id=0,
                group="old",
                prototype=torch.tensor([1.0, 0.0, 0.0]),
                mask=torch.ones(3),
                subspace=torch.zeros(3, 0),
                covariance_diag=torch.ones(3),
                thresholds={"max_residual": 1.0, "max_mahalanobis": 1.0e6, "min_margin": -1.0},
                evt_params={"tail_threshold": 1.0},
            ),
            2: ClassState(
                class_id=2,
                group="seen_new",
                prototype=torch.tensor([0.0, 0.70710677, 0.70710677]),
                mask=torch.ones(3),
                subspace=torch.zeros(3, 0),
                covariance_diag=torch.ones(3),
                thresholds={
                    "max_residual": 1.0,
                    "max_mahalanobis": 1.0e6,
                    "min_margin": -1.0,
                    "min_seen_new_support_affinity": 0.0,
                    "max_seen_new_support_residual": 1.0,
                    "min_seen_new_evidence": -1.0,
                    "min_seen_new_anchor_similarity": 0.90,
                },
                evt_params={"tail_threshold": 1.0},
                support_anchors=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            ),
        }
        head = OrbitAdaptiveMSEHead(dim=3, class_states=states, beta_residual=0.0, eta_mahalanobis=0.0)

        pred = predict_with_oa_mse_head(
            torch.tensor(
                [
                    [0.0, 1.0, 0.0],         # matches a real seen-new support anchor
                    [0.0, 0.70710677, 0.70710677],  # high prototype score, low support-density evidence
                ]
            ),
            head,
        )

        self.assertEqual(pred.predicted_labels.tolist(), [2, -1])
        self.assertEqual(pred.decisions, ["accept", "reject"])
        self.assertEqual(pred.gate_reasons[1], "seen_new_anchor_reject")
        self.assertEqual(pred.candidate_labels.tolist(), [2, 2])
        self.assertIsNotNone(pred.seen_new_anchor_similarity)
        self.assertGreater(pred.seen_new_anchor_similarity[0].item(), pred.seen_new_anchor_similarity[1].item())
        self.assertLess(pred.seen_new_anchor_delta[1].item(), 0.0)
        self.assertIn("seen_new_minus_old_score", pred.diagnostics)
        self.assertGreater(pred.diagnostics["seen_new_minus_old_score"][1].item(), 0.0)

    def test_register_old_classes_estimates_shared_orbit_subspace_and_source_weight(self):
        from cvsrffi.spaceborne_fewshot import OpenSetGateConfig, build_prototype_set, register_old_classes

        source = build_prototype_set(
            torch.tensor([[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.0, 1.0, 0.0], [0.01, 0.99, 0.0]]),
            torch.tensor([0, 0, 1, 1]),
            gate_config=OpenSetGateConfig(mode="combined", min_cosine=0.2),
        )
        states, u_orbit = register_old_classes(
            source,
            torch.tensor([[1.0, 0.25, 0.0], [0.25, 1.0, 0.0]]),
            torch.tensor([0, 1]),
            orbit_rank=1,
            active_ratio=0.67,
            stage="Stage2-B",
        )

        self.assertEqual(set(states), {0, 1})
        self.assertEqual(tuple(u_orbit.shape), (3, 1))
        self.assertGreater(states[0].source_weight, 0.0)
        self.assertLessEqual(states[0].subspace.shape[1], 1)
        self.assertIn("source_weight", states[0].thresholds)

    def test_eval_cli_oa_mse_gate_is_reachable_and_writes_decisions(self):
        import json
        import eval_spaceborne_fewshot

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out_path = tmp / "metrics.json"
            argv = [
                "eval_spaceborne_fewshot.py",
                "--protocol", "sfe",
                "--dry_run_synthetic",
                "--output_json", str(out_path),
                "--gate_mode", "oa_mse",
                "--unknown_threshold", "0.70",
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(eval_spaceborne_fewshot.main(), 0)

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["gate"]["mode"], "oa_mse")
            self.assertIn("oa_mse_head", payload["telemetry"])
            self.assertEqual(len(payload["decisions"]), len(payload["query_labels"]))
            score_table = Path(payload["score_table_csv"])
            self.assertTrue(score_table.exists())
            header = score_table.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("decision", header)
            self.assertIn("energy", header)
            self.assertIn("subspace_residual", header)
            self.assertIn("candidate_label", header)
            self.assertIn("candidate_group", header)
            self.assertIn("best_old_score", header)
            self.assertIn("best_seen_new_score", header)
            self.assertIn("seen_new_minus_old_score", header)
            self.assertIn("min_accept_delta", header)
            self.assertIn("seen_new_evidence", header)
            self.assertIn("seen_new_support_affinity", header)
            self.assertIn("seen_new_support_residual", header)
            self.assertIn("seen_new_anchor_similarity", header)
            self.assertIn("seen_new_anchor_delta", header)

    def test_eval_cli_oa_mse_stage2_a_b_reject_target_new_support_leakage(self):
        import eval_spaceborne_fewshot

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            npz_path = tmp / "features.npz"
            np.savez(
                npz_path,
                source_features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                source_labels=np.asarray([0, 1], dtype=np.int64),
                support_features=np.asarray([[-1.0, 0.0]], dtype=np.float32),
                support_labels=np.asarray([2], dtype=np.int64),
                query_features=np.asarray([[1.0, 0.0]], dtype=np.float32),
                query_labels=np.asarray([0], dtype=np.int64),
            )

            for protocol in ("source_open_set", "ftrc"):
                out_path = tmp / f"{protocol}.json"
                argv = [
                    "eval_spaceborne_fewshot.py",
                    "--protocol", protocol,
                    "--feature_npz", str(npz_path),
                    "--output_json", str(out_path),
                    "--gate_mode", "oa_mse",
                ]
                with self.subTest(protocol=protocol), mock.patch.object(sys, "argv", argv):
                    with self.assertRaisesRegex(ValueError, "support"):
                        eval_spaceborne_fewshot.main()

    def test_eval_cli_oa_mse_full_onboard_adaptation_modules_are_reachable(self):
        import json
        import eval_spaceborne_fewshot

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            npz_path = tmp / "features.npz"
            out_path = tmp / "metrics.json"
            np.savez(
                npz_path,
                source_features=np.asarray([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]], dtype=np.float32),
                source_labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
                support_features=np.asarray([[0.9, 0.1], [0.1, 0.9], [-1.0, 0.0], [-0.98, 0.02]], dtype=np.float32),
                support_labels=np.asarray([0, 1, 2, 2], dtype=np.int64),
                query_features=np.asarray([[0.92, 0.08], [0.08, 0.92], [-0.99, 0.01], [0.0, -1.0]], dtype=np.float32),
                query_labels=np.asarray([0, 1, 2, -1], dtype=np.int64),
                source_rx_ids=np.asarray(["rx0", "rx0", "rx1", "rx1"]),
                support_rx_ids=np.asarray(["rx7", "rx7", "rx7", "rx7"]),
                query_rx_ids=np.asarray(["rx7", "rx7", "rx7", "rx7"]),
            )
            argv = [
                "eval_spaceborne_fewshot.py",
                "--protocol", "sfe",
                "--feature_npz", str(npz_path),
                "--output_json", str(out_path),
                "--gate_mode", "oa_mse",
                "--oa_mse_adapter_steps", "20",
                "--old_acc_target", "0.90",
                "--seen_new_acc_target", "0.75",
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(eval_spaceborne_fewshot.main(), 0)

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            onboard = payload["telemetry"]["oa_mse_onboard_adaptation"]
            self.assertEqual(onboard["compute_profile"], "feature_level_low_rank_adapter_no_backbone_update")
            self.assertTrue(onboard["weibull_evt_required"])
            self.assertTrue(onboard["target_adapter"]["enabled"])
            self.assertTrue(onboard["pseudo_unknown_energy"]["enabled"])
            self.assertTrue(onboard["seen_new_evidence_gate"]["enabled"])
            self.assertTrue(onboard["seen_new_evidence_gate"]["anchor_gate_enabled"])
            self.assertFalse(onboard["seen_new_evidence_gate"]["unknown_query_threshold_calibration"])
            self.assertTrue(onboard["siamese_verifier"]["ambiguous_only"])
            self.assertEqual(onboard["online_update"]["update_policy"], "accepted_only")
            self.assertEqual(onboard["old_acc_target"], 0.90)
            self.assertEqual(onboard["seen_new_acc_target"], 0.75)
            self.assertTrue(payload["telemetry"]["stage2_receiver_domain"]["checked"])

    def test_eval_cli_oa_mse_rejects_target_receiver_overlap_for_old_and_new_support(self):
        import eval_spaceborne_fewshot

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            npz_path = tmp / "bad_receiver_features.npz"
            out_path = tmp / "metrics.json"
            np.savez(
                npz_path,
                source_features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                source_labels=np.asarray([0, 1], dtype=np.int64),
                support_features=np.asarray([[0.9, 0.1], [-1.0, 0.0]], dtype=np.float32),
                support_labels=np.asarray([0, 2], dtype=np.int64),
                query_features=np.asarray([[0.9, 0.1]], dtype=np.float32),
                query_labels=np.asarray([0], dtype=np.int64),
                source_rx_ids=np.asarray(["rx0", "rx1"]),
                support_rx_ids=np.asarray(["rx0", "rx7"]),
                query_rx_ids=np.asarray(["rx7"]),
            )
            argv = [
                "eval_spaceborne_fewshot.py",
                "--protocol", "sfe",
                "--feature_npz", str(npz_path),
                "--output_json", str(out_path),
                "--gate_mode", "oa_mse",
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "target receiver"):
                    eval_spaceborne_fewshot.main()

    def test_weibull_evt_fits_tail_without_unknown_query_calibration(self):
        from cvsrffi.spaceborne_fewshot import (
            OpenSetGateConfig,
            build_prototype_set,
            calibrate_thresholds,
            fit_weibull_tail,
            register_old_classes,
        )

        tail = fit_weibull_tail(torch.tensor([0.05, 0.08, 0.12, 0.20, 0.25]), tail_size=3, target_far=0.05)
        self.assertGreater(tail["shape"], 0.0)
        self.assertGreater(tail["scale"], 0.0)
        self.assertGreater(tail["threshold"], 0.20)

        source = build_prototype_set(
            torch.tensor([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]]),
            torch.tensor([0, 0, 1, 1]),
            gate_config=OpenSetGateConfig(mode="combined"),
        )
        states, _ = register_old_classes(
            source,
            torch.tensor([[1.0, 0.05], [0.05, 1.0]]),
            torch.tensor([0, 1]),
            stage="Stage2-B",
        )
        updated = calibrate_thresholds(
            states,
            torch.tensor([[1.0, 0.04], [0.98, 0.06], [0.04, 1.0], [0.06, 0.98]]),
            torch.tensor([0, 0, 1, 1]),
            evt_mode="weibull",
            target_far=0.05,
        )
        self.assertEqual(updated[0].evt_params["fit"], "weibull_moments")
        self.assertIn("weibull_shape", updated[0].evt_params)

        with self.assertRaisesRegex(ValueError, "unknown query"):
            calibrate_thresholds(
                states,
                torch.tensor([[1.0, 0.04], [0.04, 1.0]]),
                torch.tensor([0, 1]),
                surrogate_unknown=torch.tensor([[0.0, -1.0]]),
                unknown_source="unknown_query",
            )

    def test_low_compute_target_adapter_trains_on_old_and_new_support_only(self):
        from cvsrffi.spaceborne_fewshot import build_prototype_set, fit_low_compute_target_adapter

        source = build_prototype_set(
            torch.tensor([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]]),
            torch.tensor([0, 0, 1, 1]),
        )
        target_support = torch.tensor(
            [
                [0.72, 0.70],
                [0.70, 0.72],
                [-0.70, 0.72],
                [-0.72, 0.70],
                [-1.0, 0.0],
                [-0.98, 0.02],
            ]
        )
        target_labels = torch.tensor([0, 0, 1, 1, 2, 2])

        adapter, telemetry = fit_low_compute_target_adapter(
            source,
            target_support,
            target_labels,
            rank=2,
            steps=80,
            lr=0.08,
            old_acc_target=0.90,
            seen_new_acc_target=0.75,
        )
        adapted = adapter(target_support)
        logits = adapted @ telemetry["class_prototypes"].T
        predicted = telemetry["class_labels"][logits.argmax(dim=1)]

        self.assertLessEqual(telemetry["trainable_parameters"], 10)
        self.assertGreaterEqual(float((predicted[target_labels != 2] == target_labels[target_labels != 2]).float().mean().item()), 0.90)
        self.assertGreaterEqual(float((predicted[target_labels == 2] == target_labels[target_labels == 2]).float().mean().item()), 0.75)
        self.assertEqual(telemetry["training_scope"], "fewshot_target_old_and_seen_new_support_only")

    def test_pseudo_unknown_siamese_and_online_update_are_protocol_safe(self):
        from cvsrffi.spaceborne_fewshot import (
            ClassState,
            PredictionResult,
            accepted_only_online_update,
            apply_siamese_verifier_to_ambiguous,
            fit_siamese_verifier,
            generate_pseudo_unknown_features,
        )

        states = {
            0: ClassState(
                class_id=0,
                group="old",
                prototype=torch.tensor([1.0, 0.0]),
                mask=torch.ones(2),
                subspace=torch.zeros(2, 0),
                covariance_diag=torch.ones(2),
                thresholds={"max_residual": 0.2, "max_mahalanobis": 10.0},
            ),
            2: ClassState(
                class_id=2,
                group="seen_new",
                prototype=torch.tensor([0.0, 1.0]),
                mask=torch.ones(2),
                subspace=torch.zeros(2, 0),
                covariance_diag=torch.ones(2),
                thresholds={"max_residual": 0.2, "max_mahalanobis": 10.0},
            ),
        }
        pseudo = generate_pseudo_unknown_features(states, samples_per_pair=2)
        self.assertEqual(tuple(pseudo.shape), (2, 2))

        support = torch.tensor([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]])
        labels = torch.tensor([0, 0, 2, 2])
        verifier = fit_siamese_verifier(support, labels)
        result = PredictionResult(
            predicted_labels=torch.tensor([0, -1, 2]),
            scores=torch.tensor([0.4, 0.1, 0.4]),
            accepted=torch.tensor([False, False, False]),
            decisions=["uncertain", "reject", "uncertain"],
            gate_reasons=["oa_mse_uncertain", "oa_mse_reject", "oa_mse_uncertain"],
        )
        verified = apply_siamese_verifier_to_ambiguous(
            torch.tensor([[0.99, 0.01], [0.5, 0.5], [0.03, 0.97]]),
            result,
            states,
            verifier,
        )
        self.assertEqual(verified.decisions, ["accept", "reject", "accept"])
        self.assertEqual(verified.gate_reasons[1], "oa_mse_reject")

        updated, online = accepted_only_online_update(
            states,
            torch.tensor([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]]),
            verified,
            momentum=0.2,
        )
        self.assertEqual(online["updated_classes"], {0: 1, 2: 1})
        self.assertEqual(online["skipped_decisions"]["reject"], 1)
        self.assertFalse(torch.allclose(updated[0].prototype, states[0].prototype))

    def test_new_class_lifecycle_rejects_invalid_transition(self):
        from cvsrffi.spaceborne_fewshot import (
            LIFECYCLE_GROUND_CONFIRMED,
            LIFECYCLE_QUARANTINE,
            NewClassLifecycleManager,
        )

        manager = NewClassLifecycleManager()
        record = manager.enroll(7, support_count=3)

        self.assertEqual(record.state, LIFECYCLE_QUARANTINE)
        with self.assertRaisesRegex(ValueError, "invalid lifecycle transition"):
            manager.transition(record, LIFECYCLE_GROUND_CONFIRMED)

    def test_sfe_result_records_quarantine_lifecycle_and_gate_telemetry(self):
        from cvsrffi.spaceborne_fewshot import LIFECYCLE_QUARANTINE, OpenSetGateConfig, build_prototype_set, run_sfe_enrollment

        source = build_prototype_set(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]))
        result = run_sfe_enrollment(
            source,
            torch.tensor([[-1.0, 0.0], [-0.9, 0.1]]),
            torch.tensor([2, 2]),
            torch.tensor([[-0.95, 0.02]]),
            torch.tensor([2]),
            gate_config=OpenSetGateConfig(mode="cosine", min_cosine=0.70),
        )

        lifecycle = result.telemetry["new_class_lifecycle"]
        self.assertEqual(lifecycle[0]["label"], 2)
        self.assertEqual(lifecycle[0]["state"], LIFECYCLE_QUARANTINE)
        self.assertEqual(result.telemetry["gate"]["mode"], "cosine")

    def test_eval_cli_numpy_conversion_does_not_require_torch_from_numpy(self):
        import eval_spaceborne_fewshot

        arr = np.array([[1.0, 2.0]], dtype=np.float32)
        with mock.patch("torch.from_numpy", side_effect=TypeError("numpy ABI mismatch")):
            tensor = eval_spaceborne_fewshot.tensor_from_numpy_compatible(arr, dtype=torch.float32)

        self.assertTrue(torch.allclose(tensor, torch.tensor([[1.0, 2.0]])))

    def test_eval_cli_builds_sfe_payload_from_full_feature_npz_with_tx_audit(self):
        import json
        import eval_spaceborne_fewshot

        features = []
        tx_ids = []
        basis = {
            "old_a": [1.0, 0.0],
            "old_b": [0.0, 1.0],
            "new_c": [-1.0, 0.0],
            "unk_d": [0.0, -1.0],
        }
        for tx, base in basis.items():
            for i in range(8):
                features.append([base[0] + i * 0.001, base[1]])
                tx_ids.append(tx)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            npz_path = tmp / "features.npz"
            out_path = tmp / "metrics.json"
            np.savez(npz_path, features=np.asarray(features, dtype=np.float32), tx_ids=np.asarray(tx_ids))
            argv = [
                "eval_spaceborne_fewshot.py",
                "--protocol", "sfe",
                "--feature_npz", str(npz_path),
                "--output_json", str(out_path),
                "--source_tx_ids", "old_a,old_b",
                "--new_tx_ids", "new_c",
                "--unknown_tx_ids", "unk_d",
                "--shots", "2",
                "--source_proto_per_tx", "2",
                "--source_query_per_tx", "2",
                "--query_per_tx", "2",
                "--unknown_threshold", "0.70",
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(eval_spaceborne_fewshot.main(), 0)

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["manifest"]["source_tx_ids"], ["old_a", "old_b"])
            self.assertEqual(payload["manifest"]["new_tx_ids"], ["new_c"])
            self.assertEqual(payload["manifest"]["overlap_audit"]["source_new"], [])
            self.assertIn("gate", payload)
            self.assertIn("rollback", payload)
            self.assertIn("telemetry", payload)
            self.assertIn("coverage", payload["metrics"])

    def test_eval_cli_uses_export_manifest_when_cli_tx_ids_are_indices(self):
        import json
        import eval_spaceborne_fewshot

        features = []
        tx_ids = []
        for tx, base in {"TX0": [1.0, 0.0], "TX1": [0.0, 1.0], "TX2": [-1.0, 0.0], "TX3": [0.0, -1.0]}.items():
            for i in range(8):
                features.append([base[0] + i * 0.001, base[1]])
                tx_ids.append(tx)
        manifest = {
            "source_tx_ids": ["TX0", "TX1"],
            "new_tx_ids": ["TX2"],
            "unknown_tx_ids": ["TX3"],
        }

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            npz_path = tmp / "features.npz"
            out_path = tmp / "metrics.json"
            np.savez(
                npz_path,
                features=np.asarray(features, dtype=np.float32),
                tx_ids=np.asarray(tx_ids),
                manifest_json=json.dumps(manifest),
            )
            argv = [
                "eval_spaceborne_fewshot.py",
                "--protocol", "sfe",
                "--feature_npz", str(npz_path),
                "--output_json", str(out_path),
                "--source_tx_ids", "0,1",
                "--new_tx_ids", "2",
                "--unknown_tx_ids", "3",
                "--shots", "2",
                "--source_proto_per_tx", "2",
                "--source_query_per_tx", "2",
                "--query_per_tx", "2",
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(eval_spaceborne_fewshot.main(), 0)

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["manifest"]["source_tx_ids"], ["TX0", "TX1"])
            self.assertEqual(payload["manifest"]["new_tx_ids"], ["TX2"])
            self.assertEqual(payload["manifest"]["unknown_tx_ids"], ["TX3"])


if __name__ == "__main__":
    unittest.main()
