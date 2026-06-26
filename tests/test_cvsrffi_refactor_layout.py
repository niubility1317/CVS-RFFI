import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


class CvsRffiRefactorLayoutTest(unittest.TestCase):
    def test_common_training_helpers_live_in_cvsrffi_modules_and_train_reexports_them(self):
        import train
        from cvsrffi import checkpoint, eval as eval_utils, logging as log_utils, losses, presets, schedule, tensors

        expected = {
            tensors: [
                "set_seed",
                "unpack_batch",
                "extract_domain_from_extra",
                "parse_csv_indices",
                "safe_l2_normalize",
                "safe_iq_tensor",
                "make_torch_generator",
            ],
            losses: [
                "PrototypeMemoryBank",
                "SmoothGroupDROState",
                "compute_core_losses",
                "compute_aux_losses",
                "fishr_logit_gradient_variance_loss",
            ],
            eval_utils: [
                "evaluate_loader",
                "evaluate_named_loaders",
                "evaluate_sat_scenarios",
                "format_named_test_lines",
                "format_sat_test_lines",
                "aggregate_named_stats",
            ],
            schedule: [
                "build_stage_state",
                "current_weight_dict",
                "configure_augmentor_for_epoch",
                "configure_mixstyle_for_epoch",
                "training_stage_controller",
            ],
            checkpoint: [
                "AveragedModelState",
                "save_checkpoint",
                "derive_checkpoint_path",
            ],
            log_utils: [
                "format_epoch_block",
                "print_backbone_config_block",
                "format_weighted_loss_top",
            ],
            presets: [
                "parse_branch_ablation_flags",
                "apply_experiment_preset",
                "apply_slim_ablation_preset",
            ],
        }

        for module, names in expected.items():
            for name in names:
                with self.subTest(module=module.__name__, name=name):
                    self.assertIs(getattr(train, name), getattr(module, name))

    def test_removed_known_broken_compatibility_fallbacks(self):
        train_text = (CODE / "train.py").read_text(encoding="utf-8")
        self.assertNotIn("model_dual_cvsincnet_stagewise_v2", train_text)

        for rel_path in [
            "utils/sgc_diagnostics.py",
            "utils/sgc_freeze.py",
            "utils/sgc_metrics.py",
        ]:
            with self.subTest(path=rel_path):
                self.assertFalse((CODE / rel_path).exists())


if __name__ == "__main__":
    unittest.main()
