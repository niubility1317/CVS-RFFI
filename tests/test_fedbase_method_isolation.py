import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def test_fedbase_methods_are_available_from_isolated_package_and_legacy_shims():
    from Fedbase.FedFA import FedFAComplexCNN as NewFedFA
    from Fedbase.FedRIEI import fedriei_alternating_step as new_fedriei_step
    from Fedbase.FUCL import FUCL1DModel as NewFUCL
    from Fedbase.FUCL import FUCLSpectrogramCNN
    from Fedbase.FUCL import nt_xent_loss as new_nt_xent_loss
    from Fedbase.RAFL import RAFLPaperResNet2D as NewRAFL
    from Fedbase.RAFL import gradient_reverse as new_grad_reverse
    from Fedbase.RAFL import receiver_agnostic_loss as new_receiver_agnostic_loss
    from federated.contrastive_fl import nt_xent_loss as legacy_nt_xent_loss
    from federated.feature_alignment import FedFAComplexCNN as LegacyFedFA
    from federated.fedbase_paper_trainer import FUCL1DModel as TrainerFUCL
    from federated.fedbase_paper_trainer import RAFLPaperResNet2D as TrainerRAFL
    from federated.fedbase_paper_trainer import _grad_reverse as trainer_grad_reverse
    from federated.fedriei import fedriei_alternating_step as legacy_fedriei_step
    from federated.receiver_agnostic_fl import receiver_agnostic_loss as legacy_receiver_agnostic_loss

    assert TrainerFUCL is NewFUCL
    assert issubclass(NewFUCL, FUCLSpectrogramCNN)
    assert TrainerRAFL is NewRAFL
    assert trainer_grad_reverse is new_grad_reverse
    assert LegacyFedFA is NewFedFA
    assert legacy_fedriei_step is new_fedriei_step
    assert legacy_nt_xent_loss is new_nt_xent_loss
    assert legacy_receiver_agnostic_loss is new_receiver_agnostic_loss


def test_paper_method_folders_do_not_import_cvs_adapter_modules():
    forbidden = (
        "cvsrffi",
        "federated.client_split",
        "federated.fed_aggregate",
        "federated.fedbase_paper_trainer",
    )
    for method in ("FedRIEI", "FedFA", "FUCL", "RAFL"):
        for path in (ROOT / "Fedbase" / method).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                assert token not in text, f"{path} must not import CVS adapter token {token!r}"


def test_each_paper_method_lives_in_its_own_top_level_folder():
    fedbase_dir = ROOT / "Fedbase"
    assert not (fedbase_dir / "methods").exists(), "Fedbase must not contain a unified methods package"
    expected_modules = {
        "FedRIEI": {"__init__.py", "model.py", "steps.py", "compression.py"},
        "FedFA": {"__init__.py", "model.py", "losses.py"},
        "FUCL": {"__init__.py", "model.py", "losses.py", "aggregation.py", "views.py", "spectrogram.py"},
        "RAFL": {"__init__.py", "model.py", "losses.py", "selection.py", "grl.py"},
    }
    for method, required_files in expected_modules.items():
        method_dir = fedbase_dir / method
        assert method_dir.is_dir(), f"{method} must be an isolated top-level paper folder"
        for filename in required_files:
            assert (method_dir / filename).is_file(), f"{method}/{filename} is required"


def test_build_fedbase_paper_model_resolves_each_isolated_method():
    from Fedbase.FedFA import FedFAComplexCNN
    from Fedbase.FUCL import FUCL1DModel
    from Fedbase.RAFL import RAFLPaperResNet2D
    from baselines.riei_fd.model import RIEIModel
    from federated.fedbase_paper_trainer import build_fedbase_paper_model

    assert isinstance(build_fedbase_paper_model("fedriei", num_classes=4, num_receivers=2), RIEIModel)
    assert isinstance(build_fedbase_paper_model("fedfa", num_classes=4, num_receivers=2), FedFAComplexCNN)
    assert isinstance(build_fedbase_paper_model("fucl", num_classes=4, num_receivers=2), FUCL1DModel)
    assert isinstance(
        build_fedbase_paper_model("rafl", num_classes=4, num_receivers=2, rafl_input_channels=2),
        RAFLPaperResNet2D,
    )


def test_method_subdirectories_expose_expected_internal_modules():
    from Fedbase.FedFA.losses import pairwise_coral_alignment_loss
    from Fedbase.FedFA.model import FedFAComplexCNN
    from Fedbase.FedRIEI.steps import fedriei_alternating_step
    from Fedbase.FUCL.aggregation import encoder_only_state_dict
    from Fedbase.FUCL.losses import nt_xent_loss
    from Fedbase.FUCL.model import FUCL1DModel, FUCLSpectrogramCNN
    from Fedbase.FUCL.spectrogram import channel_independent_spectrogram
    from Fedbase.RAFL.grl import gradient_reverse
    from Fedbase.RAFL.losses import receiver_agnostic_loss
    from Fedbase.RAFL.model import RAFLPaperResNet2D
    from Fedbase.RAFL.selection import label_loss_driven_client_selection

    assert callable(pairwise_coral_alignment_loss)
    assert callable(fedriei_alternating_step)
    assert callable(encoder_only_state_dict)
    assert callable(nt_xent_loss)
    assert callable(channel_independent_spectrogram)
    assert callable(gradient_reverse)
    assert callable(receiver_agnostic_loss)
    assert callable(label_loss_driven_client_selection)
    assert FedFAComplexCNN.__name__ == "FedFAComplexCNN"
    assert issubclass(FUCL1DModel, FUCLSpectrogramCNN)
    assert RAFLPaperResNet2D.__name__ == "RAFLPaperResNet2D"
