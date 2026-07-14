import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model import PhysicalAwareClassifier  # noqa: E402


def test_feat_cls_is_ungated_identity_core_while_joint_feature_keeps_defect_gate():
    torch.manual_seed(20260714)
    head = PhysicalAwareClassifier(
        base_dim=6,
        emb_dim=4,
        num_classes=3,
        drop=0.0,
        gate_alpha=0.8,
        use_dac=True,
        use_pa=True,
    ).eval()
    base = torch.randn(5, 6)
    dac = torch.randn(5, 4)
    pa = torch.randn(5, 4)

    with torch.no_grad():
        head.id_gate[0].weight.zero_()
        head.id_gate[0].bias.fill_(-8.0)
        low_gate = head._compute_features_for_head(base, dac, pa, None, None)
        head.id_gate[0].bias.fill_(8.0)
        high_gate = head._compute_features_for_head(base, dac, pa, None, None)

    assert torch.allclose(low_gate[0], high_gate[0], atol=1e-7, rtol=0.0)
    assert not torch.allclose(low_gate[4], high_gate[4], atol=1e-5, rtol=0.0)
