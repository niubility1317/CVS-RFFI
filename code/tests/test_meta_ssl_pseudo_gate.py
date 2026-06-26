import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.losses import masked_pseudo_label_ce_loss, prototype_agreement_pull_loss
from cvsrffi.ssl_pseudo_label import PseudoLabelGateConfig, select_pseudo_labels


def test_pseudo_gate_requires_confidence_margin_uncertainty_and_prototype_agreement():
    logits = torch.tensor(
        [
            [4.0, 0.1, 0.0],
            [1.2, 1.1, 0.0],
            [0.0, 4.0, 0.1],
            [0.0, 0.1, 4.0],
        ]
    )
    features = F.normalize(torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ), dim=1)
    prototypes = F.normalize(torch.eye(3), dim=1)

    result = select_pseudo_labels(
        logits,
        features=features,
        class_prototypes=prototypes,
        uncertainty=torch.tensor([0.01, 0.01, 0.09, 0.01]),
        receiver_ids=torch.tensor([0, 0, 1, 1]),
        config=PseudoLabelGateConfig(min_confidence=0.80, min_margin=0.05, max_uncertainty=0.08),
    )

    assert result["mask"].tolist() == [True, False, False, True]
    assert int(result["accepted_count"].item()) == 2

    ce_loss, coverage = masked_pseudo_label_ce_loss(logits, result["pseudo_y"], result["mask"])
    proto_loss, proto_cos = prototype_agreement_pull_loss(features, result["pseudo_y"], prototypes, result["mask"])
    assert torch.isfinite(ce_loss)
    assert coverage == 0.5
    assert torch.isfinite(proto_loss)
    assert proto_cos > 0.99
