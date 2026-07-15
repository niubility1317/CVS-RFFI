from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "code" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from export_adv3b02_effective8_torchscript import (  # noqa: E402
    ADV3B02IdentityRuntime,
    _trace_and_save,
)


class _TinyIdentityBackbone(torch.nn.Module):
    def forward(
        self,
        rows: torch.Tensor,
        y=None,
        return_aux: bool = True,
        domain_labels=None,
    ):
        del y, return_aux, domain_labels
        features = rows.mean(dim=2)
        logits = torch.stack((features[:, 0], features[:, 1]), dim=1)
        return {"feat_joint": features, "logits": logits}


class _TinyADV3B02(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = _TinyIdentityBackbone()

    def _pick_z_id(self, auxiliary):
        return auxiliary["feat_joint"]


def test_identity_runtime_trace_preserves_feature_and_logit_outputs() -> None:
    wrapper = ADV3B02IdentityRuntime(_TinyADV3B02()).eval()
    rows = torch.randn(4, 2, 16)
    traced = torch.jit.trace(wrapper, rows[:2], strict=False, check_trace=True)
    eager_feature, eager_logit = wrapper(rows)
    traced_feature, traced_logit = traced(rows)
    torch.testing.assert_close(traced_feature, eager_feature, rtol=0.0, atol=0.0)
    torch.testing.assert_close(traced_logit, eager_logit, rtol=0.0, atol=0.0)


def test_export_helper_reloads_runtime_for_explicit_numerical_parity(
    tmp_path: Path,
) -> None:
    wrapper = ADV3B02IdentityRuntime(_TinyADV3B02()).eval()
    rows = torch.randn(4, 2, 16)
    runtime = _trace_and_save(wrapper, rows[:2], tmp_path / "runtime.ts")
    eager_feature, eager_logit = wrapper(rows)
    runtime_feature, runtime_logit = runtime(rows)
    torch.testing.assert_close(runtime_feature, eager_feature, rtol=0.0, atol=0.0)
    torch.testing.assert_close(runtime_logit, eager_logit, rtol=0.0, atol=0.0)
