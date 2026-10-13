"""Synthetic checks for the class-independent joint160 boundary; no data access."""
import json
from copy import deepcopy
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .model import TransferFeatures, from_phase1


def checks():
    from experiments.cvs_frequency_context50 import design as d
    from experiments.cvs_frequency_context50.model import build
    from experiments.cvs_reference_identity.model import ReferenceCVS
    from experiments.cvs_identity_ce.model import IdentityOnlyCVS
    torch.set_num_threads(2)
    torch.manual_seed(721)
    x = torch.randn(6, 2, 256)
    records = {}
    for arm in d.ARMS:
        row = next(r for r in d.rows() if r['arm'] == arm and r['model_seed'] == d.SEEDS[0])
        original = build(d.config(row), 'cpu').eval()
        reference = original.id_backbone.encoder
        backbone = reference.core.id_backbone
        head = backbone.cls_head
        # Exercise the projection as well as the reference gain in gradient checks.
        # This is synthetic verification, not a checkpoint or training mutation.
        with torch.no_grad():
            reference.response_gain.fill_(.05)
            old_aux = backbone(x, y=None, return_aux=True)
            assert torch.equal(backbone(x, y=None, return_aux=False), old_aux['logits'])
            response = reference.response_projection(reference.response(x))
            expected = old_aux['feat_joint'] + reference.response_gain.tanh() * response
            joint = reference.features(x)
            assert torch.equal(joint, expected)
            old_logits = original.forward_identity_only(x)['tx_logits']
            expected_logits = 30. * F.linear(F.normalize(joint, dim=1, eps=1e-4),
                                             F.normalize(head.weight, dim=1, eps=1e-4))
            assert torch.equal(old_logits, expected_logits)
            weight = head.weight
            saved_weight = weight.detach().clone()
            weight.normal_(0., 10.)
            assert torch.equal(reference.features(x), joint)
            weight.copy_(saved_weight)
            del head.weight
            assert torch.equal(reference.features(x), joint)
            head.weight = weight
        exported = from_phase1(original).eval()
        state = exported.encoder.state_dict()
        parent = reference.state_dict()
        class_key = 'core.id_backbone.cls_head.weight'
        assert set(state) == set(parent) - {class_key}
        assert all(torch.equal(value, parent[key]) for key, value in state.items())
        assert not any(name.endswith('cls_head.weight') or tuple(p.shape) == (6, 160)
                       for name, p in exported.named_parameters())
        assert not hasattr(exported, 'dom_backbone') and not hasattr(exported, 'nuisance')
        # Persistent and regenerated public buffers are inherited without reset.
        parent_buffers = dict(reference.named_buffers())
        assert all(torch.equal(value, parent_buffers[key])
                   for key, value in exported.encoder.named_buffers())
        with torch.no_grad():
            actual = exported.forward_features(x)
            assert actual.shape == (6, 160) and torch.equal(actual, joint)
            single = torch.cat([exported(v[None]) for v in x])
            assert torch.allclose(actual, single, atol=2e-5, rtol=2e-5)
            assert torch.isfinite(exported(torch.zeros_like(x))).all()
            assert torch.isfinite(exported.normalized_features(torch.zeros_like(x))).all()
            assert torch.equal(exported.normalized_features(x),
                               F.normalize(joint, dim=1, eps=1e-4))
        inputs = x.clone().requires_grad_(True)
        output = exported(inputs)
        objective = (output * torch.randn_like(output)).sum() + output.square().mean()
        objective.backward()
        named = dict(exported.encoder.named_parameters())
        kept = ['core.id_backbone.cls_head.base_norm.weight',
                'core.id_backbone.cls_head.base_norm.bias',
                'core.id_backbone.cls_head.pa_norm.weight',
                'core.id_backbone.cls_head.pa_norm.bias',
                'core.id_backbone.cls_head.gain', 'response_gain']
        kept += [name for name in named if name.startswith('response_projection.') or '.qkv.' in name]
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all() and inputs.grad.norm() > 0
        for name in kept:
            p = named[name]
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0, name
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in named.values())
        records[arm] = dict(shape=list(actual.shape), parameter_count=sum(p.numel() for p in exported.parameters()),
                            discarded_class_parameters=weight.numel(), kept_state_exact=True,
                            perturbed_or_deleted_W_joint_exact=True, original_cosine_logits_exact=True,
                            packet_independent=True, zero_finite=True,
                            checked_feature_gradients={name: float(named[name].grad.norm()) for name in kept},
                            input_gradient_norm=float(inputs.grad.norm()))
    negatives = 0
    def reject(call):
        nonlocal negatives
        try:
            call()
        except ValueError:
            negatives += 1
        else:
            raise AssertionError('Unsupported transfer path accepted')
    reject(lambda: from_phase1(nn.Identity()))
    reject(lambda: TransferFeatures(ReferenceCVS('reference_response')))
    bad = deepcopy(reference)
    bad.core.id_backbone.cls_head = nn.Identity()
    reject(lambda: TransferFeatures(bad))
    bad = deepcopy(reference)
    bad.core.id_backbone.nmfdu_gate = nn.Identity()
    reject(lambda: TransferFeatures(bad))
    reject(lambda: bad.core.id_backbone(x, feature_only=True))
    reject(lambda: from_phase1(SimpleNamespace(id_backbone=original.id_backbone, nonlinear=None)))
    reject(lambda: IdentityOnlyCVS().id_backbone(x, feature_only=True))
    reject(lambda: backbone(x, return_aux=True, feature_only=True))
    reject(lambda: backbone(x, y=torch.arange(6), feature_only=True))
    reject(lambda: exported(torch.zeros(2, 2, 128)))
    return dict(status='PASS', models=records, negative_cases=negatives,
                feature_schema='raw_joint160_then_optional_L2_eps1e-4',
                classifier_rows_absent=True, nuisance_absent=True,
                real_checkpoint_loaded=False, source_target_query_IQ_read=False)


if __name__ == '__main__':
    print(json.dumps(checks(), allow_nan=False))
