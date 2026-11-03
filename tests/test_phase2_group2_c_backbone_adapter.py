import copy

import pytest
import torch
from torch import nn

from experiments.pamf_phase2_practical.group2_c_backbone_adapter import (
    C_SOURCE_SPECS, CrossAdaptationAdapter, lora_source_centroid_head,
)


class SourceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv1d(2, 3, 1)
        self.last = nn.Conv1d(3, 4, 1)
        self.fc = nn.Linear(4, 6)

    def forward(self, x):
        z = self.last(torch.relu(self.first(x))).mean(-1)
        return {'features': z, 'tx_logits': self.fc(z)}


class SourceMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv1d(2, 3, 1)
        self.last = nn.Conv1d(3, 4, 1)

    def forward(self, x):
        return self.last(torch.relu(self.first(x))).mean(-1)


class SourceOrthogonal(SourceClassifier):
    def __init__(self):
        super().__init__()
        del self.fc
        self.register_buffer('classifier_weight', torch.randn(6, 4))

    def forward(self, x):
        z = self.last(torch.relu(self.first(x))).mean(-1)
        return {'features': z, 'tx_logits': torch.nn.functional.normalize(z, dim=1) @
                torch.nn.functional.normalize(self.classifier_weight, dim=1).T}


def test_dadda_uses_distinct_source_layers_and_preserves_native_logits():
    source = SourceClassifier().eval()
    before = copy.deepcopy(source.state_dict())
    adapter = CrossAdaptationAdapter(source, 'mopc_hr', 'dadda_sda', 4,
                                    tap_names=('first', 'last')).eval()
    x = torch.randn(4, 2, 9)
    original = source(x)
    output = adapter(x, return_activations=True)
    assert torch.allclose(output['features'], original['features'])
    assert torch.allclose(output['logits'], original['tx_logits'])
    assert output['global_features'].shape == (4, 3)
    assert output['local_features'].shape == (4, 4)
    assert len(output['activations']) == 2
    (output['global_features'].sum() + output['local_features'].sum()).backward()
    assert adapter.backbone.first.weight.grad is not None
    assert all(torch.equal(before[k], v) for k, v in source.state_dict().items())


def test_lora_requires_source_fitted_classifier_instead_of_random_fallback():
    source = SourceMetric()
    with pytest.raises(ValueError, match='source-fitted'):
        CrossAdaptationAdapter(source, 'lora_triplet_knn', 'mrior_sda', 4)
    head = nn.Linear(4, 6)
    adapter = CrossAdaptationAdapter(source, 'lora_triplet_knn', 'mrior_sda', 4,
                                    source_head=head,
                                    source_head_metadata={'source_role': 'L_s', 'target_contact': False,
                                                          'status': 'SOURCE_FITTED'})
    x = torch.randn(4, 2, 9)
    assert torch.allclose(adapter(x)['tx_logits'], head(source(x)))
    assert adapter.estimate_network(adapter(x)['features']).shape == (4, 1)


def test_two_stage_domain_branch_reverses_feature_gradient():
    source = SourceClassifier().eval()
    adapter = CrossAdaptationAdapter(source, 'mopc_hr', 'twostage_sda', 4,
                                    tap_names=('first', 'last')).eval()
    x = torch.randn(4, 2, 9)
    output = adapter(x)
    features = output['features']
    direct = adapter.domain_classifier(features).sum()
    direct_grad = torch.autograd.grad(direct, features, retain_graph=True)[0]
    reverse_grad = torch.autograd.grad(output['domain_logits'].sum(), features)[0]
    assert torch.allclose(reverse_grad, -direct_grad)


def test_layer_taps_must_exist_and_be_distinct():
    with pytest.raises(ValueError, match='tap'):
        CrossAdaptationAdapter(SourceClassifier(), 'mopc_hr', 'dadda_sda', 4,
                               tap_names=('first', 'first'))
    with pytest.raises(ValueError, match='tap'):
        CrossAdaptationAdapter(SourceClassifier(), 'mopc_hr', 'dadda_sda', 4,
                               tap_names=('first', 'missing'))


def test_auxiliary_initialization_is_seeded_and_does_not_change_caller_rng():
    source = SourceClassifier()
    state = torch.random.get_rng_state().clone()
    first = CrossAdaptationAdapter(source, 'mopc_hr', 'mrior_sda', 4, init_seed=42)
    assert torch.equal(state, torch.random.get_rng_state())
    second = CrossAdaptationAdapter(source, 'mopc_hr', 'mrior_sda', 4, init_seed=42)
    assert all(torch.equal(a, b) for a, b in zip(first.estimate_network.parameters(),
                                                 second.estimate_network.parameters()))


def test_orthogonal_source_head_is_trainable_for_b_without_mutating_source():
    source = SourceOrthogonal().eval()
    original_weight = source.classifier_weight.detach().clone()
    adapter = CrossAdaptationAdapter(source, 'orthogonal', 'mrior_sda', 4).eval()
    x = torch.randn(4, 2, 9)
    assert isinstance(adapter.backbone.classifier_weight, nn.Parameter)
    assert torch.allclose(adapter(x)['tx_logits'], source(x)['tx_logits'])
    adapter(x)['tx_logits'].sum().backward()
    assert adapter.backbone.classifier_weight.grad is not None
    assert torch.equal(source.classifier_weight, original_weight)


def test_lora_source_gallery_head_matches_distance_softmax_without_target_data():
    z = torch.tensor([[0., 0.], [2., 0.], [0., 1.], [0., 3.]])
    y = torch.tensor([0, 0, 1, 1])
    head, metadata = lora_source_centroid_head(z, y, classes=2,
                                              provenance={'source_only': True})
    q = torch.tensor([[1., 2.], [2., 4.]])
    centroids = torch.stack((z[y == 0].mean(0), z[y == 1].mean(0)))
    assert torch.allclose(head(q).softmax(1), (-torch.cdist(q, centroids).square()).softmax(1))
    assert metadata['target_contact'] is False and metadata['source_role'] == 'L_s'
    with pytest.raises(ValueError, match='source'):
        lora_source_centroid_head(z, y, classes=2, provenance={'source_only': False})


@pytest.mark.parametrize('profile', ['mrior_sda', 'dadda_sda', 'twostage_sda', 'radionet_ada_knn'])
def test_native_b_objectives_can_fit_c_backbone_on_source_and_support_only(profile):
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite.adaptation import fit_adaptation

    torch.manual_seed(3)
    source = SourceClassifier().eval()
    taps = ('first', 'last') if profile in ('dadda_sda', 'twostage_sda') else ()
    adapter = CrossAdaptationAdapter(source, 'mopc_hr', profile, 4, tap_names=taps)
    source_x, support_x = torch.randn(12, 2, 9), torch.randn(12, 2, 9)
    labels = torch.arange(6).repeat_interleave(2)
    cfg = dict(seed=19, device='cpu', batch_size=6, epochs=1, steps_per_epoch=1,
               estimate_steps=1, iterations=1, dann_epochs=1, lmmd_epochs=1,
               finetune_epochs=1, source_classes=list(range(6)))
    state = fit_adaptation(profile, adapter, support_x, labels, tuple(range(6)), cfg,
                           source_x=source_x, source_y=labels)
    assert state.score(support_x).shape == (12, 6)
    assert state.metadata['query_used_for_fit'] is False
    assert isinstance(state.model.base.backbone, SourceClassifier)


@pytest.mark.parametrize('c_method', ['mopc_hr', 'orthogonal', 'lora_triplet_knn'])
def test_native_architecture_taps_preserve_c_source_features(c_method):
    from experiments.pamf_phase2_practical.group1_source import model_for

    torch.set_num_threads(1)
    source = model_for(c_method, 'cpu').eval()
    spec = C_SOURCE_SPECS[c_method]
    args = {}
    if c_method == 'lora_triplet_knn':
        gallery = torch.randn(12, spec['embedding_dim'])
        labels = torch.arange(6).repeat_interleave(2)
        args['source_head'], args['source_head_metadata'] = lora_source_centroid_head(
            gallery, labels, classes=6, provenance={'source_only': True})
    adapter = CrossAdaptationAdapter(source, c_method, 'dadda_sda', spec['embedding_dim'],
                                    tap_names=spec['tap_names'], **args).eval()
    x = torch.randn(2, 2, 256)
    with torch.no_grad():
        raw = source(x)
        expected = raw if c_method == 'lora_triplet_knn' else raw['features']
        observed = adapter(x, return_activations=True)
    assert torch.allclose(observed['features'], expected)
    assert len(observed['activations']) == 2
    assert observed['global_features'].ndim == observed['local_features'].ndim == 2


def test_mopc_registration_inherits_adapted_c_source_backbone_and_support_prototypes():
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite.adaptation import fit_adaptation
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite.registration import register_new_classes
    from experiments.pamf_phase2_practical.group2_c_backbone_state import prepare_c_state

    torch.manual_seed(9)
    source = SourceClassifier().eval()
    source_before = copy.deepcopy(source.state_dict())
    adapter = CrossAdaptationAdapter(source, 'mopc_hr', 'mrior_sda', 4)
    x, y = torch.randn(12, 2, 9), torch.arange(6).repeat_interleave(2)
    cfg = dict(seed=8, device='cpu', batch_size=6, epochs=1, steps_per_epoch=1,
               estimate_steps=1, source_classes=list(range(6)))
    b = fit_adaptation('mrior_sda', adapter, x, y, tuple(range(6)), cfg, source_x=x, source_y=y)
    result = prepare_c_state(b, x, y, source_provenance={'source_only': True})
    anchor = result['anchor']
    expected = b.model.base.backbone.eval()(x)['features'].detach()
    assert torch.allclose(anchor.prototypes, torch.stack([expected[y == c].mean(0) for c in range(6)]))
    assert result['metadata']['C_inherits_adapted_encoder'] is True
    assert result['metadata']['old_prototypes'] == 'B_encoder_target_old_support'
    before = copy.deepcopy(anchor.model.state_dict())
    registered = register_new_classes(anchor, torch.randn(4, 2, 9), torch.tensor([6, 6, 7, 7]),
                                     [6, 7], config={'mopc_epochs': 1, 'mopc_batch_size': 4})
    assert registered.score(torch.randn(2, 2, 9)).shape == (2, 8)
    assert all(torch.equal(before[k], v) for k, v in anchor.model.state_dict().items())
    assert all(torch.equal(source_before[k], v) for k, v in source.state_dict().items())
    with pytest.raises(ValueError, match='source'):
        prepare_c_state(b, x, y, source_provenance={'source_only': False})


@pytest.mark.parametrize('c_method', ['orthogonal', 'lora_triplet_knn'])
def test_native_c_state_inherits_adapted_backbone(c_method):
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite.adaptation import fit_adaptation
    from experiments.pamf_phase2_practical.group2_c_backbone_state import prepare_c_state
    from experiments.pamf_phase2_practical.group2_lora_bridge import register_lora
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite.registration import register_new_classes

    torch.manual_seed(31)
    source = SourceOrthogonal() if c_method == 'orthogonal' else SourceMetric()
    extra = {}
    if c_method == 'lora_triplet_knn':
        extra['source_head'], extra['source_head_metadata'] = lora_source_centroid_head(
            torch.randn(12, 4), torch.arange(6).repeat_interleave(2), classes=6, provenance={'source_only': True})
    adapter = CrossAdaptationAdapter(source, c_method, 'mrior_sda', 4, **extra)
    x, y = torch.randn(12, 2, 9), torch.arange(6).repeat_interleave(2)
    cfg = dict(seed=12, device='cpu', batch_size=6, epochs=1, steps_per_epoch=1,
               estimate_steps=1, source_classes=list(range(6)))
    b = fit_adaptation('mrior_sda', adapter, x, y, tuple(range(6)), cfg, source_x=x, source_y=y)
    result = prepare_c_state(b, x, y, source_provenance={'source_only': True})
    anchor = result['anchor']
    assert result['metadata']['C_inherits_adapted_encoder']
    adapted = b.model.base.backbone.eval()
    new_x, new_y = torch.randn(4, 2, 9), torch.tensor([6, 6, 7, 7])
    if c_method == 'orthogonal':
        assert torch.equal(anchor.model.classifier_weight, adapted.classifier_weight)
        state = register_new_classes(anchor, new_x, new_y, [6, 7], config={'orthogonal_epochs': 1})
    else:
        assert torch.allclose(anchor.features, adapted(x))
        state = register_lora(anchor, new_x, new_y)
    assert state.score(new_x).shape == (4, 8)


@pytest.mark.parametrize('c_method', ['mopc_hr', 'orthogonal', 'lora_triplet_knn'])
def test_protonet_old_state_is_translated_and_scientific_difference_recorded(c_method):
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite.adaptation import fit_adaptation
    from experiments.pamf_phase2_practical.group2_c_backbone_state import prepare_c_state

    source = (SourceClassifier() if c_method == 'mopc_hr' else
              SourceOrthogonal() if c_method == 'orthogonal' else SourceMetric())
    extra = {}
    if c_method == 'lora_triplet_knn':
        extra['source_head'], extra['source_head_metadata'] = lora_source_centroid_head(
            torch.randn(12, 4), torch.arange(6).repeat_interleave(2), classes=6, provenance={'source_only': True})
    adapter = CrossAdaptationAdapter(source, c_method, 'protonet_cda', 4, **extra)
    x, y = torch.randn(12, 2, 9), torch.arange(6).repeat_interleave(2)
    b = fit_adaptation('protonet_cda', adapter, x, y, tuple(range(6)),
                       dict(seed=12, device='cpu', source_classes=list(range(6))))
    result = prepare_c_state(b, x, y, source_provenance={'source_only': True})
    if c_method == 'mopc_hr':
        assert torch.allclose(result['anchor'].prototypes, b.weight)
        assert 'native_MoPC_classifier_restored' in result['metadata']['B_decision_handoff']
    elif c_method == 'orthogonal':
        assert torch.allclose(result['anchor'].model.classifier_weight, b.weight)
        assert 'bias_discarded' in result['metadata']['B_decision_handoff']
    else:
        assert len(result['anchor'].features) == len(x)
        assert 'replaced_by_native_LoRa_kNN' in result['metadata']['B_decision_handoff']
