import copy

import pytest
import torch
from torch import nn

from experiments.pamf_phase2_practical.group2_c_backbone_poster import fit_poster_b
from experiments.pamf_phase2_practical.group2_c_backbone_fit import register_native_c


class SourceModel(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.encoder = nn.Linear(2, 4)
        self.method = method
        if method == 'mopc_hr':
            self.fc = nn.Linear(4, 6)
        elif method == 'orthogonal':
            self.register_buffer('classifier_weight', torch.randn(6, 4))

    def forward(self, x):
        z = self.encoder(x)
        if self.method == 'lora_triplet_knn':
            return z
        if self.method == 'mopc_hr':
            return {'features': z, 'tx_logits': self.fc(z)}
        return {'features': z, 'tx_logits': torch.nn.functional.normalize(z, dim=1) @
                torch.nn.functional.normalize(self.classifier_weight, dim=1).T}


@pytest.mark.parametrize('c_method', ['mopc_hr', 'orthogonal', 'lora_triplet_knn'])
def test_poster_dense_adaptation_is_carried_into_original_c_registration(c_method):
    from experiments.pamf_phase2_practical.group2_c_backbone_adapter import C_SOURCE_SPECS
    torch.manual_seed(12)
    source = SourceModel(c_method).eval()
    original = copy.deepcopy(source.state_dict())
    bundle = {'model': source, 'method': c_method, 'provenance': {'source_only': True},
              'source_checkpoint': 'fixture'}
    old_x, old_y = torch.randn(12, 2), torch.arange(6).repeat_interleave(2)
    fitted = fit_poster_b(bundle, old_x, old_y, seed=7, epochs=2, embedding_dim=4)
    assert fitted['metadata']['B_epochs'] == 2
    assert fitted['metadata']['source_backbone_frozen_in_B']
    assert all(torch.equal(original[k], v) for k, v in source.state_dict().items())
    new_x, new_y = torch.randn(4, 2), torch.tensor([6, 6, 7, 7])
    rt = {'native_registration_overrides': {'mopc_hr': {'mopc_epochs': 1, 'mopc_batch_size': 4},
                                             'orthogonal': {'orthogonal_epochs': 1}}}
    registered = register_native_c(rt, fitted['c_state'], new_x, new_y, [6, 7], seed=9)
    assert registered.score(new_x).shape == (4, 8)
    assert fitted['metadata']['B_dense_projector_preserved_in_C']


def test_poster_rejects_non_source_checkpoint_and_bad_support():
    model = SourceModel('mopc_hr')
    x, y = torch.randn(12, 2), torch.arange(6).repeat_interleave(2)
    with pytest.raises(ValueError, match='source'):
        fit_poster_b({'model': model, 'method': 'mopc_hr', 'provenance': {'source_only': False}},
                     x, y, seed=1, embedding_dim=4)
    with pytest.raises(ValueError, match='support'):
        fit_poster_b({'model': model, 'method': 'mopc_hr', 'provenance': {'source_only': True}},
                     x, torch.zeros(12, dtype=torch.long), seed=1, embedding_dim=4)


@pytest.mark.parametrize('c_method', ['mopc_hr', 'orthogonal', 'lora_triplet_knn'])
def test_poster_transplant_with_each_real_source_architecture(c_method):
    from experiments.pamf_phase2_practical.group1_source import model_for

    torch.set_num_threads(1)
    source = model_for(c_method, 'cpu').eval()
    bundle = {'model': source, 'method': c_method, 'provenance': {'source_only': True}}
    old_x, old_y = torch.randn(12, 2, 256), torch.arange(6).repeat_interleave(2)
    fitted = fit_poster_b(bundle, old_x, old_y, seed=7, epochs=1)
    new_x, new_y = torch.randn(4, 2, 256), torch.tensor([6, 6, 7, 7])
    rt = {'native_registration_overrides': {'mopc_hr': {'mopc_epochs': 1, 'mopc_batch_size': 4},
                                             'orthogonal': {'orthogonal_epochs': 1}}}
    state = register_native_c(rt, fitted['c_state'], new_x, new_y, [6, 7], seed=9)
    assert state.score(new_x).shape == (4, 8)
