import copy

import pytest
import torch
from torch import nn

from experiments.pamf_phase2_practical.group2_c_backbone_hu import train_hu_source_aux, fit_hu_b
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
def test_hu_source_aux_and_target_b_are_transferred_into_native_c(c_method):
    torch.manual_seed(1)
    source = SourceModel(c_method).eval()
    original = copy.deepcopy(source.state_dict())
    bundle = {'model': source, 'method': c_method, 'provenance': {'source_only': True},
              'source_checkpoint': 'source-fixture', 'source_model_seed': 7}
    source_x = torch.randn(24, 2)
    source_y = torch.arange(6).repeat_interleave(4)
    source_rx = torch.arange(24) % 5
    aux = train_hu_source_aux(bundle, source_x, source_y, source_rx, model_seed=7,
                              epochs=1, batch_size=12, embedding_dim=4)
    assert aux['metadata']['source_role'] == 'L_s'
    assert aux['metadata']['target_contact'] is False
    old_x, old_y = torch.randn(12, 2), torch.arange(6).repeat_interleave(2)
    fitted = fit_hu_b(bundle, aux, old_x, old_y, seed=7, model_seed=7, epochs=1, embedding_dim=4, required_source_epochs=1)
    assert fitted['metadata']['B_epochs'] == 1
    assert fitted['metadata']['B_rx_branch_frozen']
    assert fitted['metadata']['B_tx_projection_preserved_in_C']
    assert all(torch.equal(original[k], v) for k, v in source.state_dict().items())
    new_x, new_y = torch.randn(4, 2), torch.tensor([6, 6, 7, 7])
    rt = {'native_registration_overrides': {'mopc_hr': {'mopc_epochs': 1, 'mopc_batch_size': 4},
                                             'orthogonal': {'orthogonal_epochs': 1}}}
    state = register_native_c(rt, fitted['c_state'], new_x, new_y, [6, 7], seed=9)
    assert state.score(new_x).shape == (4, 8)


def test_hu_rejects_wrong_source_checkpoint_and_target_contamination():
    model = SourceModel('mopc_hr')
    bundle = {'model': model, 'method': 'mopc_hr', 'source_checkpoint': 'expected',
              'source_model_seed': 1, 'provenance': {'source_only': False}}
    x, y, rx = torch.randn(12, 2), torch.arange(6).repeat_interleave(2), torch.arange(12)%5
    with pytest.raises(ValueError, match='source'):
        train_hu_source_aux(bundle, x, y, rx, model_seed=1, epochs=1, embedding_dim=4)
    bundle['provenance']['source_only'] = True
    aux = train_hu_source_aux(bundle, x, y, rx, model_seed=1, epochs=1, embedding_dim=4)
    bundle['source_checkpoint'] = 'different'
    with pytest.raises(ValueError, match='source'):
        fit_hu_b(bundle, aux, x, y, seed=1, model_seed=1, epochs=1, embedding_dim=4, required_source_epochs=1)
    bundle['source_checkpoint'] = 'expected'
    aux['metadata']['model_seed'] = 2
    with pytest.raises(ValueError, match='seed'):
        fit_hu_b(bundle, aux, x, y, seed=1, model_seed=1, epochs=1, embedding_dim=4, required_source_epochs=1)


@pytest.mark.parametrize('c_method', ['mopc_hr', 'orthogonal', 'lora_triplet_knn'])
def test_hu_source_and_target_core_with_real_c_architecture(c_method):
    from experiments.pamf_phase2_practical.group1_source import model_for

    torch.set_num_threads(1)
    source = model_for(c_method, 'cpu').eval()
    bundle = {'model': source, 'method': c_method, 'provenance': {'source_only': True},
              'source_checkpoint': 'source-fixture', 'source_model_seed': 7}
    source_x = torch.randn(12, 2, 256)
    source_y = torch.arange(6).repeat_interleave(2)
    source_rx = torch.arange(12) % 5
    aux = train_hu_source_aux(bundle, source_x, source_y, source_rx, model_seed=7,
                              epochs=1, batch_size=12)
    fitted = fit_hu_b(bundle, aux, source_x, source_y, seed=7, model_seed=7, epochs=1, required_source_epochs=1)
    new_x, new_y = torch.randn(4, 2, 256), torch.tensor([6, 6, 7, 7])
    rt = {'native_registration_overrides': {'mopc_hr': {'mopc_epochs': 1, 'mopc_batch_size': 4},
                                             'orthogonal': {'orthogonal_epochs': 1}}}
    state = register_native_c(rt, fitted['c_state'], new_x, new_y, [6, 7], seed=9)
    assert state.score(new_x).shape == (4, 8)
