import pytest
import torch
from torch import nn

from experiments.pamf_phase2_practical.group2_c_backbone_fit import fit_b_from_c_source, register_native_c


class SourceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv1d(2, 3, 1)
        self.layer2 = nn.Conv1d(3, 4, 1)
        self.fc = nn.Linear(4, 6)

    def forward(self, x):
        z = self.layer2(torch.relu(self.layer1(x))).mean(-1)
        return {'features': z, 'tx_logits': self.fc(z)}


@pytest.mark.parametrize('b_method', ['protonet_cda', 'mrior_sda', 'dadda_sda',
                                      'twostage_sda', 'radionet_ada_knn'])
def test_b_from_c_source_then_mopc_registration_without_query(b_method):
    torch.manual_seed(3)
    bundle = dict(model=SourceModel().eval(), method='mopc_hr', profile='mopc_hr',
                  provenance={'source_only': True}, source_checkpoint='source-only-fixture')
    old_x, old_y = torch.randn(12, 2, 9), torch.arange(6).repeat_interleave(2)
    override = dict(epochs=1, steps_per_epoch=1, estimate_steps=1, iterations=1,
                    dann_epochs=1, lmmd_epochs=1, finetune_epochs=1, batch_size=6)
    runtime = {'native_adaptation_overrides': {b_method: override},
               'native_registration_overrides': {'mopc_hr': {'mopc_epochs': 1, 'mopc_batch_size': 4}}}
    fitted = fit_b_from_c_source(runtime, bundle, b_method, old_x, old_y, seed=12,
                                 source_x=old_x, source_y=old_y,
                                 spec={'embedding_dim': 4, 'tap_names': ('layer1', 'layer2')})
    state = register_native_c(runtime, fitted['c_state'], torch.randn(4, 2, 9),
                              torch.tensor([6, 6, 7, 7]), [6, 7], seed=13)
    assert state.score(torch.randn(2, 2, 9)).shape == (2, 8)
    assert fitted['metadata']['target_old_labels'] == 12
    assert fitted['metadata']['query_used_for_fit'] is False
    assert fitted['metadata']['cross_method_transplant'] is True


def test_unsupported_b_and_unverified_source_fail_closed():
    bundle = dict(model=SourceModel(), method='mopc_hr', provenance={'source_only': False})
    x, y = torch.randn(12, 2, 9), torch.arange(6).repeat_interleave(2)
    with pytest.raises(ValueError, match='source'):
        fit_b_from_c_source({}, bundle, 'protonet_cda', x, y, seed=1)
    bundle['provenance']['source_only'] = True
    with pytest.raises(ValueError, match='unbound'):
        fit_b_from_c_source({}, bundle, 'feature_separation_hu2024', x, y, seed=1)
