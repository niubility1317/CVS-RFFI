from types import SimpleNamespace
import pytest
import torch
from torch import nn
from cvsrffi import pair_reform_runtime as runtime


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.id_backbone = nn.Linear(4, 3, bias=False)
        self.dropout = nn.Dropout(.7)
        self.calls = []
    def forward(self, x, **kwargs):
        self.calls.append(len(x))
        z = self.dropout(self.id_backbone(x))
        return {'z_id': z, 'z_dom': z * 2, 'tx_logits': z}


def test_deterministic_pair_is_zero_for_identity_and_restores_modes():
    m = Toy().train()
    m.id_backbone.eval()
    x = torch.randn(5, 4)
    a, b = runtime.deterministic_pair(m, x, x, domain_labels=None)
    assert torch.equal(a['z_id'], b['z_id'])
    assert m.training and not m.id_backbone.training and m.dropout.training
    (a['z_id'] + b['z_id']).sum().backward()
    assert m.id_backbone.weight.grad is not None


def test_route_only_runs_subset_and_no_tangent_graph():
    m = Toy()
    x = torch.randn(8, 4)
    r = runtime.direction_objectives(m, x, x+1, domain_labels=None,
        tangent_weight=0., route_weight=.1, ratio=.25, seed=4,
        delta=.1, reference_scale=.1, budget=.1,
        nuisance_transform=lambda y: y,
        probe_transform=lambda y: y)
    assert r['forward_samples'] == 8
    assert 'tangent' not in r['weighted_components']
    assert 'route' in r['weighted_components']
    for k in ('delta_nui_id','delta_nui_dom','delta_fp_id','delta_fp_dom'):
        assert torch.allclose(r['diagnostics'][k], torch.zeros(2), atol=1e-6)


def test_disabled_directions_do_no_work():
    m = Toy()
    r = runtime.direction_objectives(m, torch.randn(3,4), torch.randn(3,4),
        domain_labels=None, tangent_weight=0., route_weight=0.,ratio=.25, seed=4,
        delta=.1, reference_scale=.1,budget=.1,
        nuisance_transform=lambda _: pytest.fail('disabled augmentation'),
        probe_transform=lambda _: pytest.fail('disabled augmentation'))
    assert m.calls == [] and r['weighted_components'] == {}


def test_family_seed_does_not_depend_on_optional_draws():
    first=runtime.sample_seed(4, 'rx/day/1', 3, 'main')
    runtime.sample_seed(4, 'rx/day/1', 3, 'probe')
    assert first == runtime.sample_seed(4, 'rx/day/1', 3, 'main')
    assert first != runtime.sample_seed(4, 'rx/day/1', 3, 'teacher')


def test_state_transaction_commits_only_applied_step():
    state=[]
    tx=runtime.StepStateTransaction()
    tx.stage(lambda: state.append(1))
    tx.finish(applied=False)
    assert state == []
    tx.stage(lambda: state.append(2))
    tx.finish(applied=True)
    assert state == [2]


def test_shared_batch_uses_only_teacher_and_unknown_u_keeps_features():
    teacher=Toy().eval()
    x=torch.randn(6,4)
    student=Toy()
    c=student(x); l=student(x+.1)
    args=SimpleNamespace(pair_reform='point', pair_weight=.5, pair_pseudo_weight=.2,
        pair_start_epoch=1,pair_pseudo_start_epoch=1,pair_alpha=.5,pair_u_tolerance=.1,
        pair_teacher_mix=0.,pair_unknown_quality='neutral',pair_tangent_weight=0.,
        pair_route_weight=0.)
    r=runtime.compute_pair_batch(model=student,teacher=teacher,clean=x,channel=x+.1,
        student_clean=c,student_channel=l,domains=None,labels=None,metadata={},
        args=args,epoch=1,physical_ids=list(range(6)))
    assert student.calls == [6,6]
    assert teacher.calls == [12]
    assert r['diagnostics']['physical_quality_unknown_count'] == 6
    assert r['diagnostics']['feature_weight_sum'] == 3
    r['loss'].backward()
    assert student.id_backbone.weight.grad is not None
    assert teacher.id_backbone.weight.grad is None


def test_pair_warmup_does_not_forward_teacher():
    m=Toy();x=torch.randn(3,4);o=m(x)
    args=SimpleNamespace(pair_reform='point',pair_start_epoch=10,pair_pseudo_start_epoch=20,
        pair_weight=.5,pair_pseudo_weight=.2,pair_tangent_weight=0.,pair_route_weight=0.)
    r=runtime.compute_pair_batch(model=m,teacher=m,clean=x,channel=x,student_clean=o,
        student_channel=o,domains=None,labels=None,metadata={},args=args,epoch=1,
        physical_ids=list(range(3)))
    assert m.calls == [3] and not r['loss'].requires_grad


def test_cache_changes_only_feature_target_and_aborted_step_does_not_write():
    from cvsrffi.orbit_teacher import DenseTemporalOrbitMemory
    from cvsrffi.deployment_orbit import stable_orbit_key_tensor
    from test_pair_reform_training import args_for
    teacher=Toy().eval(); student=Toy(); x=torch.randn(3,4)
    ids=[(0,0,0,i,i) for i in range(3)]
    keys=stable_orbit_key_tensor(ids,device=x.device)
    memory=DenseTemporalOrbitMemory(train_physical_ids=keys,feature_dim=3)
    memory.update(keys=keys,features=torch.randn(3,3),reliability=torch.ones(3)*.9,
        scenario_bin=torch.zeros(3,dtype=torch.long),receiver_bin=torch.zeros(3,dtype=torch.long),step=0)
    before=memory.features.clone(); tx=runtime.StepStateTransaction()
    kwargs=dict(model=student,teacher=teacher,clean=x,channel=x+.1,
        student_clean=student(x),student_channel=student(x+.1),domains=None,labels=None,
        metadata={'snr_db':torch.zeros(3)},args=args_for('point'),epoch=1,physical_ids=ids)
    fresh=runtime.compute_pair_batch(**kwargs)
    cached=runtime.compute_pair_batch(**kwargs,memory=memory,transaction=tx)
    for name in ('r_phys','q_cls','pseudo_weight_sum','feature_weight_sum'):
        assert torch.equal(fresh['diagnostics'][name],cached['diagnostics'][name])
    assert torch.equal(fresh['weighted_components']['orbit_logit'],cached['weighted_components']['orbit_logit'])
    tx.finish(applied=False)
    assert torch.equal(before,memory.features)
