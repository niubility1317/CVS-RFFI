import copy
import json
from pathlib import Path
import subprocess
import sys
import torch
import torch.nn.functional as F
import pytest
from experiments.cvs_identity_ce.model import IdentityOnlyCVS
from experiments.cvs_residual_identity.model import build, VARIANTS, ResidualPhysicalHead, prune_unused_identity_modules
from experiments.cvs_residual_identity.source import validate_config

torch.set_num_threads(2)


def test_unused_pruning_preserves_training_logits_and_active_gradients():
    torch.manual_seed(71)
    native = IdentityOnlyCVS()
    pruned = prune_unused_identity_modules(copy.deepcopy(native))
    x = torch.randn(4, 2, 256)
    y = torch.arange(4) % 6
    torch.manual_seed(92); a = native(x)
    torch.manual_seed(92); b = pruned(x)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    F.cross_entropy(a, y).backward(); F.cross_entropy(b, y).backward()
    reduced = dict(pruned.named_parameters())
    removed = []
    for name, p in native.named_parameters():
        if name not in reduced:
            assert p.grad is None
            removed.append(name)
        elif p.grad is not None:
            torch.testing.assert_close(p.grad, reduced[name].grad, atol=0, rtol=0)
    assert sum(p.numel() for p in native.parameters()) == 382146
    assert sum(p.numel() for p in pruned.parameters()) == 317665
    assert removed


def test_fusion_disabled_pa_is_direct_identity_path():
    head = ResidualPhysicalHead(16)
    base = torch.randn(3,16,requires_grad=True)
    pa = torch.randn(3,16,requires_grad=True)
    with torch.no_grad():head.gain.zero_()
    identity, physical, joint = head.components(base, pa)
    torch.testing.assert_close(joint, head.base_norm(base), atol=0,rtol=0)
    F.cross_entropy(head.classify(joint),torch.tensor([0,1,2])).backward()
    assert base.grad.norm() > 0
    assert head.gain.grad.norm() > 0
    assert torch.isfinite(physical).all()


def test_fusion_delta_and_bounded_gain():
    head = ResidualPhysicalHead(16)
    base,pa,delta=[torch.randn(3,16) for _ in range(3)]
    identity,physical,joint=head.components(base,pa,delta)
    torch.testing.assert_close(physical,head.pa_norm(pa+delta))
    torch.testing.assert_close(joint,identity+0.25*physical)
    with pytest.raises(ValueError):head.forward_logits(base,base,pa,labels=torch.zeros(3))


@pytest.mark.parametrize('variant',VARIANTS)
def test_ce_updates_both_native_and_physical_paths_without_parameter_growth(variant):
    model=build(variant)
    expected=164225 if variant=='residual_fusion' else 164417
    assert sum(p.numel() for p in model.parameters())==expected
    assert not model.id_backbone.use_dac_path
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0002)
    x=torch.randn(4,2,256);y=torch.arange(4)%6
    before=model.id_backbone.cls_head.gain.detach().clone()
    loss=F.cross_entropy(model(x),y);loss.backward()
    for p in [model.id_backbone.t_proj.weight,model.id_backbone.f_proj.weight,model.id_backbone.pa_proj[0].weight,model.id_backbone.cls_head.gain]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0
    optimizer.step()
    assert not torch.equal(before,model.id_backbone.cls_head.gain)


@pytest.mark.parametrize('variant',VARIANTS)
def test_frozen_per_sample_features_and_degenerate_inputs(variant):
    model=build(variant).eval()
    x=torch.randn(3,2,256)
    with torch.no_grad():
        a=model(x); b=model(x[:1]);features=model.features(x)
        assert features.shape==(3,160)
        torch.testing.assert_close(a[:1],b,atol=2e-4,rtol=2e-4)
        for special in [torch.zeros(2,2,256),torch.ones(2,2,256)]:assert torch.isfinite(model(special)).all()


def test_scratch_source_guards_and_fresh_import():
    config=dict(method='cvs_residual_identity',variant='residual_fusion',epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,
        drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005)
    validate_config(config)
    for key,value in [('checkpoint','old.pt'),('p1_capsule','target'),('augmentation',True),('variant','native')]:
        with pytest.raises(ValueError):validate_config(dict(config,**{key:value}))
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,'-m','experiments.cvs_residual_identity.source','--help'],cwd=root,check=True,capture_output=True)


def test_profile_is_disposable_and_measures_active_costs():
    from experiments.cvs_clean_design.profile import resource_profile
    model=build('residual_fusion').eval()
    state={k:v.clone() for k,v in model.state_dict().items()}
    profile=resource_profile(model,torch.device('cpu'))
    assert profile['total_parameters']==164225
    assert profile['conv_linear_macs_per_sample']>0
    assert profile['training_batch128_ms']>0
    assert not profile['benchmark_state_used_for_training']
    for key,value in model.state_dict().items():torch.testing.assert_close(value,state[key],atol=0,rtol=0)


def test_existing_queue_drains_before_new_owner_can_launch(monkeypatch):
    import experiments.cvs_residual_identity.dispatch as dispatcher
    from types import SimpleNamespace
    payload={'text':''}
    monkeypatch.setattr(dispatcher,'Path',lambda path:SimpleNamespace(read_text=lambda **kwargs:payload['text']))
    predecessor_queue_drained=dispatcher.predecessor_queue_drained
    path='source_pipeline.json'
    rows={str(i):dict(status='RUNNING') for i in range(32)}
    rows['31']['status']='QUEUED'
    payload['text']=json.dumps(dict(rows=rows,target_access=False))
    assert not predecessor_queue_drained(path)
    rows['31']['status']='RUNNING'
    payload['text']=json.dumps(dict(rows=rows,target_access=False))
    assert predecessor_queue_drained(path)
    payload['text']='{'
    assert not predecessor_queue_drained(path)
    rows['31']['status']='UNKNOWN'
    payload['text']=json.dumps(dict(rows=rows,target_access=False))
    with pytest.raises(ValueError):predecessor_queue_drained(path)


def test_selection_ignores_target_scores_and_combines_all_six_source_candidates():
    from experiments.cvs_residual_identity.dispatch import select_source_candidate,combine_research_selection
    records=[dict(variant=v,seed=seed,accuracy=.98,worst_rx=.95,parameters=164225,macs=9708836,target_accuracy=0.1 if v=='residual_fusion' else .99)
        for v in VARIANTS for seed in [2026092701,2026092702,2026092703,2026092704]]
    own=select_source_candidate(records)
    assert own['selected_variant']=='residual_fusion'
    with pytest.raises(ValueError):select_source_candidate(records[:-1])
    previous=dict(status='SOURCE_SELECTION_FROZEN',target_access=False,target_score_used=False,
        source_summaries={v:dict(score=.90,source_accuracy=.92,worst_rx_accuracy=.88,parameters=382146,conv_linear_macs=9862436)
        for v in ['native','cvcnn','real_cnn','resnet1d','orthogonal_pa','moment_pool','orthogonal_moment','shared_complex']})
    research=combine_research_selection(previous,own)
    assert research['selected_variant']=='residual_fusion'
    assert len(research['candidate_universe'])==6 and len(research['test_variants'])==5
    with pytest.raises(ValueError):combine_research_selection(dict(previous,target_score_used=True),own)
