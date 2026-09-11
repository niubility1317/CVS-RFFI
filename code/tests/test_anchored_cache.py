import sys
from pathlib import Path
from dataclasses import replace
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvsrffi.anchored_cache import CacheIdentity, FeatureRows, IdentityAnchor, save_source_cache, load_source_cache, paired_scene_schedule
from test_evidence_integration import model_args, source_batch
from post_stage_common import build_baseline_model


def toy_rows(receivers=(1, 3, 4), dimension=8, per_group=4):
    torch.manual_seed(44)
    weights = torch.randn(3, dimension)
    values=[]; labels=[]; ids=[]; views=[]; rx=[]
    for y in range(3):
        for r in receivers:
            for i in range(per_group):
                h = weights[y] + .4 * torch.randn(dimension)
                for view in ('clean', 'leo_clear_weak'):
                    values.append(h + (0 if view == 'clean' else .2 * torch.randn(dimension)))
                    labels.append(y); ids.append(f'{y}:{r}:1:{i}'); views.append(view); rx.append(r)
    h=torch.stack(values)
    identity=CacheIdentity('a'*64, 'b'*64, 'test-v1', {'version':'paired_leo_v1','seed':392005},392005,'L_s')
    return FeatureRows(h, h @ weights.T, torch.rand(len(h),3), torch.ones_like(h,dtype=torch.bool),
                       torch.tensor(labels),tuple(ids),tuple(views),torch.tensor(rx),torch.ones(len(h),dtype=torch.long),
                       torch.ones(len(h),dtype=torch.bool),identity)


def test_role_unique_keys_and_cache_binding(tmp_path):
    rows=toy_rows(); rows.validate()
    save_source_cache(rows,tmp_path/'cache', shard_rows=17)
    loaded=load_source_cache(tmp_path/'cache',rows.identity)
    torch.testing.assert_close(loaded.h,rows.h)
    assert loaded.physical_ids==rows.physical_ids
    with pytest.raises(ValueError,match='identity'):
        load_source_cache(tmp_path/'cache',replace(rows.identity,checkpoint_sha256='c'*64))
    with pytest.raises(ValueError,match='role'):
        replace(rows,identity=replace(rows.identity,role='target')).validate()
    with pytest.raises(ValueError,match='duplicate'):
        replace(rows,view_ids=tuple('clean' for _ in rows.view_ids)).validate()
    with pytest.raises(FileExistsError): save_source_cache(rows,tmp_path/'cache')
    classifier=torch.nn.Linear(rows.h.shape[1],3)
    classifier(loaded.h).sum().backward()
    assert classifier.weight.grad.abs().sum()>0


def test_inference_tensors_are_normal_after_reload(tmp_path):
    rows=toy_rows()
    with torch.inference_mode(): rows.h=rows.h.clone()
    save_source_cache(rows,tmp_path/'cache')
    loaded=load_source_cache(tmp_path/'cache',rows.identity)
    assert not torch.is_inference(loaded.h)


def test_scene_assignment_balances_full_contract_and_is_order_independent():
    physical=[]; labels=[]; rx=[]; days=[]
    for y in range(6):
        for r in (1,3,4,6,8):
            for day in (1,2,3):
                for i in range(70):
                    physical.append(f'{y}:{r}:{day}:{i}'); labels.append(y);rx.append(r);days.append(day)
    schedule=paired_scene_schedule(physical,labels,rx,days,392005)
    assert sorted(list(schedule.values()).count(x) for x in set(schedule.values()))==[2100]*3
    reverse=paired_scene_schedule(physical[::-1],labels[::-1],rx[::-1],days[::-1],392005)
    assert schedule==reverse


def test_real_architecture_identity_only_equals_h0_and_skips_domain():
    torch.set_num_threads(2);torch.manual_seed(4)
    model=build_baseline_model(model_args(),torch.device('cpu')).eval()
    x,_=source_batch();x=x[:3]
    anchor=IdentityAnchor(model)
    calls=[];hook=model.dom_backbone.register_forward_hook(lambda *args:calls.append(1))
    before={k:v.clone() for k,v in model.state_dict().items()}
    h,s0,valid=anchor.extract(x)
    assert not calls
    with torch.no_grad(): ref=model(x,y_tx=None)
    torch.testing.assert_close(s0,ref,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(s0,model.id_backbone.cls_head.head(h,labels=None))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert valid.dtype==torch.bool
    hook.remove()
