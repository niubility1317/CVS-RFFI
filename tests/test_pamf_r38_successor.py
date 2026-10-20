import copy
import json
import pytest
import torch
from torch import nn
from experiments.pamf_iq_joint_compact.models import JointCompactRestorer
from experiments.pamf_r38_successor import design as d,models as m,runtime as r,objective as o
from experiments.pamf_uncertainty_restore import design as ancestor

torch.set_num_threads(2)


class Reader(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Linear(512,160);self.head=nn.Linear(160,6)
    def features(self,x):return self.encoder(x.flatten(1)).tanh()
    def classify_features(self,x):return self.head(x)


def parent_model():
    value=JointCompactRestorer('dense_half').eval().requires_grad_(False)
    with torch.no_grad():
        for core in value.prior:core.out.weight.normal_(std=.001)
        value.frozen.coarse.temporal.out.weight.normal_(std=.001)
        value.frozen.estimator.core.out.weight.normal_(std=.001)
        value.frozen.estimator.global_head.weight.normal_(std=.001)
    return value


def lineage():
    return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,action_run=ancestor.ACTION_RUN,action_commit=ancestor.ACTION_COMMIT,
        data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)


@pytest.mark.parametrize('arm',d.ARMS)
def test_every_successor_component_updates_with_detached_parent_clean_and_frozen_reader(tmp_path,arm):
    torch.manual_seed(17);parent=parent_model();before=copy.deepcopy(parent.state_dict());model=m.compact_from_parent(parent,arm)
    model.initialization_report['seconds']=0.;r.validate_aliases(model)
    assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]
    assert sum(p.numel() for p in model.frozen.parameters())==d.CONTEXT_PARAMETERS[arm]
    x=(torch.randn(2,2,256)*.1).requires_grad_(True);received=x.detach()+torch.randn_like(x)*.005
    with torch.no_grad():desired=parent.stage_outputs(received)[0]
    if d.LEVEL[arm]=='keep':
        assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
        with torch.no_grad():assert torch.equal(model(received),parent(received))
    reader=Reader().eval().requires_grad_(False) if d.TASK[arm] else None
    reader_before=copy.deepcopy(reader.state_dict()) if reader is not None else None
    original=copy.deepcopy(model.state_dict());output=model.stage_outputs(received)[0]
    loss,terms=o.objective(output,desired,x,reader,arm)
    assert set(terms)==({'iq_nmse','teacher_stage_nmse','task_KL','task_feature_distance'} if d.TASK[arm] else {'iq_nmse','teacher_stage_nmse'})
    assert torch.allclose(loss,sum(terms.values()))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);loss.mean().backward()
    for group in (model.frozen.coarse,model.frozen.estimator,*model.prior):
        gradients=[p.grad for p in group.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(v).all() for v in gradients) and any(v.abs().sum()>0 for v in gradients)
    assert x.grad is None and all(p.grad is None for p in parent.parameters())
    optimizer.step()
    for prefix in ('frozen.coarse.','frozen.estimator.','prior.'):
        assert any(not torch.equal(value,original[key]) for key,value in model.state_dict().items() if key.startswith(prefix))
    assert all(torch.equal(v,before[k]) for k,v in parent.state_dict().items())
    if reader is not None:assert all(p.grad is None for p in reader.parameters()) and all(torch.equal(v,reader_before[k]) for k,v in reader.state_dict().items())
    model.eval();path=tmp_path/'final.pt';r.save_checkpoint(path,model,None,10,2470,lineage());fresh,value=r.load_checkpoint(path,'cpu')
    assert value['optimizer'] is None and value['cumulative_component_epochs']==50 and value['training_F_access']==d.TASK[arm]
    with torch.no_grad():assert torch.equal(model(received),fresh(received))
    with pytest.raises(FileExistsError):r.save_checkpoint(path,model,None,10,2470,lineage())


@pytest.mark.parametrize('bad',['unknown','shape','iq_reader','task_no_reader'])
def test_objective_permissions_reject_before_classifier_access(monkeypatch,bad):
    monkeypatch.setattr(o,'frozen_task_fidelity',lambda *a:pytest.fail('No classifier access'))
    clean=torch.ones(2,2,256);stages=[clean]*3;arm=d.ARMS[0];reader=None
    if bad=='unknown':arm='unknown'
    elif bad=='shape':stages=stages[:2]
    elif bad=='iq_reader':reader=object()
    else:arm='keep_task'
    with pytest.raises(ValueError):o.objective(stages,[clean]*3,clean,reader,arm)


def test_registered_ten_epochs_keep_total_component_budget_and_json_stable_plan():
    cfg=d.configuration();assert json.loads(json.dumps(cfg))==cfg
    assert d.EPOCHS==10 and d.UPDATES==2470 and cfg['cumulative_component_epochs']==50
    assert cfg['cumulative_serial_training_passes']==90 and cfg['cumulative_component_epoch_sum']==150
    assert cfg['L_exposures']==63000 and cfg['U_exposures']==567000 and len(d.ARMS)==6
    assert set(d.GPUS.values())==set(range(2,8)) and len(set(d.PARAMETERS.values()))==3
    assert d.learning_rate(0)==1e-4/247 and d.learning_rate(246)==1e-4 and d.learning_rate(2469)==pytest.approx(1e-5)
    with pytest.raises(ValueError):d.learning_rate(2470)


@pytest.mark.parametrize('field,bad',[('parent_R38_row','dense_three_quarter'),('parent_R38_commit','unknown'),('parent_training_TX_F_target_reads',1),
    ('training_F_access',True),('target_training',True),('cumulative_component_epochs',60),('updates',2471)])
def test_wrong_lineage_task_permission_or_budget_rejects(field,bad):
    model=m.compact_from_parent(parent_model(),'keep_iq');model.initialization_report['seconds']=0.
    value=r.metadata('keep_iq',10,2470,lineage(),model.initialization_report);r.validate_payload(value);value[field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)


def test_conflicting_shared_aliases_reject_and_unknown_arm_does_not_open_parent(monkeypatch):
    model=m.CompactSuccessor('half_iq');state=copy.deepcopy(model.state_dict());key=next(k for k in state if k.startswith('prior.1.') and k.endswith('net.0.weight'))
    state[key]=state[key]+1
    with pytest.raises(ValueError,match='Conflicting'):r.validate_aliases(model,state)
    monkeypatch.setattr(r,'validate_parent_root',lambda *a:pytest.fail('No parent IO'))
    with pytest.raises(ValueError):r.initialize('unknown','/runs','cpu')
