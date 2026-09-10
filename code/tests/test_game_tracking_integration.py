import sys
from pathlib import Path
from copy import deepcopy
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.game_tracking.config import parse_args
from cvsrffi.game_tracking.data import build_source,audit_indices
from cvsrffi.game_tracking.runtime import build_model,audit,calibrate
from cvsrffi.game_tracking.source_audit import SourceAuditor,AuditConfig
from cvsrffi.game_tracking.step_context import prepare_context,Core90Objective,satellite_stage
from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
from cvsrffi.game_tracking.legacy.options import _loss_weights
from cvsrffi.game_tracking.solvers import GameSolver
from cvsrffi.game_tracking.state import RNGState
from cvsrffi.schedule import build_stage_state

def setup():
    torch.set_num_threads(2)
    a=parse_args(['--output_dir','unused','--game_synthetic','--batch_size','18','--num_workers','0','--game_probe_steps','2'])
    data=build_source(a)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(a,len(data.domains),device).train()
    proto=PrototypeMemoryBank(a.num_classes,len(data.domains))
    proto._lazy_init(160,device,torch.float32)
    return a,data,model,proto,device

def test_hidden_metadata_and_capture_grouping():
    a,data,_,_,_=setup()
    _,y,_,meta=data.unlabeled[0]
    assert y==-1 and not any('tx' in k.lower() for k in meta)
    idx=audit_indices(data)
    for left,right in [('domain_fit','domain_monitor'),('capability_fit','capability_monitor')]:
        fit={data.train[i][3]['capture_group'] for i in idx[left]}
        mon={data.train[i][3]['capture_group'] for i in idx[right]}
        assert not fit&mon
    assert satellite_stage(1)==(['leo_clear_weak'],.3)
    assert satellite_stage(41)[1]==.6 and satellite_stage(91)[1]==.8

@pytest.mark.parametrize('epoch,mode',[(1,'simultaneous'),(41,'extragradient'),(80,'heun'),(131,'extragradient'),(200,'optimistic')])
def test_real_core90_all_stage_terms_and_transactions(epoch,mode):
    a,data,m,proto,device=setup()
    batch=next(iter(data.loader('train',18,seed=42,shuffle=True)))
    ub=next(iter(data.loader('unlabeled',18))) if epoch>130 else None
    ema=deepcopy(m).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    weights=_loss_weights(a,build_stage_state(epoch,a))
    ctx=prepare_context(batch,ub,m,ema,a,epoch,1,weights,torch.Generator(device=device).manual_seed(55))
    objective=Core90Objective(m,a,proto)
    before=deepcopy(proto.__dict__)
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr)
    solver=GameSolver(m,opt,mode,max_grad_norm=5)
    result=solver.step(lambda:objective(ctx))
    assert result.accepted
    assert ctx.origin_features.shape==(18,160)
    assert all(torch.isfinite(torch.tensor(v)) for v in ctx.origin_terms.values())
    if epoch>=80: assert ctx.origin_terms['sat_cls']>0
    else: assert ctx.origin_terms['sat_cls']==0
    if epoch>130: assert ctx.strong_mask is not None and 'unlabeled_ce' in ctx.origin_terms
    for k,v in before.items():
        if torch.is_tensor(v): assert torch.equal(v,vars(proto)[k])
    assert all(int(s['step'])==1 for s in opt.state.values())

def test_actual_audit_preserves_training_state_and_maps_gradient_fields():
    a,data,m,proto,device=setup()
    before=deepcopy(m.state_dict())
    rng=RNGState.capture()
    metrics,_=audit(m,data,audit_indices(data),SourceAuditor(AuditConfig(steps=2)),a,0,0)
    assert metrics['label_kind']=='domain' and 'S_domain' in metrics
    assert metrics['direction_imbalance'] is not None
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    assert m.training
    current=RNGState.capture()
    assert torch.equal(rng.cpu,current.cpu)
    ctrl,cur=calibrate([dict(metrics,step=i) for i in (0,250,500)],a)
    assert ctrl is not None and cur is not None

def test_checkpoint_initialization_permission():
    with pytest.raises(ValueError,match='PROVENANCE'):
        parse_args(['--output_dir','unused','--baseline_ckpt','old.pth'])
    with pytest.raises(ValueError,match='overlap'):
        parse_args(['--output_dir','unused','--wisig_test_rxs','1'])
