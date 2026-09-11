import copy
import pytest
import torch
from torch import nn
from cvsrffi.game_tracking.solvers import GameSolver
from cvsrffi.game_tracking.state import RNGState


def exact(a,b):
    if torch.is_tensor(a): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): exact(x,y)
    else: assert a==b


class Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(3,4),nn.BatchNorm1d(4),nn.Dropout(.3))
        self.adv_head=nn.Sequential(nn.Linear(4,4),nn.Dropout(.4),nn.Linear(4,2))
    def forward(self,x,adv=True):
        z=self.encoder(x)
        h=self.adv_head(z)
        return z.square().mean()+(h.square().mean() if adv else z.sum()*0)


@pytest.mark.parametrize('adv',[True,False])
def test_head_gradient_scope_exact_state(adv):
    torch.set_num_threads(2); torch.manual_seed(12)
    a=Small(); b=copy.deepcopy(a); x=torch.randn(9,3)
    oa=torch.optim.AdamW(a.parameters(),lr=.003,weight_decay=.1)
    ob=torch.optim.AdamW(b.parameters(),lr=.003,weight_decay=.1)
    sa=GameSolver(a,oa,'head_lookahead' if adv else 'simultaneous',max_grad_norm=.1,telemetry_interval=1)
    sb=GameSolver(b,ob,'head_lookahead',max_grad_norm=.1,b8_impl='head_grad_only',telemetry_interval=1)
    for _ in range(3):
        rng=RNGState.capture(); sa.step(lambda:a(x,adv)); after=RNGState.capture()
        rng.restore(); result=sb.step(lambda:b(x,adv))
        exact(a.state_dict(),b.state_dict()); exact(oa.state_dict(),ob.state_dict())
        assert torch.equal(after.cpu,RNGState.capture().cpu)
        assert result.telemetry['roles']
        if adv:
            for key in ('predictor_clipped','corrector','formal_clipped','virtual_update','formal_update'):
                exact(sa.last_trace[key],sb.last_trace[key])
            for i,p in enumerate(sa.parameters):
                if id(p) in sa.head_ids: exact(sa.last_trace['origin'][i],sb.last_trace['origin'][i])
    if not adv: assert all(p not in ob.state for p in b.adv_head.parameters())


def test_graph_requires_objective_contract():
    m=Small(); solver=GameSolver(m,torch.optim.AdamW(m.parameters()),'head_lookahead',b8_impl='graph_reuse')
    with pytest.raises(NotImplementedError,match='reusable'):
        solver.step(lambda:m(torch.ones(4,3)))


def core_fixture(epoch, device='cpu', synthetic=True, batch_size=12):
    from cvsrffi.game_tracking.config import parse_args
    from cvsrffi.game_tracking.data import build_source
    from cvsrffi.game_tracking.runtime import build_model
    from cvsrffi.game_tracking.step_context import prepare_context, Core90Objective
    from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
    from cvsrffi.game_tracking.legacy.options import _loss_weights
    from cvsrffi.schedule import build_stage_state
    args=parse_args(['--output_dir','unused','--batch_size',str(batch_size),'--num_workers','0']+(['--game_synthetic'] if synthetic else []))
    data=build_source(args); model=build_model(args,len(data.domains),torch.device(device)).train()
    proto=PrototypeMemoryBank(args.num_classes,len(data.domains)); proto._lazy_init(160,torch.device(device),torch.float32)
    # Populate legal synthetic diagnostic prototype state, so pull loss is active.
    proto.class_count.fill_(100); proto.class_proto.copy_(torch.nn.functional.normalize(torch.randn_like(proto.class_proto),dim=1))
    batch=next(iter(data.loader('train',batch_size,seed=42,shuffle=True)))
    ub=next(iter(data.loader('unlabeled',batch_size))) if epoch>=131 else None
    ema=copy.deepcopy(model).eval(); ema.requires_grad_(False)
    ctx=prepare_context(batch,ub,model,ema,args,epoch,1,_loss_weights(args,build_stage_state(epoch,args)),torch.Generator(device=device).manual_seed(55))
    if synthetic:
        # Controlled domains exercise the actual Fishr proxy's >=2 samples/domain gate.
        ctx.domain=torch.arange(batch_size,device=device)//3
    if ctx.strong is not None:
        ctx.base_mask[:]=True; ctx.strong_mask=ctx.base_mask.clone()
    return model,args,proto,ctx,ema


@pytest.mark.parametrize('epoch',[1,80,131])
@pytest.mark.parametrize('implementation',['head_grad_only','graph_reuse'])
@pytest.mark.parametrize('zero_adv',[False,True])
def test_full_core_objective(epoch,implementation,zero_adv):
    from cvsrffi.game_tracking.step_context import Core90Objective
    from cvsrffi.game_tracking.head_lookahead import Core90ReusableGraph
    torch.set_num_threads(2); torch.manual_seed(123)
    a,args,proto,ctx,ema=core_fixture(epoch)
    if zero_adv: ctx.weights['adv']=0
    b=copy.deepcopy(a); pb=copy.deepcopy(proto); cb=copy.deepcopy(ctx)
    oa=torch.optim.AdamW(a.parameters(),lr=.0002,weight_decay=.01)
    ob=torch.optim.AdamW(b.parameters(),lr=.0002,weight_decay=.01)
    sa=GameSolver(a,oa,'simultaneous' if zero_adv else 'head_lookahead',max_grad_norm=5)
    sb=GameSolver(b,ob,'head_lookahead',max_grad_norm=5,b8_impl=implementation)
    fa=Core90Objective(a,args,proto); fb=Core90Objective(b,args,pb)
    original_proto=copy.deepcopy(vars(proto)); original_ema=copy.deepcopy(ema.state_dict())
    rng=RNGState.capture(); sa.step(lambda:fa(ctx)); after=RNGState.capture()
    rng.restore(); sb.step(Core90ReusableGraph(fb,cb) if implementation=='graph_reuse' else lambda:fb(cb))
    if implementation=='head_grad_only' or zero_adv:
        exact(a.state_dict(),b.state_dict()); exact(oa.state_dict(),ob.state_dict())
    else:
        for key,value in a.state_dict().items(): torch.testing.assert_close(value,b.state_dict()[key],atol=1e-6,rtol=1e-5)
        for p,q in zip(a.parameters(),b.parameters()):
            assert (p in oa.state)==(q in ob.state)
            if p in oa.state:
                for key,value in oa.state[p].items(): torch.testing.assert_close(value,ob.state[q][key],atol=1e-6,rtol=1e-5)
    exact(original_proto,vars(proto)); exact(original_proto,vars(pb)); exact(original_ema,ema.state_dict())
    assert torch.equal(after.cpu,RNGState.capture().cpu)
    if ctx.strong is not None: exact(ctx.strong_mask,cb.strong_mask)
    if zero_adv:
        assert all(p not in ob.state for p in b.adv_head.parameters())
    if epoch>=80: assert ctx.origin_terms['sat_cls']>0
    assert ctx.origin_terms['fishr']>0
    assert ctx.origin_terms['proto']>0


def test_graph_existing_moments_and_failure_atomicity():
    from cvsrffi.game_tracking.step_context import Core90Objective
    from cvsrffi.game_tracking.head_lookahead import Core90ReusableGraph
    torch.set_num_threads(2); torch.manual_seed(71)
    a,args,proto,ctx,_=core_fixture(131)
    b=copy.deepcopy(a); pb=copy.deepcopy(proto)
    oa=torch.optim.AdamW(a.parameters(),lr=.0002,weight_decay=.1)
    ob=torch.optim.AdamW(b.parameters(),lr=.0002,weight_decay=.1)
    sa=GameSolver(a,oa,'head_lookahead',max_grad_norm=.1,telemetry_interval=1)
    sb=GameSolver(b,ob,'head_lookahead',max_grad_norm=.1,b8_impl='graph_reuse',telemetry_interval=1)
    for _ in range(3):
        ca=copy.deepcopy(ctx); cb=copy.deepcopy(ctx)
        fa=Core90Objective(a,args,proto); fb=Core90Objective(b,args,pb)
        rng=RNGState.capture(); ra=sa.step(Core90ReusableGraph(fa,ca)); after=RNGState.capture()
        rng.restore(); rb=sb.step(Core90ReusableGraph(fb,cb))
        assert rb.telemetry['gradient_component_diagnostics']=='AVAILABLE'
        assert rb.telemetry['nonadv_gradient_change_norm']<1e-6
        assert rb.telemetry['online_predictor_adv_cosine']==pytest.approx(ra.telemetry['online_predictor_adv_cosine'],abs=1e-6)
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p,q,atol=1e-6,rtol=1e-5)
            if p in oa.state:
                for key,value in oa.state[p].items(): torch.testing.assert_close(value,ob.state[q][key],atol=1e-6,rtol=1e-5)
        assert torch.equal(after.cpu,RNGState.capture().cpu)
    before=copy.deepcopy(b.state_dict()); moments=copy.deepcopy(ob.state_dict()); rng=RNGState.capture()
    class Failure(Core90ReusableGraph):
        def corrector(self,*args):
            super().corrector(*args)
            raise RuntimeError('injected graph failure')
    with pytest.raises(RuntimeError,match='injected'):
        sb.step(Failure(Core90Objective(b,args,pb),copy.deepcopy(ctx)))
    exact(before,b.state_dict()); exact(moments,ob.state_dict()); assert torch.equal(rng.cpu,RNGState.capture().cpu)
