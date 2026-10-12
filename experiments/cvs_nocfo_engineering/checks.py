"""Native objectives, SAM restoration/reference update and one-layer soft mixing."""
import argparse,contextlib,io,json,math
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import design as d
from .model import build,groups
from .objective import objective,BASE_WEIGHTED_KEYS
from .update import take_update,rng_state,set_rng
from .hidden_mix import plan,capture
from experiments.cvs_equivariant_identity.precision import numerical_context

def sam_reference(device):
    torch.manual_seed(914)
    a=torch.nn.Linear(3,2).to(device).double();b=deepcopy(a);x=torch.randn(5,3,dtype=torch.float64,device=device);y=torch.arange(5,device=device)%2
    c=d.config(next(r for r in d.rows() if r['arm']=='SSL_SAM'))
    oa=torch.optim.AdamW(a.parameters(),lr=.0002,weight_decay=.0001);ob=torch.optim.AdamW(b.parameters(),lr=.0002,weight_decay=.0001)
    def closure():
        z=a(x);loss=F.cross_entropy(z,y);return loss,{'ce':loss},{'ce':loss},{'U_packets':0}
    _,_,_,_,_,_,r=take_update(c,a,oa,closure)
    loss=F.cross_entropy(b(x),y);loss.backward();norm=torch.stack([p.grad.norm() for p in b.parameters()]).norm()
    before=[p.detach().clone() for p in b.parameters()]
    with torch.no_grad():
        for p in b.parameters():p.add_(p.grad,alpha=float(.02/norm))
    ob.zero_grad(set_to_none=True);F.cross_entropy(b(x),y).backward()
    with torch.no_grad():
        for p,value in zip(b.parameters(),before):p.copy_(value)
    ob.step()
    for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa,pb,rtol=1e-12,atol=1e-12)
    assert all(int(v['step'])==1 for v in oa.state.values()) and r['sam_applied'] and r['second_backward_calls']==1
    # A failed perturbed closure must restore the unmodified source model exactly.
    before=[p.detach().clone() for p in a.parameters()];calls=[];rng=rng_state()
    def failing():
        calls.append(1)
        if len(calls)==2:raise RuntimeError('injected second closure failure')
        return closure()
    try:take_update(c,a,oa,failing)
    except RuntimeError as error:assert 'injected' in str(error)
    else:raise AssertionError('Expected injected closure failure')
    assert all(torch.equal(p,value) for p,value in zip(a.parameters(),before)) and all(int(v['step'])==1 for v in oa.state.values())
    assert torch.equal(torch.get_rng_state(),rng['torch'])
    return dict(manual_SAM_AdamW_reference=True,one_permanent_update=True,failed_closure_parameter_restoration=True)

def schedule():
    for row in d.rows():
        c=d.validate(d.config(row));rates=[d.learning_rate_step(c,s) for s in range(1,4441)]
        assert math.isclose(rates[-1],1e-6,abs_tol=1e-14) and min(rates)>0
        if c['engineering']['warm_lr']:
            assert rates[0]==1e-6 and rates[443]==.0006 and rates[444]==.0006
            assert all(a<=b for a,b in zip(rates[:443],rates[1:444])) and all(a>=b for a,b in zip(rates[444:-1],rates[445:]))
        else:assert rates[0]==c['optimizer']['peak_lr']
    return dict(exact_endpoints=True,rows=24,step_count=4440)

def native(device):
    from experiments.cvs_early_geometry.checks import fixture
    from .preflight import cache_checks
    L,_=fixture();first=d.config(d.rows()[0])
    with TemporaryDirectory() as tmp:_,U,bank,_=cache_checks(first,Path(tmp))
    U.update(x=L['x'][0:1].expand(len(U['ids']),-1,-1),rx=L['rx'].repeat(9),day=L['day'].repeat(9))
    li=torch.tensor([i*70 for i in range(90)]+[i*70+1 for i in range(38)]);ui=torch.arange(128);results=[]
    for arm in d.ARMS:
        c=d.config(next(r for r in d.rows() if r['arm']==arm))
        with patch('torch.load',side_effect=AssertionError('Student loaded initialization checkpoint')):model=build(c,device)
        pg,_=groups(model,c);model.train();optimizer=torch.optim.AdamW(pg,weight_decay=.0001)
        assert not hasattr(model,'multilevel')
        from experiments.cvs_e20_cross_style.style import _paths
        _,paths=_paths(model.base)
        calls=[];handle=dict(model.base.named_modules())[paths['time_down']].register_forward_pre_hook(lambda m,args:calls.append(len(args[0])))
        def closure():return objective(c,model,bank,L,li,U,ui,1333,device)
        loss,parts,w,detail,_,norm,r=take_update(c,model,optimizer,closure)
        expected=(3+int(c['engineering']['hidden_mix'])+int(c['engineering']['clean_replay']))*(2 if c['engineering']['sam'] else 1)
        assert len(calls)==expected and torch.isfinite(loss) and math.isfinite(norm),(arm,len(calls),expected,float(loss),norm)
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        assert all(int(v['step'])==1 for v in optimizer.state.values())
        assert torch.equal(loss,sum(w[k] for k in BASE_WEIGHTED_KEYS))
        assert detail['U_packets']==128 and detail['accepted_U']==128 and detail['auxiliary_forwards']==0 and detail['teacher_forwards']==0
        assert r['sam_applied']==c['engineering']['sam']
        if c['engineering']['hidden_mix']:
            info=detail['hidden_mix'];assert info['level']=='time_down' and info['different_TX_verified'] and info['changed_packets']==128 and info['soft_target_sum_error']<1e-6
            assert 'hidden_mix' in objective(c,model,bank,L,li,U,ui,1334,device)[3]
            assert objective(c,model,bank,L,li,U,ui,1334,device)[3]['hidden_mix']['level']=='t2.pool'
        if c['engineering']['clean_replay']:
            assert detail['clean_replay']['identity_transform'] and detail['clean_replay']['packets']==128 and detail['clean_replay']['main_rng_restored']
            assert torch.equal(parts['ce'],parts['style_ce'])
        assert not model._identity_input_policy['explicit_CFO_identity_input']
        handle.remove()
        # This variant's supervised criterion must really use its configured smoothing.
        start=rng_state();plain=objective(c,model,None,L,li,None,None,1,device)[0];set_rng(start)
        manual=F.cross_entropy(model(L['x'][li].to(device)),L['y'][li].to(device),label_smoothing=c['label_smoothing'])
        assert torch.equal(plain,manual)
        results.append(dict(arm=arm,forwards=expected,sam=r,finite_gradients=True,actual_smoothing=c['label_smoothing'],no_auxiliary_module=True))
    return results

def source_entry():
    from . import source,peer_cache,provenance
    from experiments.cvs_e20_budget_fishr.checks import fixture_parent
    from experiments.cvs_early_geometry.checks import fixture
    L,_=fixture();U=dict(x=L['x'],rx=L['rx'],day=L['day'],ids=L['ids']);seen=[]
    class Stop(Exception):pass
    for arm in d.ARMS:
        c=d.config(next(r for r in d.rows() if r['arm']==arm));_,proof=fixture_parent(c)
        from .teacher import expected_proof
        teachers={str(s):expected_proof(s) for s in c['teacher']['seeds']}
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'release_commit.txt').write_text('synthetic-only',encoding='utf-8');c['output_root']=str(root/'source')
            def boundary(config,model,optimizer,closure,audit):
                record=json.loads((root/'source/resolved_config.json').read_text(encoding='utf-8'))
                assert record['engineering']==c['engineering'] and record['fishr_calibration']['status']=='NOT_APPLICABLE'
                assert not hasattr(model,'multilevel') and all(g['lr']==d.learning_rate_step(c,1) for g in optimizer.param_groups)
                raise Stop
            with contextlib.redirect_stdout(io.StringIO()),patch.object(d,'validate',lambda c:c),patch.object(d,'ROOT',root),patch.object(source,'load_source',return_value=(L,U,L,{})),patch.object(peer_cache,'load',return_value={'provenance':teachers}),patch.object(source,'take_update',side_effect=boundary):
                try:source.train(c,'cpu')
                except Stop:seen.append(arm)
                else:raise AssertionError('Actual source entry did not reach optimizer boundary')
    return dict(status='PASS',arms=seen,identity_updates=0,target_access=False)

def run(device):
    torch.set_num_threads(2)
    with numerical_context(d.FULL_FP32_POLICY):
        return dict(status='PASS',device=device,source_only=True,target_access=False,schedule=schedule(),SAM=sam_reference(device),native=native(device),source_entry=source_entry())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(a.device);d.write(a.output,result);print(json.dumps(dict(status=result['status'],device=a.device,arms=len(result['native']))))
