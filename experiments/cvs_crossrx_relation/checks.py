"""Mathematical all-class/PID tests plus actual native relation gradients."""
import argparse,tempfile,contextlib,io,math
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import design as d
from .core import support,relation_logits,reference
from .model import build,groups
from .objective import objective,BASE_WEIGHTED_KEYS
from .update import take_update,rng_state

def algebra(device):
    gen=torch.Generator().manual_seed(231);query=torch.randn(8,10,generator=gen).to(device);refs=torch.randn(30,10,generator=gen).to(device);rx=torch.tensor([1]*4+[3]*4,device=device);y=torch.arange(8,device=device)%6;results=[]
    for mode in ('merged_RX','separate_RX'):
        logits=relation_logits(query,refs,rx,mode);changed=refs.reshape(6,5,10).clone();changed[:,0]=1000*torch.randn(6,10,generator=gen).to(device)
        torch.testing.assert_close(logits[:4],relation_logits(query,changed.reshape(30,10),rx,mode)[:4],rtol=0,atol=0)
        permutation=torch.tensor([4,2,0,5,1,3],device=device);inverse=permutation.argsort();permuted=refs.reshape(6,5,10)[permutation].reshape(30,10)
        a=F.cross_entropy(logits,y);b=F.cross_entropy(relation_logits(query,permuted,rx,mode),inverse[y]);torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
        if mode=='separate_RX':
            grid=refs.reshape(6,5,10);manual=[]
            for q,r in zip(query,rx.tolist()):
                other=grid[:,[j for j,v in enumerate((1,3,4,6,8)) if v!=r]];center=other.mean((0,1));z=F.normalize(q-center,dim=0);p=F.normalize(other-center,dim=-1)
                manual.append(torch.logsumexp((p*z).sum(-1)/.1,1)-math.log(4))
            torch.testing.assert_close(logits,torch.stack(manual),rtol=1e-6,atol=1e-6)
        results.append(dict(mode=mode,own_RX_reference_excluded=True,TX_renaming_invariant=True,all6_compete=True))
    from experiments.cvs_early_geometry.checks import fixture
    L,_=fixture();li=torch.tensor([i*70 for i in range(90)]+[i*70+1 for i in range(38)])
    for step in (448,452,456):
        si,proof=support(L,li,2026092701,step);assert len(si)==30 and not set(proof['support_IDs'])&{L['ids'][i] for i in li.tolist()}
        again,_=support(L,li,2026092701,step);assert torch.equal(si,again)
    try:support(dict(L,role='U_s'),li,1,448)
    except ValueError:pass
    else:raise AssertionError('Unlabeled role accepted for relation')
    return dict(modes=results,independent_support_budget30=True,deterministic_private_sampling=True,U_role_rejected=True)

def native(device):
    from experiments.cvs_early_geometry.checks import fixture
    from experiments.cvs_nocfo_engineering.preflight import cache_checks
    L,_=fixture();c=d.config(d.rows()[0])
    with tempfile.TemporaryDirectory() as tmp:_,U,bank,_=cache_checks(c,Path(tmp))
    U.update(x=L['x'][0:1].expand(len(U['ids']),-1,-1),rx=L['rx'].repeat(9),day=L['day'].repeat(9))
    li=torch.tensor([i*70 for i in range(90)]+[i*70+1 for i in range(38)]);ui=torch.arange(128);results=[]
    for arm in d.ARMS:
        c=d.config(next(r for r in d.rows() if r['arm']==arm))
        with patch('torch.load',side_effect=AssertionError('Initial weights loaded')):model=build(c,device)
        pg,_=groups(model,c);model.train();optimizer=torch.optim.AdamW(pg,weight_decay=.0001)
        def closure():return objective(c,model,bank,L,li,U,ui,448,device)
        # The relation objective must reach actual identity parameters without auxiliary heads.
        start=rng_state();loss,parts,w,detail=closure();grads=torch.autograd.grad(w['relation'],[p for p in model.parameters() if p.requires_grad],allow_unused=True)
        norm=sum(float(g.square().sum()) for g in grads if g is not None)**.5
        assert math.isfinite(norm) and norm>0 and torch.equal(loss,sum(w[k] for k in BASE_WEIGHTED_KEYS))
        assert detail['relation']['L_packets']==30 and detail['relation']['support_query_PID_disjoint'] and not detail['relation']['prototype_grad']
        if c['relation']['mode']!='extra_CE':assert set(detail['relation']['levels'])==set(c['relation']['levels'])
        end=rng_state();main_rng_changed=not torch.equal(start['torch'],end['torch']) or any(not torch.equal(a,b) for a,b in zip(start['cuda'],end['cuda']))
        updated=take_update(c,model,optimizer,closure);assert torch.isfinite(updated[0]) and all(int(v['step'])==1 for v in optimizer.state.values())
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        assert not model._identity_input_policy['explicit_CFO_identity_input'] and not hasattr(model,'multilevel')
        before=rng_state();modes=[m.training for m in model.modules()];ref=reference(model,L['x'][li[:30]].to(device),['time_down','t2.pool','id_feat_joint']);after=rng_state()
        assert torch.equal(before['torch'],after['torch']) and all(torch.equal(a,b) for a,b in zip(before['cuda'],after['cuda'])) and modes==[m.training for m in model.modules()] and all(not v.requires_grad for v in ref.values())
        # Inactive steps retain exactly the registered plain CE+SSL objective.
        inactive=objective(c,model,bank,L,li,U,ui,449,device);assert not inactive[3]['relation']['active'] and float(inactive[2]['relation'].detach())==0
        results.append(dict(arm=arm,relation_gradient_norm=norm,all_gradients_finite=True,one_permanent_update=True,support_L=30,reference_state_rng_modes_restored=True,no_auxiliary_heads=True,main_rng_changed_measured=main_rng_changed))
    return results

def source_entry():
    from . import source,peer_cache
    from .teacher import expected_proof
    from experiments.cvs_early_geometry.checks import fixture
    L,_=fixture();U=dict(x=L['x'],rx=L['rx'],day=L['day'],ids=L['ids']);seen=[]
    class Stop(Exception):pass
    for arm in d.ARMS:
        c=d.config(next(r for r in d.rows() if r['arm']==arm));teachers={str(s):expected_proof(s) for s in c['teacher']['seeds']}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'release_commit.txt').write_text('synthetic-only');c['output_root']=str(root/'source')
            def boundary(config,model,optimizer,closure,audit):
                value=d.read(root/'source/resolved_config.json');assert value['execution']['relation']==c['relation'] and not value['execution']['explicit_CFO_identity_input'];raise Stop
            with contextlib.redirect_stdout(io.StringIO()),patch.object(d,'validate',lambda v:v),patch.object(d,'ROOT',root),patch.object(source,'load_source',return_value=(L,U,L,{})),patch.object(peer_cache,'load',return_value={'provenance':teachers}),patch.object(source,'take_update',side_effect=boundary):
                try:source.train(c,'cpu')
                except Stop:seen.append(arm)
                else:raise AssertionError('Source optimizer boundary not reached')
    return dict(status='PASS',arms=seen,identity_updates=0)

def run(device):
    from experiments.cvs_equivariant_identity.precision import numerical_context
    torch.set_num_threads(2)
    with numerical_context(d.FULL_FP32_POLICY):return dict(status='PASS',device=device,algebra=algebra(device),native=native(device),source_entry=source_entry(),target_access=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print({'status':v['status'],'arms':len(v['native']),'device':a.device})
