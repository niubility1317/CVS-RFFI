from copy import deepcopy
from itertools import product
from pathlib import Path
import argparse,json
import torch
from torch.nn import functional as F
from .geometry import geometry,clone_step
from .run import batch,modules,tree_equal
from . import design as d

def core(device):
    p=torch.nn.Parameter(torch.tensor([.2,-.3],device=device));z=torch.nn.Parameter(torch.tensor([.1+.3j,-.4+.2j],device=device));unused=torch.nn.Parameter(torch.tensor([.6],device=device))
    named=(('real',p),('complex',z),('unused',unused));groups=[dict(params=[p,z,unused],lr=.003)]
    opt=torch.optim.AdamW(groups,betas=(.8,.95),eps=1e-7,weight_decay=.02)
    for _ in range(2):
        p.grad=torch.tensor([.07,-.02],device=device);z.grad=torch.tensor([.05+.04j,-.02+.03j],device=device);unused.grad=torch.ones_like(unused);opt.step()
    for _,v in named:v.grad=None
    state=deepcopy(opt.state_dict());original=deepcopy(state);weights=[v.detach().clone() for _,v in named]
    gradients=(torch.tensor([.03,0.],device=device),torch.tensor([.02-.05j,.04+0j],device=device),None)
    for mode in d.MODES:
        delta,info=clone_step(named,[dict(params=[p,z,unused],lr=.003)],state,gradients,mode,.02)
        beta1,beta2=(.8,.95) if mode=='saved_final' else (.9,.999)
        eps=1e-7 if mode=='saved_final' else 1e-8
        for i,(old,g) in enumerate(zip(weights,gradients)):
            if g is None:assert not bool(delta[i].any());continue
            real=lambda x:torch.view_as_real(x) if x.is_complex() else x
            gr=real(g);before=real(old)
            previous=state['state'][i] if mode=='saved_final' else None
            step=float(previous['step'])+1 if previous else 1
            m=beta1*real(previous['exp_avg'])+(1-beta1)*gr if previous else (1-beta1)*gr
            v=beta2*real(previous['exp_avg_sq'])+(1-beta2)*gr.square() if previous else (1-beta2)*gr.square()
            expected=before*(1-.003*.02)-.003*(m/(1-beta1**step))/((v/(1-beta2**step)).sqrt()+eps)
            torch.testing.assert_close(real(old+delta[i]),expected,atol=6e-8,rtol=2e-6)
        assert info['state_step_max']==(3. if mode=='saved_final' else 1.)
    assert tree_equal(state,original) and all(torch.equal(v,old) and v.grad is None for (_,v),old in zip(named,weights))
    orthogonal=geometry((torch.tensor([1+1j],device=device),),(torch.tensor([1-1j],device=device),),[0]);assert orthogonal['cosine']==0.
    assert geometry((None,),(None,),[0])['cosine'] is None
    return dict(status='PASS',device=device,analytic_real_complex_AdamW=True,gradient_none_skips_update=True,saved_state_unchanged=True,complex_real_inner_product=True)

def integration(device):
    torch.set_num_threads(2);records=[]
    cells=list(product(range(6),(1,3,4,6,8),(1,2,3),range(4)));n=len(cells)
    L=dict(x=torch.randn(n,2,256,generator=torch.Generator().manual_seed(422)),y=torch.tensor([x[0] for x in cells]),rx=torch.tensor([x[1] for x in cells]),day=torch.tensor([x[2] for x in cells]),ids=list(range(n)))
    U=dict(x=torch.randn(384,2,256,generator=torch.Generator().manual_seed(423)),ids=list(range(10000,10384)),role='U_s')
    for entry in d.config(d.rows()[0])['models']:
        c=entry['config'];m=modules(entry['family']);model=m['model'].build(c,device).train()
        model.multilevel.fixed_coefficients={'time_down':.1,'t2.pool':.1}
        pg,_=m['model'].groups(model,c);opt=torch.optim.AdamW(pg,weight_decay=c['optimizer']['weight_decay'])
        for _ in range(2):
            opt.zero_grad(set_to_none=True);F.cross_entropy(model(L['x'][:32].to(device)),L['y'][:32].to(device)).backward();opt.step()
        opt.zero_grad(set_to_none=True);saved=deepcopy(opt.state_dict())
        # Reproduce the actual load path: a freshly built checkpoint model has
        # not yet passed through the optimizer group's identity-only freezing.
        fresh=m['model'].build(c,device).train();fresh.load_state_dict(model.state_dict(),strict=True)
        fresh.multilevel.fixed_coefficients=dict(model.multilevel.fixed_coefficients)
        model=fresh
        bank=dict(q=torch.randn(384,6,generator=torch.Generator().manual_seed(424)).softmax(1),ids=U['ids'],teacher_seeds=c['teacher']['seeds'],peer_disagreement=torch.zeros(384,dtype=torch.bool),peer_JS=torch.zeros(384))
        result=batch(model,saved,m,c,L,U,bank,0,device)
        assert len(result['cases'])==8 and result['model_forwards']==12 and result['persistent_identity_updates']==0
        assert set((v['mode'],v['objective']) for v in result['cases'])==set(product(d.MODES,d.OBJECTIVES))
        records.append(dict(model=entry['key'],result=result))
    return dict(status='PASS',device=device,models=6,batches=6,temporary_optimizer_steps=48,fresh_model_optimizer_setup_regression=True,source_only_synthetic=True,target_access=False,records=records)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--integration',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    v=integration(a.device) if a.integration else core(a.device);d.write(a.output,v);print(json.dumps({k:v[k] for k in v if k!='records'}))
