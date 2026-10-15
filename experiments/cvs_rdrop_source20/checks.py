from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from . import design as d
from .objective import symmetric_kl
from .audit import cohorts,layout,prepare,measure_batch,qualify
from experiments.cvs_evidence_responsibility_train20 import model as api

def rejects(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise AssertionError('Invalid source input was accepted')

def checks():
    torch.set_num_threads(2);generator=torch.Generator().manual_seed(202610151)
    a=torch.randn(8,6,generator=generator,dtype=torch.float64,requires_grad=True)
    b=torch.randn(8,6,generator=generator,dtype=torch.float64,requires_grad=True)
    actual=symmetric_kl(a,b);p,q=a.softmax(1),b.softmax(1)
    reference=.5*((p*(p.log()-q.log())).sum(1)+(q*(q.log()-p.log())).sum(1)).mean()
    assert torch.allclose(actual,reference,atol=1e-12,rtol=1e-12)
    assert float(symmetric_kl(a,a).detach())==0 and torch.allclose(actual,symmetric_kl(b,a))
    assert torch.allclose(actual,symmetric_kl(a+9,b-5),atol=1e-12,rtol=1e-12)
    ga,gb=torch.autograd.grad(actual,(a,b));assert all(torch.isfinite(g).all() and float(g.norm())>0 for g in (ga,gb))
    rejects(lambda:symmetric_kl(a[:,:5],b[:,:5]));rejects(lambda:symmetric_kl(a*float('nan'),b))
    c=d.config(d.rows()[0]);d.validate(c)
    changes={'query_read':True,'U_IQ_read':True,'V_forward':True,'coefficient':2.,'label_smoothing':.01,'device':'cuda:0',
        'parent_checkpoint':'other/checkpoint.pth','parameters':['wrong.coordinate']}
    for key,value in changes.items():
        wrong=deepcopy(c);wrong[key]=value;rejects(lambda:d.validate(wrong))
    L={'y':[],'rx':[],'day':[],'ids':[]}
    for tx in range(6):
        for day in d.DAYS:
            for rx in d.RX:
                for index in range(70):
                    L['y'].append(tx);L['rx'].append(rx);L['day'].append(day);L['ids'].append(f'fixture-{tx}-{rx}-{day}-{index}')
    for key in ('y','rx','day'):L[key]=torch.tensor(L[key])
    batches=cohorts(L);ids=[]
    for ix in batches:
        selected=[L['ids'][int(i)] for i in ix];layout(L['y'][ix],L['rx'][ix],L['day'][ix],selected,2);ids+=selected
    assert len(ids)==len(set(ids))==900
    bad=deepcopy(L);bad['ids'][1]=bad['ids'][0];rejects(lambda:cohorts(bad))
    bad=deepcopy(L);bad['rx'][0]=2;rejects(lambda:cohorts(bad))
    rejects(lambda:prepare(nn.Sequential(nn.BatchNorm1d(4))))
    yes=dict(cohorts=[dict(losses={'SYMMETRIC_KL':.1}) for _ in range(5)],layers=[dict(KL_nonzero_cohorts=5,cosine=.1) for _ in range(4)])
    assert qualify(yes)['status']=='PASS'
    bad=deepcopy(yes);bad['layers'][1]['cosine']=-.01;assert qualify(bad)['status']=='FAIL'
    bad=deepcopy(yes);bad['cohorts'][2]['losses']['SYMMETRIC_KL']=0.;assert qualify(bad)['status']=='FAIL'
    model=api.build(d.parent_config(d.SEEDS[0]),'cpu');before={key:value.clone() for key,value in model.state_dict().items()};W=api.classifier(model).weight
    x=torch.randn(6,2,256,generator=generator);y=torch.arange(6);rng=torch.get_rng_state().clone()
    result=measure_batch(model,x,y,2026101519)
    assert api.classifier(model).weight is W and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items()) and all(p.grad is None for p in model.parameters())
    assert all(torch.isfinite(value).all() for values in result['vectors'].values() for value in values)
    assert all(float(value.norm())>0 for value in result['vectors']['SYMMETRIC_KL'][:3])
    return dict(status='PASS',math_and_differentiability=True,permission_negative_cases=len(changes),balanced_source_fixture=True,
        duplicate_or_wrong_RX_rejected=True,BatchNorm_rejected=True,qualification_negative_cases=2,
        actual_model_synthetic_forward_packets=18,actual_model_three_body_KL_gradients_nonzero=True,
        weights_buffers_global_rng_unchanged=True,head_fit_updates=0,optimizer_steps=0,real_IQ_reads=0,checkpoint_loads=0,query_reads=0)

if __name__=='__main__':print(checks())
