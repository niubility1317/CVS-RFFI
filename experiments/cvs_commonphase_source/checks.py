"""Synthetic input/phase/scope negatives; no IQ or model checkpoint read."""
from copy import deepcopy
import torch
from . import design as d
from .core import rotate,select_fit,qualify

def run():
    torch.set_num_threads(2);negative=0
    def rejected(call):
        nonlocal negative
        try:call()
        except ValueError:negative+=1
        else:raise AssertionError('Unregistered input accepted')
    raw=torch.randn(16,2,256,generator=torch.Generator().manual_seed(31));before=raw.clone()
    plus=rotate(raw,.37);minus=rotate(plus,-.37)
    assert torch.equal(raw,before) and torch.allclose(raw,minus,atol=8e-7,rtol=8e-7)
    rel=((plus.square().sum(1).mean(1)-raw.square().sum(1).mean(1)).abs()/raw.square().sum(1).mean(1)).max()
    assert float(rel)<1e-6 and torch.equal(rotate(raw,0.),raw)
    rejected(lambda:rotate(raw,.38));rejected(lambda:rotate(torch.zeros(2,2,512),.37));rejected(lambda:rotate(torch.full((2,2,256),float('nan')),.37))
    for key,value in (('training_updates',1),('fit_RX',[1,3,4,6,8]),('target_access',True),('views',dict(identity=0.,phase_plus=.5,phase_minus=-.5))):
        c=deepcopy(d.config());c[key]=value;rejected(lambda:d.validate(c))
    rx=[];tx=[];days=[]
    for r in (1,3,4,6,8):
        for dd in (1,2,3):
            for y in range(6):rx += [r]*70;days += [dd]*70;tx += [y]*70
    L=dict(x=torch.ones(6300,2,256),y=torch.tensor(tx),rx=torch.tensor(rx),day=torch.tensor(days),ids=['L'+str(i) for i in range(6300)])
    carrier=dict(role_ids=dict(L_s=L['ids'],U_s=['U'],V=['V']));trained=dict(role_ids=dict(L_s=[v for v,r in zip(L['ids'],rx) if r!=1]))
    # Held RX1's IQ cannot affect selection or finite checks.
    L['x'][L['rx']==1]=float('nan');fit=select_fit(L,carrier,trained)
    assert fit['x'].shape==(5040,2,256) and bool(torch.isfinite(fit['x']).all())
    bad=deepcopy(L);bad['rx'][1260]=2;rejected(lambda:select_fit(bad,carrier,trained))
    bad=deepcopy(L);bad['y'][1260]=-1;rejected(lambda:select_fit(bad,carrier,trained))
    bad=deepcopy(L);bad['ids'][1260]='U';rejected(lambda:select_fit(bad,carrier,trained))
    scores={view:dict(accuracy=.99 if view=='identity' else .95,by_RX={str(r):dict(accuracy=.99 if view=='identity' else .95) for r in d.FIT}) for view in d.VIEWS}
    assert qualify(scores)['decision']=='PASS'
    bad=deepcopy(scores);bad['phase_minus']['accuracy']=.98;assert qualify(bad)['decision']=='FAIL'
    bad=deepcopy(scores)
    for r in (3,4):bad['phase_plus']['by_RX'][str(r)]['accuracy']=.99
    assert qualify(bad)['decision']=='FAIL'
    value=dict(status='PASS',negative_cases=negative,unit_phase_inverse=True,unit_phase_relative_power_error=float(rel),held_RX1_nan_excluded_before_IQ_checks=True,source_IQ_reads=0,checkpoint_loads=0,model_forwards=0,optimizer_updates=0)
    d.write(d.ROOT/d.LOCAL/'local_validation.json',value);print(value);return value

if __name__=='__main__':run()
