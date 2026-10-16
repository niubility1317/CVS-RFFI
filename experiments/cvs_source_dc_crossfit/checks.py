import copy
import numpy as np
from . import design as d
from .core import cells,effect,prediction,analyze
def reject(call):
    try:call()
    except ValueError:return
    raise AssertionError('Invalid source statistic accepted')
def run():
    rng=np.random.default_rng(20261016901);profile=np.asarray([[-.04,.01],[-.01,-.03],[.01,.02],[.04,0.]])
    a=rng.normal(0,.3,size=(6,1,3,2))+profile[None,:,None,:]
    good,estimated=analyze(a,.005)
    assert good['status']=='PASS' and good['positive_folds']==18 and good['aggregate_risk_reduction']>1.-1e-12
    assert np.allclose(estimated,profile-profile.mean(0))
    e=effect(a);changed=a.copy();changed[2,:,:,:]+=rng.normal(size=(4,3,2))*1000;changed[:,:,1,:]+=rng.normal(size=(6,4,2))*1000
    assert np.array_equal(prediction(e,2,1),prediction(effect(changed),2,1))
    entangled=a-a.mean(axis=1,keepdims=True)
    entangled[:3]*=-1
    bad,_=analyze(entangled,.005);assert bad['status']=='FAIL' and bad['aggregate_risk_reduction']<0 and bad['positive_folds']==0
    small,_=analyze(a*.001,.005);assert small['status']=='FAIL' and not small['checks']['outside_old_ordinary'] and small['checks']['risk']
    zero,_=analyze(np.zeros_like(a),.005);assert zero['status']=='FAIL' and zero['aggregate_risk_reduction'] is None
    z=[];tx=[];rx=[];day=[]
    for t in d.TX:
        for ri,r in enumerate(d.FIT):
            for di,date in enumerate(d.DAYS):
                for i in range(70):z.append(a[t,ri,di]);tx.append(t);rx.append(r);day.append(date)
    z,tx,rx,day=map(np.asarray,(z,tx,rx,day));assert np.allclose(cells(z,tx,rx,day),a)
    negatives=0
    for zz,tt,rr,dd in ((z[:-1],tx[:-1],rx[:-1],day[:-1]),(z,tx-1,rx,day),(z,tx,np.where(rx==3,1,rx),day),(z,tx,np.where(rx==3,2,rx),day),(z,tx,rx,day-1),(z*np.nan,tx,rx,day)):
        reject(lambda:cells(zz,tt,rr,dd));negatives+=1
    for bad_input in (a[:5],a*np.nan):reject(lambda:effect(bad_input));negatives+=1
    for bound in (0.,-1.,float('nan')):reject(lambda:analyze(a,bound));negatives+=1
    for c in (dict(d.config(d.rows()[0]),source_RX=[1,3,4,6]),dict(d.config(d.rows()[0]),target_access=True),dict(d.config(d.rows()[0]),checkpoint_loads=1),dict(d.config(d.rows()[0]),primary=dict(aggregate_risk_reduction_min=0.))):
        reject(lambda:d.validate(c));negatives+=1
    return dict(status='PASS',synthetic_descriptor_packets=5040,synthetic_positive_18folds=True,entangled_and_zero_effect_FAIL=True,
        stable_but_already_covered_FAIL=True,held_TX_AND_day_prediction_isolation=True,protocol_negatives=negatives,
        real_IQ_packets=0,checkpoint_loads=0,model_forwards=0,optimizer_updates=0,synthetic_fixtures_not_results=True)
