import numpy as np
from . import design as d
from .core import descriptor,cells,effect,prediction,analyze
def reject(f):
    try:f()
    except ValueError:return
    raise AssertionError('Invalid statistic accepted')
def run():
    rng=np.random.default_rng(20261016911)
    n=np.arange(256);x=np.exp(2j*np.pi*13*n/256)[None,:]
    def iq(z):return np.stack((z.real,z.imag),axis=1)
    b=.03+.02j;y=x+b*x.conj();q=descriptor(iq(y))
    assert np.allclose(q,2*abs(b)/(1+abs(b)**2),atol=1e-12)
    assert np.allclose(descriptor(iq(y)),descriptor(iq(1.7*np.exp(.9j)*y+.01+.02j)),atol=1e-12)
    assert np.allclose((y-b*y.conj())/(1-abs(b)**2),x,atol=1e-12)
    profile=np.array([-.03,-.01,.01,.03]);a=.2+rng.uniform(0,.2,size=(6,1,3))+profile[None,:,None]
    s,p=analyze(a);assert s['status']=='PASS' and s['positive_folds']==18 and s['aggregate_risk_reduction']>1-1e-12
    e=effect(a);changed=a.copy();changed[2,:,:]+=rng.normal(size=(4,3));changed[:,:,1]+=rng.normal(size=(6,4))
    assert np.array_equal(prediction(e,2,1),prediction(effect(changed),2,1))
    entangled=effect(a);entangled[:3]*=-1;bad,_=analyze(entangled);assert bad['status']=='FAIL' and bad['positive_folds']==0
    small,_=analyze(a*.01);assert small['status']=='FAIL' and small['checks']['risk'] and not small['checks']['ellipticity_scale']
    zero,_=analyze(np.zeros_like(a));assert zero['status']=='FAIL' and zero['aggregate_risk_reduction'] is None
    z=[];tx=[];rx=[];day=[]
    for t in d.TX:
        for ri,r in enumerate(d.FIT):
            for di,date in enumerate(d.DAYS):
                for i in range(70):z.append(a[t,ri,di]);tx.append(t);rx.append(r);day.append(date)
    z,tx,rx,day=map(np.asarray,(z,tx,rx,day));assert np.allclose(cells(z,tx,rx,day),a)
    count=0
    for xx in (np.zeros((1,2,256)),np.ones((1,2,255)),np.full((1,2,256),np.nan)):
        reject(lambda:descriptor(xx));count+=1
    for zz,tt,rr,dd in ((z[:-1],tx[:-1],rx[:-1],day[:-1]),(z,tx-1,rx,day),(z,tx,np.where(rx==3,1,rx),day),(z,tx,rx,day-1),(z*np.nan,tx,rx,day),(z-1,tx,rx,day),(z+1,tx,rx,day)):
        reject(lambda:cells(zz,tt,rr,dd));count+=1
    for aa in (a[:5],a*np.nan):reject(lambda:effect(aa));count+=1
    for c in (dict(d.config(d.rows()[0]),source_RX=[1,3,4,6]),dict(d.config(d.rows()[0]),target_access=True),dict(d.config(d.rows()[0]),checkpoint_loads=1),dict(d.config(d.rows()[0]),primary={})):
        reject(lambda:d.validate(c));count+=1
    return dict(status='PASS',protocol_negatives=count,synthetic_descriptor_packets=5040,exact_widely_linear_descriptor=True,scalar_phase_gain_DC_invariance=True,
        stable_effect_18fold_PASS=True,TX_RX_interaction_and_zero_FAIL=True,tiny_stable_effect_FAIL=True,held_TX_AND_day_prediction_isolation=True,
        real_IQ_packets=0,checkpoint_loads=0,model_forwards=0,optimizer_updates=0,synthetic_fixtures_not_results=True)
