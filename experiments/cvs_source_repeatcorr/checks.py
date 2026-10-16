import json
import numpy as np
from . import design as d
from .core import describe,cells,prediction,analyze

def run():
    rng=np.random.default_rng(20261017161);profile=np.asarray([[-.4,.1],[-.1,-.3],[.15,.35],[.45,.1]])
    a=np.broadcast_to(profile[None,:,None,:],(6,4,3,2)).copy()+rng.normal(0,.002,(6,4,3,2))
    quality=dict(valid_packets=5040,coherence_p10=.99,RX_median_residual_ratio=[.001]*4)
    good,p=analyze(a,quality);assert good['status']=='PASS' and good['positive_folds']==18
    e=a-a.mean(axis=1,keepdims=True);altered=a.copy();altered[2]=rng.normal(size=(4,3,2));altered[:,:,1]=rng.normal(size=(6,4,2))
    ee=altered-altered.mean(axis=1,keepdims=True);assert np.array_equal(prediction(e,2,1),prediction(ee,2,1))
    flipped=e.copy();flipped[:3]*=-1;bad,_=analyze(flipped,quality);assert bad['status']=='FAIL' and bad['aggregate_risk_reduction']<0
    white,_=analyze(np.zeros_like(a),quality);assert white['status']=='FAIL' and not white['checks']['nonwhite']
    for key,value in [('valid_packets',5039),('coherence_p10',.79),('RX_median_residual_ratio',[.01,.01,.101,.01])]:
        q=dict(quality,**{key:value});s,_=analyze(a,q);assert s['status']=='FAIL'
    # Known disjoint residuals: joining the two20-point blocks would change rho.
    x=np.ones((1,2,256));x[:,1]=0;x[:,0]=100
    first=np.linspace(-3,4,20);last=np.linspace(6,-2,20);x[0,0,100:120]+=first;x[0,0,140:160]+=last
    z,q=describe(x);r=np.stack((first,last))/np.sqrt(2);expected=np.sum(r[:,1:]*r[:,:-1])/np.sqrt(np.sum(r[:,1:]**2)*np.sum(r[:,:-1]**2))
    assert abs(z[0,0]-expected)<1e-13 and z[0,1]==0 and q['rho_valid'][0]
    random=rng.normal(size=(128,2,256));z,q=describe(random);scale=rng.uniform(.3,3,(128,1,1));zz,qq=describe(random*scale)
    assert np.max(np.abs(z-zz))<1e-12 and all(np.max(np.abs(q[k].astype(float)-qq[k].astype(float)))<1e-12 for k in q)
    changed=random.copy();changed[:,:,:80]*=100;changed[:,:,160:]*=-100;assert np.array_equal(describe(changed)[0],z)
    periodic=np.tile(rng.normal(size=(8,2,20)),(1,1,13))[:,:,:256];z0,q0=describe(periodic);assert not q0['rho_valid'].any() and np.array_equal(z0,np.zeros((8,2)))
    # Synthetic waveform validation, no real IQ or neural network involved.
    tx=[];rx=[];day=[]
    for t in d.TX:
        for rr in d.FIT:
            for dd in d.DAYS:
                for _ in range(70):tx.append(t);rx.append(rr);day.append(dd)
    tx,rx,day=map(np.asarray,(tx,rx,day));coefficient=profile[:,0]+1j*profile[:,1];coef=np.asarray([coefficient[d.FIT.index(r)] for r in rx])
    signal=rng.normal(size=(6,20))+1j*rng.normal(size=(6,20));signal/=np.sqrt(np.mean(np.abs(signal)**2,axis=1))[:,None]
    noise=(rng.normal(size=(5040,256))+1j*rng.normal(size=(5040,256)))/np.sqrt(2)
    for n in range(1,256):noise[:,n]=coef*noise[:,n-1]+np.sqrt(1-np.abs(coef)**2)*noise[:,n]
    waveform=signal[tx][:,np.arange(256)%20]+.02*noise;x=np.stack((waveform.real,waveform.imag),axis=1)
    z,q=describe(x);a=cells(z,tx,rx,day)
    quality=dict(valid_packets=int(q['rho_valid'].sum()),coherence_p10=float(np.quantile(q['coherence'],.1)),RX_median_residual_ratio=[float(np.median(q['residual_ratio'][rx==r])) for r in d.FIT])
    realshape,_=analyze(a,quality);assert realshape['status']=='PASS'
    negatives=0
    def reject(call):
        nonlocal negatives
        try:call()
        except ValueError:negatives+=1;return
        raise AssertionError('Invalid input accepted')
    for bad in (x[:,:,:255],x*np.nan,np.zeros_like(x[:1])):reject(lambda:describe(bad))
    for zz,tt,rr,dd in ((z[:-1],tx[:-1],rx[:-1],day[:-1]),(z,tx-1,rx,day),(z,tx,np.where(rx==3,1,rx),day),(z,tx,np.where(rx==3,2,rx),day),(z,tx,rx,day-1),(z*np.nan,tx,rx,day)):
        reject(lambda:cells(zz,tt,rr,dd))
    for bad in (a[:5],a*np.nan,a*100):reject(lambda:analyze(bad,quality))
    for c in (dict(d.config(d.rows()[0]),target_access=True),dict(d.config(d.rows()[0]),checkpoint_loads=1),dict(d.config(d.rows()[0]),source_RX=[1,3,4,6]),dict(d.config(d.rows()[0]),observation=dict(window=[0,80])),dict(d.config(d.rows()[0]),model_forwards=2)):
        reject(lambda:d.validate(c))
    return dict(status='PASS',protocol_negatives=negatives,valid_synthetic_descriptor_evaluations=5433,synthetic_waveform_packets=5040,synthetic_positive18folds=True,white_entangled_bad_quality_FAIL=True,
        disjoint_residual_boundaries=True,RMS_invariant=True,undefined_rho_fails_without_dropping=True,held_TX_AND_day_isolation=True,real_IQ_packets=0,checkpoint_loads=0,model_forwards=0,optimizer_updates=0)
if __name__=='__main__':print(json.dumps(run()))
