"""Known DC generative cases; no research IQ, model, checkpoint or target."""
import numpy as np


def data(mode):
    rng=np.random.default_rng(20261016081)
    values=[];tx=[];rx=[];day=[]
    offsets=np.array([[.02,.01],[-.02,.02],[.01,-.03],[-.01,-.01]])
    for r in range(4):
        for d in range(3):
            for t in range(6):
                if mode=='RX':mean=offsets[r]+np.array([t-2.5,2.5-t])*.0001
                elif mode=='TX':mean=np.array([t-2.5,2.5-t])*.01
                elif mode=='DAYFLIP':mean=offsets[r]*(1 if d==0 else -1 if d==1 else 0)
                else:raise ValueError('Unknown synthetic case')
                for _ in range(70):
                    values.append(mean+rng.normal(0,.0005,2));tx.append(t);rx.append((3,4,6,8)[r]);day.append(d+1)
    return np.array(values),np.array(tx),np.array(rx),np.array(day)


def run():
    from .core import crossfit,observe
    cases={name:crossfit(*data(name)) for name in ('RX','TX','DAYFLIP')}
    assert cases['RX']['decision']=='PASS'
    assert cases['TX']['decision']=='FAIL'
    assert cases['DAYFLIP']['decision']=='FAIL'
    c,t,r,d=data('RX')
    zero=crossfit(np.zeros_like(c),t,r,d)
    assert zero['decision']=='FAIL' and zero['mean_risk_gain'] is None
    perm=np.array([4,1,5,0,3,2]);again=crossfit(c,perm[t],r,d)
    assert abs(again['mean_risk_gain']-cases['RX']['mean_risk_gain'])<1e-12
    x=np.zeros((4,2,256));x[:,0]=.03;x[:,1]=-.02
    obs=observe(x)
    assert np.allclose(obs['dc'],np.array([.03,-.02]))
    assert np.allclose(obs['odd_even_DC_difference'],0.)
    negatives=0
    for args in ((c[:-1],t,r,d),(c,np.full(len(t),-1),r,d),(c,t,np.ones(len(r)),d),(c,t,r,np.zeros(len(d))), (np.full_like(c,np.nan),t,r,d)):
        try:crossfit(*args)
        except ValueError:negatives+=1
        else:raise AssertionError('Illegal source statistic accepted')
    for bad in (np.zeros((4,2,255)),np.full((4,2,256),np.nan)):
        try:observe(bad)
        except ValueError:negatives+=1
        else:raise AssertionError('Illegal IQ accepted')
    return dict(status='PASS',known_cases={k:v['decision'] for k,v in cases.items()},negative_cases=negatives,TX_permutation_invariant=True,synthetic_packets=15120,network_forwards=0,optimizer_steps=0)


if __name__=='__main__':
    import json
    print(json.dumps(run(),indent=2))
