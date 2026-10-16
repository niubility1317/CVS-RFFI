"""Known gain-envelope cases, no dataset or model."""
import numpy as np


def run():
    from .core import observe,crossfit
    rng=np.random.default_rng(20261016091)
    n=5040;tx=np.tile(np.repeat(np.arange(6),70),12);day=np.tile(np.repeat(np.arange(1,4),420),4);rx=np.repeat((3,4,6,8),1260)
    values={}
    for name in ('RX','TX','DAYFLIP'):
        slope={3:.03,4:.08,6:.15,8:.25}
        s=np.array([slope[int(r)] for r in rx])
        if name=='TX':s=.04+.02*tx
        if name=='DAYFLIP':s=np.where(day==1,s,np.where(day==2,.30-s,.15))
        s=s+rng.normal(0,.0001,n)
        x=np.ones((n,2,256));q=(np.arange(256)-119.5)/30.
        x*=10**((s[:,None,None]*q[None,None])/20)
        o=observe(x);values[name]=crossfit(o['slope_db'],tx,rx,day)
    assert values['RX']['decision']=='PASS'
    assert values['TX']['decision']=='FAIL'
    assert values['DAYFLIP']['decision']=='FAIL'
    # Period means of exp(slope*n) have exact dB differences despite finite bins.
    s=np.array([-.1,0.,.2]);q=(np.arange(256)-119.5)/30.
    x=np.ones((3,2,256))*10**((s[:,None,None]*q[None,None])/20)
    a=observe(x);b=observe(x/np.sqrt((x*x).mean((1,2),keepdims=True)))
    assert np.allclose(a['slope_db'],s,atol=1e-12)
    assert np.allclose(a['slope_db'],b['slope_db'],atol=1e-12)
    assert a['linear_R2'][0]>.999999 and a['linear_R2'][2]>.999999
    assert a['linear_R2'][1]==0
    zero=crossfit(np.zeros(n),tx,rx,day)
    assert zero['decision']=='FAIL' and zero['mean_risk_gain'] is None
    negatives=0
    for args in ((np.zeros(n-1),tx,rx,day),(np.zeros(n),np.full(n,-1),rx,day),(np.zeros(n),tx,np.ones(n),day),(np.full(n,np.nan),tx,rx,day)):
        try:crossfit(*args)
        except ValueError:negatives+=1
        else:raise AssertionError('Invalid source matrix accepted')
    for bad in (np.zeros((3,2,255)),np.zeros((3,2,256)),np.full((3,2,256),np.nan)):
        try:observe(bad)
        except ValueError:negatives+=1
        else:raise AssertionError('Invalid gain observation accepted')
    return dict(status='PASS',known_cases={k:v['decision'] for k,v in values.items()},slope_formula_exact=True,RMS_invariant=True,negative_cases=negatives,synthetic_packets=15120,network_forwards=0,optimizer_steps=0)


if __name__=='__main__':
    import json
    print(json.dumps(run(),indent=2))
