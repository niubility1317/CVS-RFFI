"""Focused input/privacy and mathematical controls; no real data or model."""
import json, random
import numpy as np
from .iq import apply_allpass, response
def run():
    rng=np.random.default_rng(1901);x=rng.normal(size=(2,2,256)).astype(np.float32)
    saved=x.copy();py=random.getstate();ns=np.random.get_state()
    assert np.array_equal(apply_allpass(x,0),x)
    constant=np.ones((2,2,256),np.float32)
    assert np.max(np.abs(apply_allpass(constant,[.04,-.04])-constant))<1e-6
    out=apply_allpass(x,[.04,-.04])
    assert out.dtype==np.float32 and out.shape==x.shape and np.array_equal(x,saved)
    assert np.array_equal(apply_allpass(x[::-1],[-.04,.04])[::-1],out)
    assert np.array_equal(apply_allpass(x,[.04,-.04]),out)
    assert py==random.getstate()
    now=np.random.get_state()
    assert ns[0]==now[0] and np.array_equal(ns[1],now[1]) and ns[2:]==now[2:]
    # An uncompensated pure sample shift fails identity; it must not stand in for this view.
    assert not np.array_equal(np.roll(x,1,axis=-1),x)
    w=np.linspace(-np.pi,np.pi,257);assert np.max(np.abs(response(0,w)-1))==0
    negatives=0
    invalid=[(x.astype(np.float64),0),(x[0],0),(x[:,:,:255],0),
             (np.zeros_like(x),0),(x,float("nan")),(x,float("inf")),
             (x,.040001),(x,True),(x,[.01]),(x,[.01,.01,.01])]
    bad=x.copy();bad[0,0,0]=float("nan");invalid.append((bad,0))
    bad=x.copy();bad[0,0,0]=float("inf");invalid.append((bad,0))
    for z,a in invalid:
        try:apply_allpass(z,a)
        except (ValueError,TypeError):negatives+=1
        else:raise AssertionError("Invalid IQ/coefficient was accepted")
    try:apply_allpass(x,.01,TX=[0,1])
    except TypeError:negatives+=1
    else:raise AssertionError("Identity-conditioned argument accepted")
    return dict(status="PASS",protocol_negatives=negatives,successful_apply_calls=5,response_calls=1,
                real_IQ=0,CP=0,NN=0,updates=0)
if __name__=="__main__":print(json.dumps(run()))
