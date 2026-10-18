"""Focused fixed-scope negatives,energy-matched DSP and ranking controls."""
import copy,json
import numpy as np
from . import design as d,dsp
from .analyze import metric,rerank,qualify

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError):negative+=1
  else:raise AssertionError('Invalid fixed scope accepted')
 for k,v in [('target_access',True),('U_used',True),('source_RX_holdout',1),('training_updates',1),('planned_NN',118980),('automatic_training',True),('L_fit',179),('per_cell',19)]:
  c=copy.deepcopy(d.config());c[k]=v;bad(lambda c=c:d.validate(c))
 for k,v in [('grid',8),('gain_bound',.2),('nominal_relative_IQ_step',.02),('extra_RMS',True),('new_noise',True)]:
  c=copy.deepcopy(d.config());c['operator'][k]=v;bad(lambda c=c:d.validate(c))
 rng=np.random.default_rng(20261018111);errors={};n=0
 for dtype,tol in ((np.float64,1e-12),(np.float32,2e-7)):
  x=rng.normal(size=(7,2,256)).astype(dtype);x[0]=0;x[1]=1;x[2,:,0:16]=0;old=x.copy();delta,amp=dsp.prepare(x);actual=[]
  manual_w=np.array([[np.sin(np.pi*(i-16*b)/15)**2 if 16*b<i<16*b+15 else 0. for i in range(256)] for b in range(16)])
  for b in range(16):
   for s in (-1,1):
    y,st=dsp.apply(x,b,s);expected=(x.astype(float)*(1+s*amp[:,b,None,None]*manual_w[b][None,None,:])).astype(dtype)
    assert np.max(np.abs(y-expected))<tol and st['outside_unchanged'] and st['step_error']<tol and st['max_gain']<=.1+1e-12
    assert np.array_equal(y[0],x[0]) and np.array_equal(y[2],x[2]);actual.append(np.linalg.norm((y.astype(float)-x).reshape(7,-1),axis=1)/np.maximum(np.linalg.norm(x.astype(float).reshape(7,-1),axis=1),1e-24));n+=len(x)
  max_spread=float(np.ptp(np.asarray(actual),axis=0).max());assert max_spread<tol and np.array_equal(x,old)
  perm=[6,0,5,1,4,2,3];assert np.array_equal(dsp.apply(x[perm],4,-1)[0],dsp.apply(x,4,-1)[0][perm])
  errors[str(dtype)]=dict(max_across32_view_relative_step_spread=max_spread,max_gain=float(amp.max()))
 x=np.ones((2,2,256),dtype=np.float32)
 for z,b,s in [(x[:,:,:255],0,1),(x.astype(int),0,1),(x.tolist(),0,1),(x,-1,1),(x,16,1),(x,True,1),(x,0,0),(x,0,.5)]:bad(lambda z=z,b=b,s=s:dsp.apply(z,b,s))
 z=x.copy();z[0,0,0]=np.nan;bad(lambda:dsp.apply(z,0,1))
 cells=[(t,r,z) for t in range(6) for r in d.FIT for z in (1,2,3) for i in range(2)];y,r,z=map(list,zip(*cells));a=np.tile(np.arange(1,17,dtype=float),(180,1));ranking=dsp.rank(a,y,r,z)
 fit=dict(ids=list(range(180)),y=y,rx=r,day=z,derivatives=a.tolist(),ranking=ranking);ind=rerank(fit);assert ind['low_blocks']==[0,1,2,3] and ind['high_blocks']==[12,13,14,15]
 perm=rng.permutation(180);assert dsp.rank(a[perm],np.asarray(y)[perm],np.asarray(r)[perm],np.asarray(z)[perm])==ranking
 bad(lambda:dsp.rank(a[:179],y[:179],r[:179],z[:179]));wrong=y.copy();wrong[0]=1;bad(lambda:dsp.rank(a,wrong,r,z));badfit=copy.deepcopy(fit);badfit['ranking']['low_blocks']=[1,2,3,4];bad(lambda:rerank(badfit))
 numerics=[dict(block=b,sign=s,max_gain=.1,step_error=0.,outside_unchanged=True,max_relative_change=.01) for b in range(16) for s in (-1,1)];pairs=[]
 for arm in d.MODELS:
  for b in range(16):
   for s in (-1,1):
    new=1 if b in ranking['low_blocks'] else 5 if b in ranking['high_blocks'] else 3
    pairs.append(dict(arm=arm,block=b,sign=s,baseline_correct=1000,new_errors=new,repairs=0,delta_accuracy_pp=-new/18,correct_retention=1-new/1000,RX_delta_pp={str(r):0. for r in d.FIT},TX_delta_pp={str(t):0. for t in range(6)}))
 assert qualify(pairs,ranking,numerics,[])['passed']
 damaged=copy.deepcopy(pairs);damaged[0]['correct_retention']=.99;assert not qualify(damaged,ranking,numerics,[])['passed']
 uninformative=copy.deepcopy(pairs)
 for p in uninformative:p['new_errors']=0
 assert not qualify(uninformative,ranking,numerics,[])['passed']
 assert not qualify(pairs,dsp.rank(np.zeros((180,16)),y,r,z),numerics,[])['passed'] and not qualify(pairs,ranking,numerics,['unavailable'])['passed']
 known=metric([0,0,1,2],[0,1,1,2]);assert known['correct']==3 and known['count']==4 and abs(known['macro_F1']-7/18)<1e-12
 return dict(status='PASS',protocol_negatives=negative,synthetic_transformed_packets=n,numerics=errors,independent_L_ranking=True,energy_equal_all32views=True,source_criteria_positive_and_failure_controls=True,real_IQ=0,CP=0,NN=0,updates=0)
if __name__=='__main__':print(json.dumps(run()))
