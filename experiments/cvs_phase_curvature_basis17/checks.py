"""Focused source-permission,nonaffine observation and crossfit falsification checks."""
import copy,json
import numpy as np
from . import design as d
from .core import observe,analyze,decide

def main():
 rng=np.random.default_rng(202610173101);neg=0
 def reject(fn):
  nonlocal neg
  try:fn()
  except (ValueError,KeyError):neg+=1;return
  raise AssertionError('Invalid sourceinput/config accepted')
 x=rng.normal(size=(64,2,256));saved=x.copy();a=observe(x);z=x[:,0]+1j*x[:,1];n=np.arange(256);y=z*np.exp(1j*(.37+.002*n));b=observe(np.stack((y.real,y.imag),1));assert np.array_equal(x,saved) and np.allclose(a['curve'],b['curve'],atol=1e-12,rtol=1e-10)
 alpha=.0007;y=z*np.exp(1j*alpha*n*n);c=observe(np.stack((y.real,y.imag),1));old=a['curve'].reshape(64,78,2);expected=(old[:,:,0]+1j*old[:,:,1])*np.exp(-2j*alpha);assert np.allclose(c['curve'].reshape(64,78,2),np.stack((expected.real,expected.imag),-1),atol=2e-12,rtol=1e-10)
 outside=x.copy();outside[:,:,:80]=rng.normal(size=(64,2,80))*100;outside[:,:,160:]=rng.normal(size=(64,2,96))*100;assert np.array_equal(a['curve'],observe(outside)['curve'])
 tone=np.exp(1j*(.1+.003*n));tones=np.tile(np.stack((tone.real,tone.imag))[None],(4,1,1));t=observe(tones);assert np.allclose(t['curve'].reshape(4,78,2)[:,:,0],1/(1+d.EPS),atol=1e-12) and np.max(abs(t['curve'].reshape(4,78,2)[:,:,1]))<1e-12
 q=tone*np.exp(1j*alpha*n*n);u=observe(np.tile(np.stack((q.real,q.imag))[None],(4,1,1)));assert np.allclose(u['curve'].reshape(4,78,2)[:,:,0],np.cos(-2*alpha)/(1+d.EPS),atol=2e-12)
 zero=observe(np.zeros((2,2,256)));assert np.max(abs(zero['curve']))==0 and np.all(zero['zero_triplets']==78)
 for bad in (np.zeros((0,2,256)),np.zeros((1,1,256)),np.zeros((1,2,255)),x*np.nan,x*np.inf,np.ones((1,2,256),dtype=int),np.ones((1,2,256),dtype=complex)):reject(lambda bad=bad:observe(bad))
 tx=[];rx=[];day=[];features=[];domain=rng.uniform(-.15,.15,size=(4,156));txpart=rng.uniform(-.02,.02,size=(6,156));daypart=rng.uniform(-.02,.02,size=(3,156))
 for i,r in enumerate(d.FIT):
  for j,dd in enumerate((1,2,3)):
   for t in range(6):
    for k in range(70):tx.append(t);rx.append(r);day.append(dd);features.append(domain[i]+txpart[t]+daypart[j]+rng.normal(scale=.0001,size=156))
 tx,rx,day=(np.asarray(v,dtype=np.int64) for v in (tx,rx,day));obs=dict(curve=np.asarray(features));good=analyze(obs,tx,rx,day);assert good['decision']=='PASS' and good['positive_groups']==18
 for f in good['folds']:assert f['held_TX'] not in f['train_TX'] and f['held_day'] not in f['train_days'] and len(f['train_TX'])==5 and len(f['train_days'])==2 and f['train_group_count']==10
 noRX=dict(curve=np.asarray([txpart[t]+daypart[dd-1] for t,dd in zip(tx,day)]));assert analyze(noRX,tx,rx,day)['decision']=='FAIL'
 dayflip=dict(curve=np.asarray([domain[d.FIT.index(r)]*(1 if dd!=2 else -1)+txpart[t] for t,r,dd in zip(tx,rx,day)]));assert analyze(dayflip,tx,rx,day)['decision']=='FAIL'
 txflip=dict(curve=np.asarray([domain[d.FIT.index(r)]*(1 if t<3 else -1)+daypart[dd-1] for t,r,dd in zip(tx,rx,day)]));assert analyze(txflip,tx,rx,day)['decision']=='FAIL'
 base=np.ones((18,4,156));model=base*.5;assert decide(base,model)['decision']=='PASS'
 lowgain=base*.91;assert decide(base,lowgain)['decision']=='FAIL'
 onebad=model.copy();onebad[:,3]=1.1;v=decide(base,onebad);assert v['conditions']['pooled_transfer'] and not v['conditions']['all_RX_nonnegative'] and v['decision']=='FAIL'
 few=model.copy();few[:4]=1.1;v=decide(base,few);assert v['conditions']['pooled_transfer'] and v['positive_groups']==14 and v['decision']=='FAIL'
 assert decide(base*0,base*0)['decision']=='FAIL'
 for key,value in [('tx',-1),('tx',6),('rx',1),('day',4)]:
  meta=dict(tx=tx.copy(),rx=rx.copy(),day=day.copy());meta[key][0]=value;reject(lambda meta=meta:analyze(obs,**meta))
 reject(lambda:analyze(obs,tx,rx.astype(float),day));wrong=tx.copy();wrong[0]=1;reject(lambda:analyze(obs,wrong,rx,day))
 for curve in (np.zeros((5040,155)),np.full((5040,156),np.nan),np.full((5040,156),2.)):reject(lambda curve=curve:analyze(dict(curve=curve),tx,rx,day))
 reject(lambda:analyze({},tx,rx,day))
 for bb,mm in ((base[:17],model[:17]),(base*np.nan,model),(base,-model)):reject(lambda bb=bb,mm=mm:decide(bb,mm))
 for key,value in [('window',[20,180]),('triplet_lag',2),('numerical_epsilon',1e-8),('fit_RX',[1,3,4,6]),('network_forwards',2),('checkpoint_loads',1),('target_access',True),('automatic_training',True)]:
  c=d.config();c[key]=value;reject(lambda c=c:d.validate(c))
 c=d.config();c['criteria']['min_pooled_risk_gain']=.05;reject(lambda:d.validate(c))
 result=dict(status='PASS',input_config_negatives=neg,scientific_counterexamples=7,synthetic_IQ_observations=266,synthetic_metadata_vectors=5040,affine_phase_invariance=True,known_quadratic_phase_response=True,strict_reference_window_no_wrap=True,zeros_and_all_points_retained=True,held_TX_and_day_both_excluded=True,persistent_RX_case_PASS=True,TX_day_only_or_reversal_FAIL=True,onebadRX_and_14positive_groups_FAIL=True,source_IQ_reads=0,checkpoint_loads=0,network_forwards=0,optimizer_steps=0,head_fit_updates=0,target_access=False)
 print(json.dumps(result));return result
if __name__=='__main__':main()
