"""Native synthetic physics/protocol checks;zero realIQ/CP/NN."""
import copy,json,math,sys
from pathlib import Path
import numpy as np
from . import design as d
from .physics import measure,power,qdb,held_receiver
from .metrics import criteria,group_summary
from .analyze import calc,quality,summarize,identical

def run(engine_path):
 sys.path.insert(0,str(engine_path))
 import leo_practical.channel as ch
 import leo_practical.residual as res
 from leo_practical.equalization import linear_filter
 c=d.config(d.rows()[0]);d.validate(c);n=0
 def reject(call):
  nonlocal n
  try:call()
  except (ValueError,KeyError):n+=1;return
  raise AssertionError('Invalid input or fixed design accepted')
 for key,value in [('target_access',True),('source_L',6300),('checkpoint_loads',1),('automatic_training',True),('epochs',20),('held_RX1_used',True),('U_used',True),('CUDA_VISIBLE_DEVICES','0')]:
  bad=copy.deepcopy(c);bad[key]=value;reject(lambda:d.validate(bad))
 cfg=ch.Config(fs_hz=25000000.,fc_hz=2462000000.,scenario='practical_low_urban',processing_route='residual',mode='post_sync',equalization_enabled=False);receiver=ch.receiver_for_session(cfg,d.RECEIVER_SEED,'rx3:day1:eq1')
 x=np.exp(2j*np.pi*.123*np.arange(256));reject(lambda:measure(ch,res,linear_filter,cfg,x[:255],d.SEED,receiver));bad=x.copy();bad[1]=np.nan;reject(lambda:measure(ch,res,linear_filter,cfg,bad,d.SEED,receiver));reject(lambda:qdb(0.,1.));reject(lambda:qdb(1.,0.))
 for kwargs in [dict(equalization_enabled=True),dict(reference_snr_db=31.),dict(boundary='edge')]:
  from dataclasses import replace
  reject(lambda:measure(ch,res,linear_filter,replace(cfg,**kwargs),x,d.SEED,receiver))
 results=[]
 for scene in d.SCENES:
  cc=ch.Config(fs_hz=25000000.,fc_hz=2462000000.,scenario=scene,processing_route='residual',mode='post_sync',equalization_enabled=False)
  r=measure(ch,res,linear_filter,cc,x,ch.stable_seed(d.SEED,'synthetic',scene),receiver);assert r['proxy_replay_MSE']==r['fixed_receiver_MSE']==0 and r['same_H'] and r['same_noise_draw_stream'] and r['native_process_calls']==5 and r['numerical_filter_latency_samples']==24 and r['interior_count']>=128;r.update(RX=3,TX=0,day=1,draw=0,scene=scene,session='rx3:day1:eq1');calc(r);assert identical(summarize([r]),group_summary([r]));results.append(r)
 for threshold,status in [(5.,'degraded'),(15.,'locked'),(4.999,'unlocked'),(14.999,'degraded')]:assert quality(10**(threshold/10),1.)[1]==status
 for field,value in [('RX',1),('native_process_calls',4),('interior_start',0),('true_total_snr_known',True),('same_H',False),('fixed_receiver_MSE',.01),('noise_variance_reference',.002)]:
  bad=copy.deepcopy(results[-1]);bad[field]=value;reject(lambda:calc(bad))
 # Causal FIR boundary-free proof and a colored-input white-proxy counterexample.
 h=np.array([1.,1.]);ones=np.ones(256,complex);y,_=linear_filter(ones,h,'reflect');ye,_=linear_filter(ones,h,'edge');assert np.array_equal(y[1:],ye[1:]) and abs(qdb(power(y),1)-qdb(power(ones)*sum(abs(h)**2),1)-10*math.log10(2))<1e-12
 # White ensemble expectation tends to tap-energy proxy;no fitted parameters.
 rng=np.random.default_rng(15);white=(rng.normal(size=(4096,256))+1j*rng.normal(size=(4096,256)))/math.sqrt(2);p=np.mean(abs(white[:,1:]+white[:,:-1])**2);assert abs(p-2)<.01
 groups={str(i)+'/'+s:dict(median_abs_bias_db=2.,whole_lock_flip_fraction=.1,median_whole_vs_interior_abs_db=.1,boundary_robust_flip_fraction=1.) for i in range(2) for s in d.PRIMARY};assert all(criteria(groups,128).values())
 counter=0
 for key,value in [('median_abs_bias_db',.99),('whole_lock_flip_fraction',.049),('median_whole_vs_interior_abs_db',.51),('boundary_robust_flip_fraction',.49)]:
  g=copy.deepcopy(groups);g['0/'+d.PRIMARY[0]][key]=value;assert not all(criteria(g,128).values());counter+=1
 assert not all(criteria(groups,127).values());counter+=1
 old=res.receiver_quality
 try:
  with held_receiver(res,'degraded',2.):raise RuntimeError('restore test')
 except RuntimeError:pass
 assert res.receiver_quality is old
 return dict(status='PASS',input_protocol_counterexamples=n,scientific_counterexamples=counter,native_synthetic_packets=len(results),native_process_calls=5*len(results),analytic_white_observations=4096*256,real_IQ=0,checkpoint_loads=0,NN=0,optimizer_updates=0,target_access=False,engine_files=dict(channel=ch.__file__,residual=res.__file__))
if __name__=='__main__':print(json.dumps(run(sys.argv[1])))
