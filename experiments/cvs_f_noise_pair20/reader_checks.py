"""Independent-scoring protocol negatives and exact threshold oracles;0NN."""
from copy import deepcopy
from .analyze import pair_valid,independent_qualify,gradient_record,ARMS,VIEWS,FIT

def run():
 negative=0
 def rejects(call):
  nonlocal negative
  try:call()
  except (AssertionError,ValueError,KeyError):negative+=1;return
  raise AssertionError('Bad independent-readback record accepted')
 p=dict(physical_id=[1,3,1,0,10],TX=0,RX=3,day=1,step=1,epoch=1,scene='practical_low_urban',session='rx3:day1:eq1',state='LOS',lock='locked',quality_snr_db=25.,output_frequency_hz=100.,residual_scale=1.,residual_kernel_energy=.1,residual_kernel_length=62,numerical_filter_latency_samples=24,boundary='synthetic_reflect',phase_tracking_applied=True,iq_compensation_applied=True,EQ=False,G=1,nonnoise_latents_shared=True,AGC_recomputed_per_view=True,added_noise_variance_reference=.001,signal_to_added_noise_db=20.,agc_gain_db=12.,agc_limited=False,view=0)
 q=dict(p,view=1,agc_gain_db=12.1);pair_valid(p,q)
 for key,value in [('RX',1),('TX',6),('day',0),('G',.99),('EQ',True),('boundary','synthetic_edge'),('added_noise_variance_reference',.01),('numerical_filter_latency_samples',0),('quality_snr_db',float('nan'))]:
  a=deepcopy(p);b=deepcopy(q);a[key]=b[key]=value;rejects(lambda a=a,b=b:pair_valid(a,b))
 b=dict(q,output_frequency_hz=101.);rejects(lambda:pair_valid(p,b));rejects(lambda:pair_valid(p,dict(q,view=2)))
 results={};scores={}
 for arm in ARMS:
  practical=.4 if arm=='RAW10' else (.43 if arm=='INDEP' else .45);low=.3 if arm=='RAW10' else (.34 if arm=='INDEP' else .35)
  results[arm]=dict(actual_steps=4440,L_forward_packets=5683200,extra_L_forward_packets=2273280,noise_pair_steps=4440,supcon_applied=3330,supcon_layer_forward_terms=13320,final_base_L_objective=.5,noise_pair_physical_presentations=568320,As_training_observations=0 if arm=='RAW10' else 1136640,teacher_noise_pair_calls=0,head_fit_updates=0,U_truth_read=False,target_access=False,checkpoint_sources=[],held_RX=1,fit_RX=list(FIT),seen_V=dict(source_val_accuracy=.98))
  scores[arm]={v:dict(accuracy=.8 if v=='clean' else (low if v==VIEWS[-1] else practical),macro_F1=.78) for v in VIEWS}
 assert independent_qualify(results,scores)['qualified']
 for arm,view,key,value in [('ANTI','clean','accuracy',.79999),('ANTI','clean','macro_F1',.77999),('ANTI',VIEWS[-1],'accuracy',.34999),('INDEP','clean','accuracy',.806)]:
  ss=deepcopy(scores);ss[arm][view][key]=value;assert not independent_qualify(results,ss)['qualified'];negative+=1
 for key,value in [('extra_L_forward_packets',1136640),('teacher_noise_pair_calls',1),('head_fit_updates',1),('target_access',True),('actual_steps',4439)]:
  rr=deepcopy(results);rr['ANTI'][key]=value;assert not independent_qualify(rr,scores)['qualified'];negative+=1
 gg={'train/grad_before_clip':10.,'train/grad_total':5.,'train/grad_backbone':5.,'train/grad_clip_limit':5.};assert gradient_record(gg)['grad_pre_exceeds_limit']==1.
 assert gradient_record(dict(gg,**{'train/grad_before_clip':5.}))['grad_pre_exceeds_limit']==0.
 for key,value in [('train/grad_before_clip',None),('train/grad_total',float('nan')),('train/grad_backbone',-1.),('train/grad_clip_limit',0.)]:
  bad=dict(gg);bad[key]=value;rejects(lambda bad=bad:gradient_record(bad))
 return dict(status='PASS',negative_cases=negative,independent_variance_training_gate_boundary=True,per_view_AGC_difference_allowed=True,old8L_counter_rejected=True,new_NN=0,updates=0,IQ_reads=0,target_access=False)
if __name__=='__main__':
 import json
 print(json.dumps(run()))
