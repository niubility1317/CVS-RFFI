"""Focused configuration,load-path and unchanged physical-model numerical checks."""
import copy,json,sys
import numpy as np
from . import design as d
from .quality import process
from .checkpoint_policy import load

def run():
 from leo_practical.channel import Config,receiver_for_session,normalize_input
 from leo_practical.residual import ResidualChannel
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Unauthorized scope accepted')
 base=d.config(d.rows()[0])
 for key,value in [('epochs',11),('label_epochs',1),('pseudo_epochs',9),('updates',2442),('cumulative_epochs',31),('parent_source','other'),('quality_mode','noise_scaled'),('quality_source_evidence','other'),('control_run','other'),('new_losses',1),('new_U_rules',1),('new_neural_forwards',1),('physical_channel_prepasses',1),('model_seed',2026092702)]:
  c=copy.deepcopy(base);c[key]=value;bad(lambda c=c:d.validate(c))
 for key,value in [('RX',[3,4,6,8]),('L',5580),('U_TX_truth_hidden',False),('source_RX_holdout',1)]:
  c=copy.deepcopy(base);c['data_contract'][key]=value;bad(lambda c=c:d.validate(c))
 sentinel=object();calls=[]
 def actual(path,dev):calls.append((path,dev));return 'OWN_FINAL'
 assert load('/p/final','cpu','/p/final','/r/final',sentinel,actual) is sentinel and not calls
 assert load('/r/final','cpu','/p/final','/r/final',sentinel,actual)=='OWN_FINAL' and len(calls)==1
 for path in ('/other/final','/r/latest','/p/best'):
  bad(lambda path=path:load(path,'cpu','/p/final','/r/final',sentinel,actual))
 t=np.arange(256);n=0;states=set()
 for scene in d.TRAIN_SCENES:
  cfg=Config(fs_hz=25000000.,fc_hz=2462000000.,scenario=scene,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');hw=receiver_for_session(cfg,d.CHANNEL_SEEDS['receiver'],'synthetic')
  for seed in range(1,9):
   for raw in (np.exp(.01j*t),np.exp(.9j*t)+.5*np.exp(-.4j*t)):
    x,_=normalize_input(raw);saved=x.copy();m=ResidualChannel(cfg,seed,hw);a=process(m,x,True);b=ResidualChannel(cfg,seed,hw).process(x,quality_snr_db=a[1]['quality_snr_db'],return_traces=True)
    assert np.array_equal(a[0],b[0]) and np.array_equal(x,saved) and m.used and a[1]['quality_prefilter_calls']==1 and a[1]['no_complete_channel_prepass'] and np.array_equal(a[2]['propagation_kernel'],b[2]['propagation_kernel'])
    states.add(a[1]['state_end']);n+=1
 return dict(status='PASS',protocol_negatives=negative,actual_quality_single_pass_cases=n,paired_reference_calls=n,states=sorted(states),real_IQ=0,CP=0,NN=0,updates=0,own_saved_final_loader_recovery=True)
if __name__=='__main__':
 sys.path.insert(0,str(d.ROOT/'experiments/adv3b02_xuc/code'));print(json.dumps(run()))
