"""Focused protocol negatives and same-latent calibration-only equivalence."""
import copy,json,sys
from dataclasses import asdict,replace
import numpy as np
from . import design as d
from .calibration import refresh
from .checkpoint_policy import load

def run():
 from leo_practical.channel import Config,receiver_for_session,normalize_input,iq_imbalance,compensate_iq
 from leo_practical.residual import ResidualChannel
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Unauthorized scope accepted')
 base=d.config(d.rows()[0])
 for key,value in [('epochs',11),('label_epochs',1),('pseudo_epochs',9),('updates',2442),('cumulative_epochs',31),('parent_source','other'),('calibration_mode','packet_Tx'),('calibration_seed',20261017542),('control_run','other'),('new_losses',1),('new_U_rules',1),('new_neural_forwards',1),('physical_channel_prepasses',1),('extra_physical_filters',1),('model_seed',2026092702)]:
  c=copy.deepcopy(base);c[key]=value;bad(lambda c=c:d.validate(c))
 for key,value in [('RX',[3,4,6,8]),('L',5580),('U_TX_truth_hidden',False),('source_RX_holdout',1)]:
  c=copy.deepcopy(base);c['data_contract'][key]=value;bad(lambda c=c:d.validate(c))
 for key,value in [('amp_error_std_db',.1),('phase_error_std_deg',1.),('TX_dependence',True)]:
  c=copy.deepcopy(base);c['calibration_policy'][key]=value;bad(lambda c=c:d.validate(c))
 sentinel=object();calls=[]
 def actual(path,dev):calls.append((path,dev));return 'OWN_FINAL'
 assert load('/p/final','cpu','/p/final','/r/final',sentinel,actual) is sentinel and not calls
 assert load('/r/final','cpu','/p/final','/r/final',sentinel,actual)=='OWN_FINAL' and len(calls)==1
 for path in ('/other/final','/r/latest','/p/best'):bad(lambda path=path:load(path,'cpu','/p/final','/r/final',sentinel,actual))
 cfg=Config(fs_hz=25000000.,fc_hz=2462000000.,processing_route='residual');hw=receiver_for_session(cfg,d.CHANNEL_SEEDS['receiver'],'synthetic')
 for ep in (20,31,True,21.5):bad(lambda ep=ep:refresh(cfg,hw,ep,d.CALIBRATION_SEED))
 for c in (replace(cfg,iq_estimation_amp_std_db=.1),replace(cfg,iq_estimation_phase_std_deg=1.),replace(cfg,iq_compensation_enabled=False)):bad(lambda c=c:refresh(c,hw,21,d.CALIBRATION_SEED))
 state=np.random.get_state();t=np.arange(256);n=0;states=set();maximum_relative_change=0.;calibrations=[]
 for scene in d.TRAIN_SCENES:
  c=replace(cfg,scenario=scene);hw=receiver_for_session(c,d.CHANNEL_SEEDS['receiver'],'synthetic')
  for ep in range(21,31):
   alt=refresh(c,hw,ep,d.CALIBRATION_SEED);again=refresh(c,hw,ep,d.CALIBRATION_SEED)
   assert alt==again and all(asdict(alt)[k]==asdict(hw)[k] for k in ('session_id','amplitude_imbalance_db','quadrature_error_deg','oscillator_cfo_hz'))
   calibrations.append((alt.estimated_amplitude_imbalance_db,alt.estimated_quadrature_error_deg))
   for seed in range(1,5):
    for raw in (np.exp(.01j*t),np.exp(.9j*t)+.5*np.exp(-.4j*t)):
     x,_=normalize_input(raw);saved=x.copy();a=ResidualChannel(c,seed,hw).process(x,return_traces=True);b=ResidualChannel(c,seed,alt).process(x,return_traces=True)
     assert np.array_equal(x,saved) and np.isfinite(b[0]).all() and np.array_equal(a[2]['propagation_kernel'],b[2]['propagation_kernel']) and a[2]['equalizer_kernel'].tolist()==[1.+0j] and b[2]['equalizer_kernel'].tolist()==[1.+0j]
     for k in ('geometry','quality_snr_db','receiver_lock_status','residual_scale','output_frequency_hz','estimated_frequency_hz','added_noise_variance_reference','state_end','channel_equalization_applied','phase_tracking_applied'):assert a[1][k]==b[1][k],k
     assert b[1]['receiver']==asdict(alt) and b[1]['iq_compensation_applied'] and not b[1]['tx_impairment_correction_applied']
     expected=ResidualChannel(c,seed,again).process(x)[0];assert np.array_equal(b[0],expected)
     maximum_relative_change=max(maximum_relative_change,float(np.linalg.norm(b[0]-a[0])/np.linalg.norm(a[0])))
     states.add(a[1]['state_end']);n+=1
 assert len(set(calibrations))==10
 after=np.random.get_state();assert state[0]==after[0] and np.array_equal(state[1],after[1]) and state[2:]==after[2:]
 # Exact zero-error compensation inverts injected hardware, not source TX.
 exact=replace(hw,estimated_amplitude_imbalance_db=hw.amplitude_imbalance_db,estimated_quadrature_error_deg=hw.quadrature_error_deg);z=np.exp(.3j*t)
 assert np.allclose(compensate_iq(iq_imbalance(z,exact),exact.estimated_amplitude_imbalance_db,exact.estimated_quadrature_error_deg),z,atol=1e-12)
 return dict(status='PASS',protocol_negatives=negative,calibration_only_same_latent_cases=n,reference_residual_calls=n,candidate_residual_calls=2*n,states=sorted(states),max_relative_IQ_change_synthetic=maximum_relative_change,epoch_session_repeatable=True,true_hardware_unchanged=True,quality_and_noise_law_unchanged=True,global_numpy_RNG_unchanged=True,real_IQ=0,CP=0,NN=0,updates=0,REJECTED_EXTRA_GATE='Separate brainstorming approval is NONBLOCKING under explicit user minimal experiment workflow')

if __name__=='__main__':
 sys.path.insert(0,str(d.ROOT/'code'));print(json.dumps(run()))
