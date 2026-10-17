"""Focused scope negatives and same-latent,original-profile exact replay."""
import copy,json,sys
from dataclasses import asdict,replace
import numpy as np
from . import design as d
from . import pdp_profile as profile
from .checkpoint_policy import load

def run():
 from leo_practical.channel import Config,receiver_for_session,normalize_input
 from leo_practical.residual import ResidualChannel
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Invalid experiment scope accepted')
 c=d.config(d.rows()[0])
 for key,value in [('epochs',11),('label_epochs',1),('pseudo_epochs',9),('updates',2442),('cumulative_epochs',31),('parent_source','other'),('profile_seed',20261017712),('control_run','other'),('new_losses',1),('new_U_rules',1),('new_neural_forwards',1),('physical_channel_prepasses',1),('extra_physical_filters',1),('model_seed',2026092702)]:
  q=copy.deepcopy(c);q[key]=value;bad(lambda q=q:d.validate(q))
 for key,value in [('RX',[3,4,6,8]),('L',5580),('U_TX_truth_hidden',False),('source_RX_holdout',1)]:
  q=copy.deepcopy(c);q['data_contract'][key]=value;bad(lambda q=q:d.validate(q))
 for key,value in [('shapes',[[0.,1.,3.],[0.,1.,4.]]),('probabilities',[.25,.75]),('amplitude_renormalization',True)]:
  q=copy.deepcopy(c);q['profile_policy'][key]=value;bad(lambda q=q:d.validate(q))
 sentinel=object();calls=[]
 def actual(path,dev):calls.append((path,dev));return 'OWN_FINAL'
 assert load('/p/final','cpu','/p/final','/r/final',sentinel,actual) is sentinel and not calls
 assert load('/r/final','cpu','/p/final','/r/final',sentinel,actual)=='OWN_FINAL' and len(calls)==1
 for path in ('/other/final','/r/latest','/p/best'):bad(lambda path=path:load(path,'cpu','/p/final','/r/final',sentinel,actual))
 cfg=Config(fs_hz=25000000.,fc_hz=2462000000.,processing_route='residual');hw=receiver_for_session(cfg,d.CHANNEL_SEEDS['receiver'],'synthetic')
 probe=ResidualChannel(cfg,1,hw).latent
 for shape in ([0,1,4],[0,0,1],[1,2,3],[],[0,1,float('nan')]):bad(lambda shape=shape:profile.delays(probe,shape))
 state=np.random.get_state();t=np.arange(256);n=0;states=set();max_rms_error=0.;iq_changed=0;status_changed=0;deltas=[]
 for scene in d.TRAIN_SCENES:
  cfg=replace(cfg,scenario=scene)
  for seed in range(1,13):
   for raw in (np.exp(.01j*t),np.exp(.9j*t)+.5*np.exp(-.4j*t),np.exp(1j*(.3*t+.001*t*t))):
    x,_=normalize_input(raw);saved=x.copy();a=ResidualChannel(cfg,seed,hw);b=ResidualChannel(cfg,seed,hw);r=ResidualChannel(cfg,seed,hw)
    original=profile.apply(r.latent,(0.,1.,3.));info=profile.apply(b.latent,(0.,1.,2.))
    for k in ('pdp','diffuse','mean_los_power','ray_coef','ray_hz'):
     assert np.array_equal(getattr(a.latent,k),getattr(b.latent,k)),k
    for k in ('reference_power','state','z','phase','tracking_z','residual_hz','coarse_doppler_error_hz','f_doppler','f_total'):assert getattr(a.latent,k)==getattr(b.latent,k),k
    assert asdict(a.latent.geometry)==asdict(b.latent.geometry) and a.receiver==b.receiver
    ya,ma,ta=a.process(x,return_traces=True);yb,mb,tb=b.process(x,return_traces=True);yr,mr,tr=r.process(x,return_traces=True)
    assert np.array_equal(x,saved) and np.array_equal(ya,yr) and ma==mr and np.array_equal(ta['propagation_kernel'],tr['propagation_kernel'])
    assert np.isfinite(yb).all() and not np.array_equal(a.latent.delays,b.latent.delays)
    for k in ('geometry','receiver','state_end','added_noise_variance_reference','numerical_filter_latency_samples','boundary','noise_boundary','channel_equalization_applied','tx_impairment_correction_applied'):assert ma[k]==mb[k],k
    assert ta['equalizer_kernel'].tolist()==tb['equalizer_kernel'].tolist()==[1.+0j] and not ma['tx_impairment_correction_applied']
    from leo_practical.channel import receiver_quality
    assert (mb['receiver_lock_status'],mb['residual_scale'])==receiver_quality(cfg,mb['quality_snr_db'])
    h=tb['propagation_kernel'];expected=10*np.log10(max(np.mean(abs(x)**2)*sum(abs(h)**2),1e-300)/10**(-cfg.reference_snr_db/10))
    assert np.isclose(expected,mb['quality_snr_db'],atol=1e-12)
    err=max(abs(np.asarray(info['ensemble_RMS_ns'])-np.asarray(cfg.rms_delay_ns)));max_rms_error=max(max_rms_error,float(err));iq_changed+=not np.array_equal(ya,yb);status_changed+=ma['receiver_lock_status']!=mb['receiver_lock_status'];deltas.append(mb['quality_snr_db']-ma['quality_snr_db']);states.add(ma['state_end']);n+=1
 after=np.random.get_state();assert state[0]==after[0] and np.array_equal(state[1],after[1]) and state[2:]==after[2:]
 assert states=={'LOS','PARTIAL','BLOCKED'} and iq_changed==n
 # Exercise the actual training adapter, including cumulative offsets and metadata.
 import torch
 from .channel import Practical
 torch.set_num_threads(2)
 adapter=Practical();meta={k:[] for k in ('eq_i','rx_i','day_i','sig_i','base_index')}
 for rx in d.FIT:
  for day in (1,2,3):
   for k,v in zip(meta,(1,rx,day,0,len(meta['base_index']))):meta[k].append(v)
 xx=torch.tensor([[np.cos(.2*t).tolist(),np.sin(.2*t).tolist()] for _ in meta['eq_i']],dtype=torch.float32)
 xx0=xx.clone();out,records=adapter.apply(xx,meta,epoch=21,step=4441)
 assert torch.equal(xx,xx0) and out.shape==xx.shape and len(records)==15 and sum(adapter.profile_counts.values())==15 and adapter.profile_calls==15 and len(adapter.sessions)==15
 assert set(adapter.profile_counts)=={0,1}
 other=Practical();again,rr=other.apply(xx,meta,epoch=21,step=4441)
 assert torch.equal(out,again) and records==rr and adapter.summary()==other.summary()
 for ep,step in ((20,4441),(21,4440),(22,4441),(31,6661)):
  bad(lambda ep=ep,step=step:adapter.apply(xx,meta,epoch=ep,step=step))
 for key,val in (('eq_i',0),('rx_i',2),('day_i',4)):
  mm=copy.deepcopy(meta);mm[key][0]=val;bad(lambda mm=mm:adapter.apply(xx,mm,epoch=21,step=4441))
 for role in ('U','eval','test'):bad(lambda role=role:Practical(role))
 from . import model as api
 m=api.build(c,'cpu');m.eval()
 with torch.no_grad():yy=m.forward_identity_only(torch.zeros(2,2,256))
 assert yy['tx_logits'].shape==(2,6) and yy['id_feat_joint'].shape==(2,160) and torch.isfinite(yy['tx_logits']).all()
 assert sum(p.numel() for p in m.parameters())==900722
 return dict(status='PASS',protocol_negatives=negative,same_latent_cases=n,complete_residual_calls=3*n,states=sorted(states),max_ensemble_RMS_error_ns=max_rms_error,original_profile_waveform_metadata_bitwise=True,changed_IQ_cases=iq_changed,receiver_status_changes=status_changed,synthetic_quality_delta_dB_quantiles=np.quantile(deltas,[0,.1,.5,.9,1]).tolist(),power_PDP_geometry_RX_noise_prior_preserved=True,quality_recomputed_from_changed_H=True,amplitude_renormalization=False,same_reflect_and_common24_latency=True,global_numpy_RNG_unchanged=True,actual_adapter_cases=30,actual_adapter_shape_counts=dict(adapter.profile_counts),actual_F_parameters=900722,real_IQ=0,CP=0,NN=2,updates=0,identity_benefit_proved=False,REJECTED_EXTRA_GATE='Separate design approval NONBLOCKING under explicit minimal project workflow')

if __name__=='__main__':
 sys.path.insert(0,str(d.ROOT/'experiments/adv3b02_xuc/code'));print(json.dumps(run()))
