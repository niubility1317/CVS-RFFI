"""Training-only propagation dependence; original marginal physics and evaluation."""
from copy import deepcopy
import numpy as np
import torch
from experiments.cvs_crossrx_satmix20.channel import PairedChannel
from experiments.cvs_crossrx_satmix20 import design as original
POLICIES=('IID','SCENE_GROUP','PROP_GROUP');FIT=(1,3,4,8)
PROPAGATION=('geometry','probs','state','z','ray_hz','ray_coef','f_doppler')

def mix_propagation(personal,environment):
 """Frozen snapshots only. Keep own receiver, noise, phase, tracking and sync."""
 if personal.used or environment.used or personal.cfg!=environment.cfg:raise ValueError('Unused matching native snapshots required')
 a,b=personal.latent,environment.latent
 for name in PROPAGATION:setattr(a,name,deepcopy(getattr(b,name)))
 a.f_total=a.f_doppler+personal.receiver.oscillator_cfo_hz;a.f_hat=a.f_total-a.residual_hz
 return personal

class GroupedPractical(PairedChannel):
 def __init__(self,policy):
  if policy not in POLICIES:raise ValueError('Fixed IID/scene/propagation policy only')
  super().__init__();self.policy=policy
 def __deepcopy__(self,memo):
  value=type(self)(self.policy);memo[id(self)]=value;return value
 def apply(self,x,meta,*,namespace,scene=None):
  if namespace.startswith('source_eval/'):
   return super().apply(x,meta,namespace=namespace,scene=scene)
  if not namespace.startswith('train/epoch') or scene is not None:raise ValueError('Training namespace only,no scene override')
  if x.ndim!=3 or x.shape[1:]!=(2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):raise ValueError('Finite raw float32 IQ256')
  keys=('rx_i','day_i','eq_i','sig_i','base_index')
  if any(k not in meta or len(meta[k])!=len(x) for k in keys):raise ValueError('Physical fit-L metadata required')
  values={k:meta[k].detach().cpu().tolist() if hasattr(meta[k],'detach') else list(meta[k]) for k in keys}
  if any(int(r) not in FIT for r in values['rx_i']) or any(int(v) not in (1,2,3) for v in values['day_i']) or any(int(v)!=1 for v in values['eq_i']):raise ValueError('Fixed fit RX1/3/4/8,day1/2/3,eq1 only')
  raw=np.asarray(x.detach().cpu().tolist(),dtype=np.float64);outputs=[];details=[];seed0=original.CHANNEL_SEED
  for i in range(len(x)):
   pid=tuple(int(values[k][i]) for k in keys);group=pid[:3];session='rx%d:day%d:eq%d'%group
   selection=pid if self.policy=='IID' else group
   rng=np.random.default_rng(self.ch.stable_seed(seed0,namespace,'scene',selection));selected=original.SCENES[int(rng.integers(len(original.SCENES)))];cfg=self.configs[selected]
   seed=self.ch.stable_seed(seed0,namespace,selected,pid);env_seed=self.ch.stable_seed(seed0,namespace,selected,'environment',group);receiver=self.ch.receiver_for_session(cfg,original.RECEIVER_SEED,session)
   personal=self.residual.ResidualChannel(cfg,seed,receiver);environment=self.residual.ResidualChannel(cfg,env_seed,receiver)
   if self.policy=='PROP_GROUP':mix_propagation(personal,environment)
   # All arms construct both native snapshots and call process once. Only the
   # selected propagation dependence differs; no full waveform or weakened EQ.
   private=dict(packet_residual_hz=personal.latent.residual_hz,packet_coarse_error_hz=personal.latent.coarse_doppler_error_hz,packet_phase=personal.latent.phase,packet_tracking_z=personal.latent.tracking_z)
   record,_=self.ch.normalize_input(raw[i,0]+1j*raw[i,1]);out,m=personal.process(record)
   if not np.isfinite(out).all():raise FloatingPointError('Invalid physical output;never drop sample')
   outputs.append(np.stack((out.real,out.imag)));details.append(dict(scene=selected,seed=seed,session=session,RX=pid[0],day=pid[1],state=m['state_end'],lock=m['receiver_lock_status'],quality_snr_db=m['quality_snr_db'],signal_to_added_noise_db=m['signal_to_added_noise_db'],agc_limited=m['agc_limited'],policy=self.policy,environment_group=list(group),environment_seed=env_seed,packet_seed=seed,noise_stream_seed=self.ch.stable_seed(seed,'noise'),kernel_energy=m['residual_kernel_energy'],**private))
  array=np.ascontiguousarray(outputs,dtype=np.float32);result=torch.frombuffer(bytearray(array.tobytes()),dtype=torch.float32).reshape(array.shape).to(x.device)
  return result,details
