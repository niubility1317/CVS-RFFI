"""Same latent/noEQ source counterfactual;actual formal IQ never changes."""
from dataclasses import asdict
import numpy as np
from . import design as d
from .protocol import validate_config
class Generator:
 def __init__(self):
  import leo_practical.channel as ch
  import leo_practical.residual as residual
  from experiments.cvs_fullsource_residual_diag24.controls import render
  self.ch=ch;self.residual=residual;self.render=render;self.native_calls=0;self.render_calls=0;self.sessions={};self.max_parity_error=0.
  audit=d.read(d.PARENT/'B0-s2026092701/stage20/actual_channel.json')
  if ch.__file__!=audit['channel_module_file'] or residual.__file__!=audit['residual_module_file']:raise ValueError('Actual training leo_practical __file__ differs')
  self.configs={}
  for scene in d.SCENES:
   v=audit['configurations'][scene];validate_config(v);self.configs[scene]=ch.Config(**v)
 def apply(self,x,pid,rx,day,draw,scene):
  if rx not in d.RX or day not in d.DAYS or draw not in (0,1) or scene not in d.SCENES:raise ValueError('Legal original L source context only')
  cfg=self.configs[scene];session=f'rx{rx}:day{day}:eq1';hw=self.ch.receiver_for_session(cfg,d.HARDWARE_SEED,session);hardware=asdict(hw)
  if self.sessions.setdefault(session,hardware)!=hardware:raise ValueError('Session RX hardware changed')
  seed=self.ch.stable_seed(d.CHANNEL_SEED,'L-only/noise-gradient40/draw'+str(draw),scene,pid)
  xx,_=self.ch.normalize_input(np.asarray(x,complex));baseline,meta,traces=self.residual.ResidualChannel(cfg,seed,hw).process(xx,return_traces=True);self.native_calls+=1
  rendered,details=self.render(xx,cfg,seed,hw);self.render_calls+=1
  frozen=dict(meta,_equalizer=traces['equalizer_kernel']);replayed,_=self.render(xx,cfg,seed,hw,frozen=frozen);self.render_calls+=1
  for v in [rendered,replayed]:
   self.max_parity_error=max(self.max_parity_error,float(np.max(abs(v-baseline))))
   if not np.allclose(v,baseline,atol=2e-6,rtol=2e-6):raise ValueError('Controlled renderer/native residual parity differs')
  if meta['channel_equalization_applied'] or len(traces['equalizer_kernel'])!=1 or not np.allclose(traces['equalizer_kernel'],[1.]):raise ValueError('Original EQoff must giveG1')
  values={'full':baseline};records={'full':details}
  for name,state in [('no_noise/fixed',frozen),('no_noise/total',None)]:
   values[name],records[name]=self.render(xx,cfg,seed,hw,control='no_noise',frozen=state);self.render_calls+=1
  for k in ['receiver_lock_status','residual_scale','estimated_frequency_hz','output_frequency_hz','phase_tracking_applied','channel_equalization_applied','agc_gain_db']:
   if records['no_noise/fixed'][k]!=meta[k]:raise ValueError('Fixed control changed actual receiver/compensation '+k)
  if not all(np.isfinite(v).all() for v in values.values()):raise ValueError('No nonfinite sample may be dropped')
  return values,{k:{n:v for n,v in r.items() if not n.startswith('_')} for k,r in records.items()}
 def audit(self):return dict(native_module_files=dict(channel=self.ch.__file__,residual=self.residual.__file__),actual_configs={s:asdict(c) for s,c in self.configs.items()},native_calls=self.native_calls,controlled_renderer_calls=self.render_calls,stable_RX_day_EQ_sessions=self.sessions,maximum_native_replay_absolute_error=self.max_parity_error,formal_test_changed=False,original_recording_noise_removed=False,newly_added_noise_removed_in_controls_only=True,control_quality200_not_total_SNR=True)
