"""Read actual engineering configuration; compute short-window bounds only.

No imports of native runtime, NumPy, Torch, model, channel generation or data.
Bounds describe this engineering proxy, never measured identity performance.
"""
import argparse
import ast
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
NATIVE=ROOT/'experiments/adv3b02_xuc/code/leo_practical'
ADAPTER=ROOT/'experiments/cvs_pdp_shape30/channel.py'

def config_defaults(text):
 tree=ast.parse(text)
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Config')
 return {n.target.id:ast.literal_eval(n.value) for n in cls.body
         if isinstance(n,ast.AnnAssign) and n.value is not None}

def audit():
 channel=(NATIVE/'channel.py').read_text(encoding='utf-8-sig')
 residual=(NATIVE/'residual.py').read_text(encoding='utf-8-sig')
 adapter=ADAPTER.read_text(encoding='utf-8-sig')
 defaults=config_defaults(channel)
 calls=[n for n in ast.walk(ast.parse(adapter)) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and n.func.attr=='Config']
 if len(calls)!=1:raise ValueError('Actual training adapter Config call ambiguous')
 kw={k.arg:ast.literal_eval(k.value) for k in calls[0].keywords if k.arg in ('fs_hz','fc_hz','processing_route','mode','equalization_enabled')}
 if kw.get('processing_route')!='residual' or kw.get('mode')!='post_sync' or kw.get('equalization_enabled') is not False:raise ValueError('Actual residual/noEQ context changed')
 if 'latent.ray_coef.sum(axis=1)' not in residual or 'latent.ray_hz' in residual:raise ValueError('Frozen-scatter assumption changed; re-inspect before interpreting')
 if 'fd * np.cos' not in channel or 'cfg.terminal_speed_mps * cfg.fc_hz / C' not in channel:raise ValueError('Actual scatter law changed')
 constants={}
 for n in ast.parse(channel).body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='C' for t in n.targets):constants['C']=ast.literal_eval(n.value)
 c=constants['C'];fs=kw['fs_hz'];fc=kw['fc_hz'];n=256
 speed=defaults['terminal_speed_mps'];fd=defaults['scatter_doppler_hz']
 fd=speed*fc/c if fd is None else fd
 span=(n-1)/fs;extent=n/fs
 phase=2*math.pi*fd*span
 tau=defaults['shadow_corr_time_s'] or defaults['shadow_corr_distance_m']/speed
 state_refresh=-math.expm1(-extent/defaults['state_mix_time_s'])
 stationary_shadow_delta=math.sqrt(-2*math.expm1(-extent/tau))
 max_shadow_sigma=max(defaults['los_std_db'])
 states=[];a=math.log(10)/20
 for s,name in enumerate(('LOS','PARTIAL','BLOCKED')):
  diffuse=10**(defaults['diffuse_power_db'][s]/10)
  mean_los=0 if s==2 else math.exp(2*a*defaults['los_mean_db'][s]+2*a*a*defaults['los_std_db'][s]**2)
  fraction=diffuse/(mean_los+diffuse)
  states.append(dict(state=name,ensemble_diffuse_fraction=fraction,
   coefficient_vector_ensemble_relative_RMS_bound=phase*math.sqrt(fraction)))
 # A conservative scale screen for a candidate rationale, not an identity gate.
 screen=dict(max_ray_rotation_below_1mrad=phase<.001,
             packet_state_refresh_probability_below_1e_4=state_refresh<1e-4,
             stationary_shadow_change_RMS_below_0_01dB=max_shadow_sigma*stationary_shadow_delta<.01)
 return dict(schema='F_FROZEN_SCATTER_TIMESCALE_CODE_ONLY_V1',
  status='COMPLETED_CODE_ONLY_NOT_IDENTITY_TESTED',
  native_channel='experiments/adv3b02_xuc/code/leo_practical/channel.py',
  native_residual='experiments/adv3b02_xuc/code/leo_practical/residual.py',
  configuration_source='actual training adapter explicit fs/fc/route plus current native Config defaults',
  config=kw,IQ_length=n,sample_span_s=span,packet_extent_s=extent,
  terminal_speed_mps=speed,scatter_max_doppler_hz=fd,
  max_ray_rotation_rad=phase,max_ray_rotation_deg=math.degrees(phase),
  max_unit_ray_complex_change=2*math.sin(phase/2),
  state_refresh_probability_upper_bound=state_refresh,
  shadow_correlation_time_s=tau,stationary_shadow_change_RMS_dB=max_shadow_sigma*stationary_shadow_delta,
  coefficient_vector_bounds=states,screen=screen,
  fixed_hypothesis='Omitted within-packet scatter/state/shadow dynamics already have material scale under existing nominal training prior',
  conclusion='CLOSED_INSUFFICIENT_NOMINAL_PHYSICAL_MAGNITUDE' if all(screen.values()) else 'NEEDS_SOURCE_IQ_EVIDENCE_NOT_AUTOMATIC_TRAINING',
  limitations=['No actual identity margins or accuracy measured;this cannot prove zero classifier sensitivity',
  'Ensemble coefficient-vector bound is not a uniform relative IQ bound in near-null channels',
  'No extra scatter Doppler is justified by already compensated common orbital Doppler',
  'No speed/strength tuning,source RX holdout,new data,neural forward or training is authorized by this calculation'],
  source_RX_holdout=None,real_IQ_reads=0,CP_loads=0,channel_process_calls=0,NN=0,
  raw_truth_reads=0,optimizer_updates=0,new_model_epochs=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
 result=audit();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
 print(json.dumps(result,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
