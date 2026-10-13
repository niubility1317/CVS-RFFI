"""Read actual Sinc coordinate/physical/EMA/AdamW saved states; never fit."""
import copy,json
from types import SimpleNamespace
from pathlib import Path
import torch
from . import design as d
from .model import PREFIX,installed
from .predict import source_proof
from experiments.cvs_feature_diagnostics.sinc_change import compare
from experiments.cvs_feature_diagnostics.sinc_optimizer import inspect_state

def physical(state,mode):
 keys=['low_coordinate','band_coordinate','low_origin_hz','band_origin_hz','t_','window_']
 if any(PREFIX+k not in state for k in keys):raise ValueError('Missing Sinc coordinate state')
 s={k:state[PREFIX+k].detach().cpu() for k in keys}
 if any(s[k].shape!=(24,1) for k in keys[:4]) or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in s.values()):raise ValueError('Invalid Sinc coordinate state')
 if mode not in ('RAW_HZ','CENTER_25K'):raise ValueError('Unexpected coordinate mode')
 low=s['low_coordinate'] if mode=='RAW_HZ' else s['low_origin_hz']+25000*s['low_coordinate']
 band=s['band_coordinate'] if mode=='RAW_HZ' else s['band_origin_hz']+25000*s['band_coordinate']
 return {PREFIX+k:v for k,v in dict(low_hz_=low,band_hz_=band,t_=s['t_'],window_=s['window_']).items()}

def optimizer(model,checkpoint,native,mode):
 # Only translate the two parameter names in a read-only view. The original
 # native collector and saved full parameter order/shape verification are used.
 names=list(model.named_parameters());mapping={PREFIX+'low_coordinate':PREFIX+'low_hz_',PREFIX+'band_coordinate':PREFIX+'band_hz_'}
 proxy=SimpleNamespace(parameters=model.parameters,named_parameters=lambda:[(mapping.get(n,n),p) for n,p in names])
 report=inspect_state(proxy,checkpoint,native)
 result={}
 for old,name in [('low_hz_','low_coordinate'),('band_hz_','band_coordinate')]:
  r=report['Sinc'][old]
  if mode=='CENTER_25K':
   r={k.replace('_Hz','_coordinate'):v for k,v in r.items()}
   r['coordinate_units']='dimensionless offsets; physical rawHz=originHz+25000coordinate'
   r['physical_Hz_adaptive_term_unquantized_max']=25000*r['final_moment_adaptive_term_coordinate_max']
   r['physical_Hz_adaptive_term_unquantized_per_channel']=[25000*v for v in r['per_channel_final_moment_adaptive_term_coordinate']]
   r['adaptive_term_below_half_ULP_channels_scope']='coordinate float32 ULP, not reconstructed physical Hz ULP'
  else:r['coordinate_units']='raw Hz'
  r['actual_parameter_name']=PREFIX+name;result[name]=r
 report['Sinc']=result;report['read_only_name_translation']=mapping
 return report

def row(c):
 if c['source_config'] is not None:raise ValueError('Only new coordinate source rows')
 source=source_proof(c);mode=c['arm']
 from experiments.cvs_phase1_overlay.model import native_modules
 native_modules()
 initial=torch.load(source/'initial_smoke.pt',map_location='cpu',weights_only=False)['model']
 ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['epoch']!=50 or ck['a1_ema_successful_updates']!=11100 or ck['run_id']!=d.RUN or ck['candidate_id']!=c['row_id'] or ck['checkpoint_selection']!='final_only':raise ValueError('Actual own final checkpoint identity differs')
 with installed(c) as native:
  model=native.build_baseline_model(SimpleNamespace(**ck['baseline_args']),torch.device('cpu'))
  model.load_state_dict(ck['model'],strict=True);before={k:v.clone() for k,v in model.state_dict().items()}
  stats=optimizer(model,ck,native,mode)
  SincType=model.id_backbone.encoder.core.id_backbone.sinc._native_type
 states=[initial,ck['model'],ck['ema_model']]
 for state in states[1:]:
  if any(not torch.equal(initial[PREFIX+k],state[PREFIX+k]) for k in ('low_origin_hz','band_origin_hz','t_','window_')):raise ValueError('Fixed Sinc origin/window buffers changed')
 converted=[physical(state,mode) for state in states]
 student=compare(converted[0],converted[1],SincType);teacher=compare(converted[0],converted[2],SincType);gap=compare(converted[1],converted[2],SincType)
 epoch=d.read(source/'source_Sinc_epoch_diagnostic.json')
 if len(epoch['records'])!=50 or student!=epoch['records'][-1]['Sinc']:raise ValueError('Saved final kernel differs from actual source epoch record')
 coordinates=[]
 for name in ('low_coordinate','band_coordinate'):
  a=initial[PREFIX+name];b=ck['model'][PREFIX+name];t=ck['ema_model'][PREFIX+name]
  coordinates.append(dict(parameter=PREFIX+name,changed_elements=int((a!=b).sum()),maximum_abs_coordinate_change=float((b-a).abs().max()),student_initial=a.flatten().tolist(),student_final=b.flatten().tolist(),teacher_final=t.flatten().tolist(),teacher_student_maximum_abs_difference=float((b-t).abs().max())))
 if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Read-only diagnostic changed model state')
 return dict(row_id=c['row_id'],arm=mode,seed=c['model_seed'],status='VERIFIED',epochs=50,updates=11100,source_roles='EXACT_MATCH',ancestors=[],optimizer=stats,coordinates=coordinates,initial_to_student=student,initial_to_EMA=teacher,student_to_EMA=gap,actual_final_epoch_record_matches=True,model_parameters_buffers_unchanged=True,fixed_buffers_unchanged=True,IQ_read=False,query_read=False,head_updates=0,optimizer_updates_by_diagnostic=0)

def checks():
 from .model import build
 c=d.config(d.training_rows()[1]);torch.manual_seed(c['model_seed']);model=build(c)
 state=model.state_dict();out=physical(state,'CENTER_25K')
 assert torch.equal(out[PREFIX+'low_hz_'],state[PREFIX+'low_origin_hz'])
 bad=copy.deepcopy(state);del bad[PREFIX+'low_origin_hz']
 try:physical(bad,'CENTER_25K')
 except ValueError:pass
 else:raise AssertionError('Incomplete actual state accepted')
 with installed(c) as native:
  parameters=native._optimizer_parameters(model,None)
  states={i:dict(step=torch.tensor(11100.),exp_avg=torch.full_like(p,1e-4),exp_avg_sq=torch.full_like(p,1e-8)) for i,p in enumerate(parameters)}
  ck=dict(args=dict(freeze_backbone=False,use_muse_ssdg=False),optimizer=dict(state=states,param_groups=[dict(params=list(states),betas=(.9,.999),eps=1e-8,lr=1e-6,weight_decay=.0001)]),a1_ema_successful_updates=11100)
  result=optimizer(model,ck,native,'CENTER_25K')
  assert result['actual_optimizer_parameter_count']==len(parameters)
  for name,r in result['Sinc'].items():
   assert r['actual_parameter_name']==PREFIX+name and r['actual_step']==11100
   assert r['physical_Hz_adaptive_term_unquantized_max']==25000*r['final_moment_adaptive_term_coordinate_max']
 return dict(status='PASS',synthetic_only=True,no_real_checkpoint_or_IQ_read=True,physical_coordinate_mapping_exact=True,saved_full_order_and_unit_translation_checked=True,negative_checks=1,optimizer_steps=0,head_fit=False)
if __name__=='__main__':print(json.dumps(checks()))
