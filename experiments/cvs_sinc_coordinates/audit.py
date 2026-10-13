import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,json,gzip,math,time
from pathlib import Path
import numpy as np
import torch
release=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_sinc_coordinates_20261014_r01')
sys.path[:0]=[str(release),str(release/'code')]
from experiments.cvs_sinc_coordinates import design as d
from experiments.cvs_sinc_coordinates.predict import source_proof,preflight,baseline
from experiments.cvs_report_ssl.evaluate import truth_arrays
from experiments.cvs_receiver_residual_capacity4.control import process
from experiments.cvs_phase1_overlay.model import native_modules
native_modules();torch.set_num_threads(2)
diag_reader={'__name__':'read_only_Sinc_analysis','__package__':'experiments.cvs_sinc_coordinates'}
exec("\"\"\"Read actual Sinc coordinate/physical/EMA/AdamW saved states; never fit.\"\"\"\nimport copy,json\nfrom types import SimpleNamespace\nfrom pathlib import Path\nimport torch\nfrom . import design as d\nfrom .model import PREFIX,installed\nfrom .predict import source_proof\nfrom experiments.cvs_feature_diagnostics.sinc_change import compare\nfrom experiments.cvs_feature_diagnostics.sinc_optimizer import inspect_state\n\ndef physical(state,mode):\n keys=['low_coordinate','band_coordinate','low_origin_hz','band_origin_hz','t_','window_']\n if any(PREFIX+k not in state for k in keys):raise ValueError('Missing Sinc coordinate state')\n s={k:state[PREFIX+k].detach().cpu() for k in keys}\n if any(s[k].shape!=(24,1) for k in keys[:4]) or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in s.values()):raise ValueError('Invalid Sinc coordinate state')\n if mode not in ('RAW_HZ','CENTER_25K'):raise ValueError('Unexpected coordinate mode')\n low=s['low_coordinate'] if mode=='RAW_HZ' else s['low_origin_hz']+25000*s['low_coordinate']\n band=s['band_coordinate'] if mode=='RAW_HZ' else s['band_origin_hz']+25000*s['band_coordinate']\n return {PREFIX+k:v for k,v in dict(low_hz_=low,band_hz_=band,t_=s['t_'],window_=s['window_']).items()}\n\ndef optimizer(model,checkpoint,native,mode):\n # Only translate the two parameter names in a read-only view. The original\n # native collector and saved full parameter order/shape verification are used.\n names=list(model.named_parameters());mapping={PREFIX+'low_coordinate':PREFIX+'low_hz_',PREFIX+'band_coordinate':PREFIX+'band_hz_'}\n proxy=SimpleNamespace(parameters=model.parameters,named_parameters=lambda:[(mapping.get(n,n),p) for n,p in names])\n report=inspect_state(proxy,checkpoint,native)\n result={}\n for old,name in [('low_hz_','low_coordinate'),('band_hz_','band_coordinate')]:\n  r=report['Sinc'][old]\n  if mode=='CENTER_25K':\n   r={k.replace('_Hz','_coordinate'):v for k,v in r.items()}\n   r['coordinate_units']='dimensionless offsets; physical rawHz=originHz+25000coordinate'\n   r['physical_Hz_adaptive_term_unquantized_max']=25000*r['final_moment_adaptive_term_coordinate_max']\n   r['physical_Hz_adaptive_term_unquantized_per_channel']=[25000*v for v in r['per_channel_final_moment_adaptive_term_coordinate']]\n   r['adaptive_term_below_half_ULP_channels_scope']='coordinate float32 ULP, not reconstructed physical Hz ULP'\n  else:r['coordinate_units']='raw Hz'\n  r['actual_parameter_name']=PREFIX+name;result[name]=r\n report['Sinc']=result;report['read_only_name_translation']=mapping\n return report\n\ndef row(c):\n if c['source_config'] is not None:raise ValueError('Only new coordinate source rows')\n source=source_proof(c);mode=c['arm']\n from experiments.cvs_phase1_overlay.model import native_modules\n native_modules()\n initial=torch.load(source/'initial_smoke.pt',map_location='cpu',weights_only=False)['model']\n ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)\n if ck['epoch']!=50 or ck['a1_ema_successful_updates']!=11100 or ck['run_id']!=d.RUN or ck['candidate_id']!=c['row_id'] or ck['checkpoint_selection']!='final_only':raise ValueError('Actual own final checkpoint identity differs')\n with installed(c) as native:\n  model=native.build_baseline_model(SimpleNamespace(**ck['baseline_args']),torch.device('cpu'))\n  model.load_state_dict(ck['model'],strict=True);before={k:v.clone() for k,v in model.state_dict().items()}\n  stats=optimizer(model,ck,native,mode)\n  SincType=model.id_backbone.encoder.core.id_backbone.sinc._native_type\n states=[initial,ck['model'],ck['ema_model']]\n for state in states[1:]:\n  if any(not torch.equal(initial[PREFIX+k],state[PREFIX+k]) for k in ('low_origin_hz','band_origin_hz','t_','window_')):raise ValueError('Fixed Sinc origin/window buffers changed')\n converted=[physical(state,mode) for state in states]\n student=compare(converted[0],converted[1],SincType);teacher=compare(converted[0],converted[2],SincType);gap=compare(converted[1],converted[2],SincType)\n epoch=d.read(source/'source_Sinc_epoch_diagnostic.json')\n if len(epoch['records'])!=50 or student!=epoch['records'][-1]['Sinc']:raise ValueError('Saved final kernel differs from actual source epoch record')\n coordinates=[]\n for name in ('low_coordinate','band_coordinate'):\n  a=initial[PREFIX+name];b=ck['model'][PREFIX+name];t=ck['ema_model'][PREFIX+name]\n  coordinates.append(dict(parameter=PREFIX+name,changed_elements=int((a!=b).sum()),maximum_abs_coordinate_change=float((b-a).abs().max()),student_initial=a.flatten().tolist(),student_final=b.flatten().tolist(),teacher_final=t.flatten().tolist(),teacher_student_maximum_abs_difference=float((b-t).abs().max())))\n if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Read-only diagnostic changed model state')\n return dict(row_id=c['row_id'],arm=mode,seed=c['model_seed'],status='VERIFIED',epochs=50,updates=11100,source_roles='EXACT_MATCH',ancestors=[],optimizer=stats,coordinates=coordinates,initial_to_student=student,initial_to_EMA=teacher,student_to_EMA=gap,actual_final_epoch_record_matches=True,model_parameters_buffers_unchanged=True,fixed_buffers_unchanged=True,IQ_read=False,query_read=False,head_updates=0,optimizer_updates_by_diagnostic=0)\n\ndef checks():\n from .model import build\n c=d.config(d.training_rows()[1]);torch.manual_seed(c['model_seed']);model=build(c)\n state=model.state_dict();out=physical(state,'CENTER_25K')\n assert torch.equal(out[PREFIX+'low_hz_'],state[PREFIX+'low_origin_hz'])\n bad=copy.deepcopy(state);del bad[PREFIX+'low_origin_hz']\n try:physical(bad,'CENTER_25K')\n except ValueError:pass\n else:raise AssertionError('Incomplete actual state accepted')\n with installed(c) as native:\n  parameters=native._optimizer_parameters(model,None)\n  states={i:dict(step=torch.tensor(11100.),exp_avg=torch.full_like(p,1e-4),exp_avg_sq=torch.full_like(p,1e-8)) for i,p in enumerate(parameters)}\n  ck=dict(args=dict(freeze_backbone=False,use_muse_ssdg=False),optimizer=dict(state=states,param_groups=[dict(params=list(states),betas=(.9,.999),eps=1e-8,lr=1e-6,weight_decay=.0001)]),a1_ema_successful_updates=11100)\n  result=optimizer(model,ck,native,'CENTER_25K')\n  assert result['actual_optimizer_parameter_count']==len(parameters)\n  for name,r in result['Sinc'].items():\n   assert r['actual_parameter_name']==PREFIX+name and r['actual_step']==11100\n   assert r['physical_Hz_adaptive_term_unquantized_max']==25000*r['final_moment_adaptive_term_coordinate_max']\n return dict(status='PASS',synthetic_only=True,no_real_checkpoint_or_IQ_read=True,physical_coordinate_mapping_exact=True,saved_full_order_and_unit_translation_checked=True,negative_checks=1,optimizer_steps=0,head_fit=False)\nif __name__=='__main__':print(json.dumps(checks()))\n",diag_reader)
records=[];rows=[];decisions={};total_steps=0;total_epochs=0;start=time.time()
analysis=d.read(d.BASE/'analysis.json');completion=d.read(d.BASE/'completion.json')
assert completion['status']=='EXPERIMENT_COMPLETE' and completion['models']==9 and completion['truth_last']
for row in d.rows():
 c=d.config(row);source=source_proof(c);preflight(row['row_id'])
 epochs=[json.loads(v) for v in (source/'epoch_metrics.jsonl').read_text().splitlines()]
 assert len(epochs)==50 and [v['epoch'] for v in epochs]==list(range(1,51))
 assert all(v['test_tx_total']==0 and v['test_tx_acc'] is None and v['val_tx_total']==27000 for v in epochs)
 total_epochs+=50;steps=0;u=accepted=0
 if c['source_config'] is None:
  with gzip.open(source/'step_metrics.jsonl.gz','rt') as stream:
   for line in stream:
    v=json.loads(line);steps+=1;m=v['metrics']
    assert v['step']==steps and v['epoch']==(steps-1)//222+1 and math.isfinite(m['train/loss'])
    assert m['train/skipped_nonfinite_loss']==0 and m['train/skipped_nonfinite_grad']==0
    assert m['train/a1_ema_successful_updates']==steps and abs(m['train/a1_ema_effective_decay']-.999)<1e-10
    expected=0 if steps<=33*222 else 124 if steps%222==0 else 256
    assert m['train/pseudo_total']==expected and m.get('train/pseudo_truth_available',0)==0
    u+=int(m['train/pseudo_total']);accepted+=int(m['train/pseudo_selected'])
  assert steps==11100 and u==963900;total_steps+=steps
  U=d.read(source/'source_U_epoch_diagnostic.json')
  assert set(map(int,U['epochs']))==set(range(34,51)) and U['U_truth_read'] is False and U['new_forward_calls']==0
  assert all(sum(v['count'] for v in group.values())==56700 for group in U['epochs'].values())
  assert sum(v['accepted'] for group in U['epochs'].values() for v in group.values())==accepted
  assert all(sum(v['predicted_class_counts'])==v['count'] and sum(v['accepted_predicted_class_counts'])==v['accepted'] for group in U['epochs'].values() for v in group.values())
  movement=d.read(source/'source_Sinc_epoch_diagnostic.json')
  assert [v['epoch'] for v in movement['records']]==list(range(1,51)) and movement['extra_V_passes']==2 and not movement['head_fit']
  assert all(v['source_V_RX_health']['source_val_count']==27000 for v in movement['records'] if v['epoch'] in (33,34))
  initial=torch.load(source/'initial_smoke.pt',map_location='cpu',weights_only=False)['model']
  ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False);changed=[]
  assert ck['epoch']==50 and ck['a1_ema_successful_updates']==11100 and ck['run_id']==d.RUN and ck['candidate_id']==row['row_id']
  assert ck['args']['from_scratch'] and not ck['args']['baseline_ckpt'] and not ck['args']['teacher_ckpt']
  for name,value in ck['model'].items():
   if name.endswith(('.sinc.low_coordinate','.sinc.band_coordinate')):
    assert torch.isfinite(value).all();difference=float((value-initial[name]).abs().max())
    if c['arm']=='CENTER_25K':assert difference>0
    changed.append(dict(parameter=name,maximum_abs_coordinate_update=difference,changed_elements=int((value!=initial[name]).sum())))
  assert len(changed)==2
  sinc=diag_reader['row'](c)
 else:
  steps=11100;U=None;movement=None;changed=[];sinc=None
  # Native epoch reducer stores pseudo counters as epoch sums, not step means.
  assert all(e.get('train_pseudo_total',0)==(0 if e['epoch']<=33 else 56700) for e in epochs)
  u=sum(int(e.get('train_pseudo_total',0)) for e in epochs)
  accepted=sum(int(e.get('train_pseudo_selected',0)) for e in epochs)
  assert u==963900
 metrics=d.read(d.BASE/row['row_id']/'scores/metrics.json')['records'];assert len(metrics)==18
 with np.load(d.BASE/row['row_id']/'prediction/predictions.npz',allow_pickle=False) as saved:
  ids=saved['ids'];pred=saved['ORIGINAL_HEAD'];assert len(ids)==168000 and len(np.unique(ids))==168000 and set(saved.files)=={'ids','ORIGINAL_HEAD'}
  y,rx,day=truth_arrays(d.read(d.TRUTH),ids);decisions[row['arm'],row['model_seed']]=pred.copy()
  if c['source_config'] is not None:assert np.array_equal(pred,baseline(c,ids))
  for v in metrics:
   mask=np.ones(len(y),dtype=bool) if v['dimension']=='overall' else rx==v['stratum'] if v['dimension']=='receiver' else day==v['stratum'] if v['dimension']=='day' else y==list(d.CLASSES).index(v['stratum'])
   cm=np.bincount(y[mask]*6+pred[mask],minlength=36).reshape(6,6);n=int(cm.sum());accuracy=float(np.trace(cm)/n);den=cm.sum(0)+cm.sum(1)
   f1=float(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean())
   assert v['confusion']==cm.tolist() and v['query_count']==n and v['accuracy']==accuracy and v['macro_f1']==f1
 records.extend(metrics)
 keys=('epoch','epoch_time_s','val_tx_acc','train_loss_tx_labeled','train_pseudo_total','train_pseudo_selected','train_pseudo_truth_available','train_a1_ema_successful_updates','train_reliable_ratio','train_pseudo_conf','train_strong_pass','train_temporal_pass')
 rows.append(dict(row_id=row['row_id'],arm=row['arm'],seed=row['model_seed'],source_reused=c['source_config'] is not None,epochs=50,updates=steps,U_packets=u,U_accepted=accepted,updated_Sinc_coordinates=changed,source_epoch_curve=[{k:e[k] for k in keys if k in e} for e in epochs],source_U_diagnostic=U,source_Sinc_diagnostic=movement,saved_Sinc_state_analysis=sinc))
errors=[]
for arm,control in d.PAIRS:
 for seed in d.SEEDS:
  good=decisions[arm,seed]==y;old=decisions[control,seed]==y
  errors.append(dict(arm=arm,control=control,seed=seed,corrected_errors=int((good&~old).sum()),introduced_errors=int((~good&old).sum()),both_correct=int((good&old).sum()),both_wrong=int((~good&~old).sum()),count=len(y)))
# Actual same-seed initial states are checked for all three seeds, without a file hash.
equal=[]
for seed in d.SEEDS:
 states=[torch.load(Path(d.config(r)['output_root'])/'initial_smoke.pt',map_location='cpu',weights_only=False)['model'] for r in d.training_rows() if r['model_seed']==seed]
 assert len(states)==2
 prefix='id_backbone.encoder.core.id_backbone.sinc.'
 differing={prefix+'low_coordinate',prefix+'band_coordinate'}
 assert set(states[0])==set(states[1]) and all(torch.equal(v,states[1][k]) for k,v in states[0].items() if k not in differing)
 physical=[diag_reader['physical'](state,mode) for state,mode in zip(states,('RAW_HZ','CENTER_25K'))]
 assert all(torch.equal(v,physical[1][k]) for k,v in physical[0].items())
 equal.append(dict(seed=seed,models=2,identical_noncoordinate_state_tensors=len(states[0])-2,identical_actual_initial_physical_Sinc=True))
queue=d.read(d.BASE/'queue_state.json');launch=d.read(d.BASE/'launch.json')
assert not queue['active'] and not queue['pending'] and not queue['failures'] and not any(p and p['cwd']==str(release) and p['argv']==r['argv'] for r in launch['rows'] for p in [process(r['pid'])])
assert total_steps==66600 and total_epochs==450 and len(records)==162
print(json.dumps(dict(status='VERIFIED',run_id=d.RUN,executing_code=(release/'release_commit.txt').read_text().strip(),analysis=analysis,rows=rows,error_transitions=errors,actual_initial_equality=equal,new_step_records_parsed=total_steps,epoch_records_parsed=total_epochs,prediction_count=1512000,metric_records_independently_recounted=162,all_workers_terminal=True,head_fit=False,head_updates=0,seconds=time.time()-start)))
