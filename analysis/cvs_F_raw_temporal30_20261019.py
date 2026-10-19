"""Complete source replay and optional independent frozen formal prediction recount."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.cvs_raw_temporal30 import design as d
from experiments.cvs_raw_temporal30.publish import remote
SCRIPT=r'''
import gzip,json,math,sys
from pathlib import Path
import numpy as np
import torch
c=CONFIG;project=Path(c['project']);run=project/'runs'/c['run_id'];bank=project/'runs'/c['bank_run']/'observation_bank';control=project/'runs'/c['control_run'];sys.path.insert(0,c['release'])
from experiments.cvs_raw_temporal30.runner import bootstrap
bootstrap()
def read(p):return json.loads(p.read_text())
decision=read(run/'source_matrix_decision.json');stream=np.load(bank/'indices.npy',allow_pickle=False);meta=read(bank/'actual_first_training_metadata.json');scenes=np.asarray([v['scene'] for v in meta]);reuse=read(run/'observation_bank_reuse.json')
if decision['formal_scores_read'] or decision['current_cumulative_epochs']!=30 or reuse['new_physical_simulations'] or Path(reuse['source_bank'])!=bank:raise ValueError('Fixed source-only/shared observation plan differs')
arms=['RAW_NoAs','RAW_As','PAIR_NoAs','PAIR_As'];head='id_backbone.encoder.core.id_backbone.cls_head.weight';prefix='id_backbone.encoder.core.id_backbone.raw_temporal.';source={};logs={};states={};source_pairs={};first=None;initial_diff={}
for arm in arms:
 out=run/(arm+'-s2026092701');cfg=read(out/'source/resolved_config.json');done=read(out/'source/completion.json');before=read(out/'source/initial_source_readout/complete.json');after=read(out/'source_qualification/complete.json');names=set(cfg['trainable_names'])
 if done['epoch']!=30 or done['optimizer_steps']!=2220 or done['U_used'] or done['EMA_used'] or done['default_head_frozen'] or not done['default_head_joint'] or done['head_joint_optimizer_steps']!=2220 or done['raw_supplement_training_NN']!=568320 or done['student_training_NN']!=568320 or done['parent_reference_NN']!=284160 or done['head_fit_updates'] or before['NN']!=50400 or after['NN']!=50400 or done['total_parameters']!=902974 or done['trainable_parameters']!=3212:raise ValueError('Actual complete raw budget/head/source metadata differs')
 if len(names)!=6 or head not in names or set(names)!={head,*[prefix+n+'.weight' for n in ('down','mix','temporal','B1','B2')]}:raise ValueError('Original W and exact2252 raw feature parameters required')
 values=[];error=0.;squared={n:0. for n in names}
 with gzip.open(out/'source/step_metrics.jsonl.gz','rt') as f:
  for i,line in enumerate(f):
   v=json.loads(line);grads=v['gradient_squared_by_parameter'];error=max(error,abs(v['total_loss']-v['clean_CE']-v['second_CE']-v['keep_KL']))
   if v['step']!=i+1 or v['epoch']!=1+i//222 or v['cumulative_epoch']!=21+i//222 or v['L_indices']!=stream[i].tolist() or v['bank_step']!=i+1 or v['As_consumed']!=arm.endswith('_As') or v['U_used'] or v['head_frozen'] or not v['head_joint'] or v['student_NN']!=256 or v['parent_NN']!=128 or v['raw_supplement_NN']!=256 or set(grads)!=names or not all(math.isfinite(g) and g>=0 for g in grads.values()):raise ValueError('Complete raw/source/gradient stream differs')
   for n,g in grads.items():squared[n]+=g
   values.append(v)
 if len(values)!=2220 or error>2e-5 or not all(g>0 for g in squared.values()):raise ValueError('All actual updates/three-term objective/inner gradients required')
 ini=torch.load(out/'source/initialization.pt',map_location='cpu',weights_only=False)['model'];fin=torch.load(out/'source/final_ssdg.pth',map_location='cpu',weights_only=False)['model'];changed=[n for n in ini if not torch.equal(ini[n],fin[n])]
 if set(changed)-names or head not in changed or any(prefix+n+'.weight' not in changed for n in ('B1','B2')):raise ValueError('Original body must freeze while real features and W change')
 states[arm]=dict(changed_parameters=changed,all_native_F_state_except_W_bit_equal=True,original_W_jointly_changed=True,both_feature_projections_changed=True,all_raw_encoder_parameters_have_nonzero_gradients=True,gradient_squared_sum=squared,total_parameters=902974,trainable_parameters=3212)
 logs[arm]=dict(actual_steps=2220,shared_L_and_As_bank_indices_equal=True,max_loss_sum_abs_error=error,wall_seconds=done['wall_seconds'],curves=[dict(epoch=e+1,**{k:float(np.mean([v[k] for v in values[e*222:(e+1)*222]])) for k in ('clean_CE','second_CE','keep_KL','total_loss','gradient_norm')}) for e in range(10)])
 byscene={};sp={}
 with np.load(out/'source/initial_source_readout/predictions.npz',allow_pickle=False) as a,np.load(out/'source_qualification/predictions.npz',allow_pickle=False) as b:
  if not np.array_equal(a['L_ids'],b['L_ids']) or not np.array_equal(a['L_ids'],np.asarray([v['physical_id'] for v in meta])):raise ValueError('Actual As diagnostic physical ID ordering differs')
  current={k:a[k].copy() for k in ('L_clean','L_actual_training_As','L_new','V_clean','V_satellite','V_full_clean')}
  if first is None:first=current
  initial_diff[arm]=max(float(np.abs(current[k]-first[k]).max()) for k in current)
  y=a['L_TX'];pa=a['L_actual_training_As'].argmax(-1);pb=b['L_actual_training_As'].argmax(-1)
  for scene in sorted(set(scenes)):
   mask=scenes==scene;t=y[mask];p=pa[mask];q=pb[mask];byscene[scene]=dict(count=int(mask.sum()),initial_accuracy=float((p==t).mean()),final_accuracy=float((q==t).mean()),repair=int(((q==t)&(p!=t)).sum()),damage=int(((q!=t)&(p==t)).sum()))
  for key,truthkey in [('L_clean','L_TX'),('L_actual_training_As','L_TX'),('L_new','L_TX'),('V_clean','V_TX'),('V_satellite','V_TX'),('V_full_clean','V_full_TX')]:
   p=a[key].argmax(-1);q=b[key].argmax(-1);t=a[truthkey];sp[key]=dict(count=int(q.size),repair=int(((q==t)&(p!=t)).sum()),damage=int(((q!=t)&(p==t)).sum()),initial_accuracy=float((p==t).mean()),final_accuracy=float((q==t).mean()))
  if abs(sp['V_full_clean']['final_accuracy']-after['full_V_clean_accuracy'])>1e-12 or abs(sp['L_actual_training_As']['final_accuracy']-after['L_actual_training_As_accuracy'])>1e-12:raise ValueError('Source default-head accuracy recount differs')
 source[arm]=dict(initial=before,final=after,actual_training_As_by_scene=byscene);source_pairs[arm]=sp
controls={a:read(control/(a+'_JOINT-s2026092701')/'source_qualification/complete.json') for a in ('FULL_NoAs','FULL_As','TEMP_NoAs','TEMP_As')};all_source={a:source[a]['final'] for a in arms};all_source.update(controls)
if decision['new_readouts']!={a:source[a]['final'] for a in arms} or decision['control_readouts']!=controls:raise ValueError('Fixed source decision inputs differ')
urban=lambda a:float(np.asarray(all_source[a]['source_accuracies'])[:,1:].mean());clean=lambda a:all_source[a]['full_V_clean_accuracy'];q=decision['criteria'];du=100*(urban('RAW_As')-urban('PAIR_As'));df=100*(urban('RAW_As')-urban('FULL_As'));kp=100*(clean('RAW_As')-clean('PAIR_As'));kf=100*(clean('RAW_As')-clean('FULL_As'));net=100*(clean('RAW_As')-clean('RAW_NoAs'));no=100*(urban('RAW_NoAs')-urban('PAIR_NoAs'))
for key,v in dict(RAW_minus_PAIR_As_urban_pp=du,RAW_minus_FULL_As_urban_pp=df,RAW_minus_PAIR_As_fullVclean_pp=kp,RAW_minus_FULL_As_fullVclean_pp=kf,RAW_As_minus_RAW_NoAs_fullVclean_pp=net,RAW_minus_PAIR_NoAs_urban_pp=no,RAW_PAIR_As_specific_urban_interaction_pp=du-no).items():
 if abs(q[key]-v)>1e-10:raise ValueError('Source effect independent recount differs')
if bool(du>=.5 and df>=.5 and kp>=-.2 and kf>=-.2 and net>=-.2)!=q['qualified']:raise ValueError('Fixed source qualification differs')
result=dict(status='SOURCE_REPLAY_VERIFIED',run_id=c['run_id'],decision=decision,source=source,source_parent_error_redistribution=source_pairs,logs=logs,states=states,initial_logits_max_abs_difference=initial_diff,source_criterion_independently_recomputed=True,all_new_updates_replayed=8880,new_physical_simulations=0,model_seed_count=1,formal_scores_read=False,goal_complete=False)
if c['source_only']:
 print(json.dumps(result));raise SystemExit(0)
summary=read(run/'analysis.json');paths={a:run/(a+'-s2026092701') for a in arms};paths.update({a:control/(a+'_JOINT-s2026092701') for a in controls});pred={};scores={};ids=None
for arm,out in paths.items():
 done=read(out/'scoring/complete.json')
 if done['status']!='SCORED_COMPLETE' or not done['truth_last'] or (arm in arms and decision['timestamp']>=(out/'scoring/complete.json').stat().st_mtime):raise ValueError('All fixed predictions/independent truth-last/source order required')
 with np.load(out/'prediction/predictions.npz',allow_pickle=False) as f:
  if ids is None:ids=f['ids'].copy()
  if not np.array_equal(ids,f['ids']):raise ValueError('New/reused formal physical IDs differ')
  pred[arm]={v:f[v].copy() for v in c['views']}
 scores[arm]=read(out/'scoring/scores.json')['results']
from experiments.cvs_early_geometry.evaluate import truth_arrays
y,rx,day=truth_arrays(read(Path(c['truth'])),ids)
def mask(dim,v):return np.ones(len(y),bool) if dim=='overall' else rx==v if dim=='receiver' else day==v if dim=='day' else y==int(v) if dim=='transmitter' else None
confusion={}
for arm,records in scores.items():
 values=[]
 for r in records:
  take=mask(r['dimension'],r['stratum'])
  if take is None or not take.any():raise ValueError('Unknown scoring stratum')
  p=pred[arm][r['view']][take];t=y[take];cm=np.bincount(t*6+p,minlength=36).reshape(6,6);acc=float(np.trace(cm)/cm.sum());den=cm.sum(0)+cm.sum(1);f1=float(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean())
  if abs(acc-r['accuracy'])>1e-12 or abs(f1-r['macro_f1'])>1e-12:raise ValueError('Independent confusion recount differs')
  values.append(dict(view=r['view'],dimension=r['dimension'],stratum=r['stratum'],count=int(cm.sum()),accuracy=acc,macro_f1=f1,confusion=cm.tolist()))
 confusion[arm]=values
for key,records in summary['paired_comparisons'].items():
 a,b=key.split('_minus_')
 for r in records:
  take=mask(r['dimension'],r['stratum']);pa,pb,t=pred[a][r['view']][take],pred[b][r['view']][take],y[take]
  if int(((pa==t)&(pb!=t)).sum())!=r['repair'] or int(((pa!=t)&(pb==t)).sum())!=r['damage']:raise ValueError('Independent paired repair/damage differs')
for v in c['views']:
 interaction=100*((pred['RAW_As'][v]==y).mean()-(pred['PAIR_As'][v]==y).mean()-(pred['RAW_NoAs'][v]==y).mean()+(pred['PAIR_NoAs'][v]==y).mean())
 if abs(interaction-summary['RAW_PAIR_As_interaction_pp'][v])>1e-10:raise ValueError('RAW/PAIR As interaction differs')
parent=project/'runs'/c['reference_run']/'B0-s2026092701';parent_pairs={}
with np.load(parent/'prediction/predictions.npz',allow_pickle=False) as f:
 if not np.array_equal(ids,f['ids']):raise ValueError('Reused original F parent prediction IDs differ')
 for arm in arms:
  parent_pairs[arm]={v:dict(count=len(y),repair=int(((pred[arm][v]==y)&(f[v]!=y)).sum()),damage=int(((pred[arm][v]!=y)&(f[v]==y)).sum()),net_pp=float(100*((pred[arm][v]==y).mean()-(f[v]==y).mean()))) for v in c['views']}
result.update(status='ANALYZED',formal_scores_read=True,summary=summary,confusion_recount=confusion,all_formal_confusions_recounted=sum(len(v) for v in confusion.values()),new_formal_confusions_recounted=sum(len(confusion[a]) for a in arms),all_paired_strata_recounted=sum(len(v) for v in summary['paired_comparisons'].values()),parent_error_redistribution=parent_pairs,parent_frozen_reference=dict(run_id=c['reference_run'],row_id='B0-s2026092701',new_predictions=0))
print(json.dumps(result))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--output',required=True);p.add_argument('--source-only',action='store_true');a=p.parse_args();rt=d.read(a.runtime)
 if rt['run_id']!=d.RUN:raise ValueError('Wrong raw temporal runtime')
 compile(SCRIPT.replace('CONFIG','{}'),'<remote-raw-analysis>','exec');c=dict(project=rt['project'],run_id=d.RUN,bank_run=d.BANK_RUN,control_run=d.CONTROL_RUN,reference_run=d.REFERENCE_RUN,release=rt['project']+'/releases/'+d.RELEASE,truth=str(d.TRUTH).replace('PROJECT',rt['project']).replace('\\','/'),views=list(d.VIEWS),source_only=a.source_only)
 r=remote(SCRIPT.replace('CONFIG',repr(c)),rt);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);d.write(out/('source_analysis.json' if a.source_only else 'complete_analysis.json'),r);print(json.dumps(dict(status=r['status'],source=r['decision']['criteria'],preferred=r['decision']['preferred'],updates=r['all_new_updates_replayed'],formal_scores_read=r['formal_scores_read'],confusions=r.get('all_formal_confusions_recounted'),new_confusions=r.get('new_formal_confusions_recounted'),pairs=r.get('all_paired_strata_recounted'),goal_complete=False)))
if __name__=='__main__':main()
