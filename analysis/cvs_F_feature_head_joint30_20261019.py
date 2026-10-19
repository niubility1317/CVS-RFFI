"""Replay the complete joint-head adaptation and independently recount fixed predictions."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.cvs_feature_head_joint30.publish import remote
from experiments.cvs_feature_head_joint30 import design as d

SCRIPT=r'''
import gzip,json,math,sys
from pathlib import Path
import numpy as np
import torch
c=CONFIG;run=Path(c['project'])/'runs'/c['run_id'];frozen=Path(c['project'])/'runs'/c['frozen_run'];release=Path(c['release']);sys.path.insert(0,str(release))
from experiments.cvs_feature_head_joint30.runner import bootstrap
bootstrap()
def read(p):return json.loads(p.read_text())
decision=read(run/'source_matrix_decision.json');bankroot=frozen/'observation_bank';bank=read(bankroot/'complete.json');reuse=read(run/'observation_bank_reuse.json');indices=np.load(bankroot/'indices.npy',allow_pickle=False)
if decision['formal_scores_read'] or decision['current_cumulative_epochs']!=30 or bank['updates']!=2220 or not bank['all_six_share_actual_IQ'] or bank['physical_training_observations']!=284160 or reuse['new_physical_simulations']!=0 or Path(reuse['source_bank'])!=bankroot:raise ValueError('Fixed source-only plan/reused actual stream differs')
arms=[s+'_'+a for s in ('FULL','LORA','TEMP') for a in ('NoAs','As')];source={};logs={};states={};first_logits=None;initial_maxdiff={};firstmeta=read(bankroot/'actual_first_training_metadata.json');actual_scenes=np.asarray([v['scene'] for v in firstmeta]);source_pairs={};head='id_backbone.encoder.core.id_backbone.cls_head.weight'
for arm in arms:
 out=run/(arm+'_JOINT-s2026092701');cfg=read(out/'source/resolved_config.json');done=read(out/'source/completion.json');final=read(out/'source_qualification/complete.json');initial=read(out/'source/initial_source_readout/complete.json')
 if done['optimizer_steps']!=2220 or done['epoch']!=30 or done['U_used'] or done['EMA_used'] or done['default_head_frozen'] or not done['default_head_joint'] or done['head_joint_optimizer_steps']!=2220 or done['head_fit_updates']!=0 or done['student_training_NN']!=568320 or done['parent_reference_NN']!=284160 or initial['NN']!=50400 or final['NN']!=50400:raise ValueError('Actual budget/joint-W/L-only metadata differs')
 values=[];maxerror=0.;names=set(cfg['trainable_names'])
 if head not in names or len(names)<2:raise ValueError('Missing simultaneous head and feature trainables')
 with gzip.open(out/'source/step_metrics.jsonl.gz','rt') as f:
  for i,line in enumerate(f):
   v=json.loads(line);expected=v['clean_CE']+v['second_CE']+v['keep_KL'];maxerror=max(maxerror,abs(expected-v['total_loss']));grads=v['gradient_squared_by_parameter']
   if v['step']!=i+1 or v['epoch']!=i//222+1 or v['cumulative_epoch']!=21+i//222 or v['L_indices']!=indices[i].tolist() or v['bank_step']!=i+1 or v['As_consumed']!=arm.endswith('_As') or v['U_used'] or v['head_frozen'] or not v['head_joint'] or v['student_NN']!=256 or v['parent_NN']!=128 or set(grads)-names or head not in grads or not any(n!=head for n in grads) or not all(math.isfinite(g) and g>=0 for g in grads.values()):raise ValueError('Full actual stream/joint gradient replay differs')
   values.append(v)
 if len(values)!=2220 or maxerror>2e-5:raise ValueError('Incomplete updates or three-term objective differs')
 initial_state=torch.load(out/'source/initialization.pt',map_location='cpu',weights_only=False)['model'];final_state=torch.load(out/'source/final_ssdg.pth',map_location='cpu',weights_only=False)['model'];changed=[k for k,v in initial_state.items() if not torch.equal(v,final_state[k])]
 if set(changed)-names or head not in changed or not any(k!=head for k in changed):raise ValueError('Features and original W must both change; frozen native state must remain equal')
 wi=np.asarray(initial_state[head].tolist(),dtype=np.float64);wf=np.asarray(final_state[head].tolist(),dtype=np.float64);cos=np.sum(wi*wf,axis=1)/(np.linalg.norm(wi,axis=1)*np.linalg.norm(wf,axis=1))
 states[arm]=dict(changed_parameters=changed,original_default_W_jointly_changed=True,real_feature_parameters_changed=True,all_frozen_native_state_bit_equal=True,trainable_parameters=cfg['trainable_parameters'],total_parameters=cfg['total_parameters'],W_row_angles_degrees=np.rad2deg(np.arccos(np.clip(cos,-1,1))).tolist())
 with np.load(out/'source/initial_source_readout/predictions.npz',allow_pickle=False) as f:
  current={k:f[k].copy() for k in ('L_ids','L_clean','L_actual_training_As','L_new','V_clean','V_satellite','V_full_clean')}
  if first_logits is None:first_logits=current
  if not np.array_equal(current['L_ids'],first_logits['L_ids']):raise ValueError('Initial source diagnostic IDs differ')
  initial_maxdiff[arm]=max(float(np.abs(current[k]-first_logits[k]).max()) for k in current if k!='L_ids')
 curve=[dict(epoch=e+1,**{k:float(np.mean([v[k] for v in values[e*222:(e+1)*222]])) for k in ('clean_CE','second_CE','keep_KL','total_loss','gradient_norm')},head_gradient_squared_mean=float(np.mean([v['gradient_squared_by_parameter'][head] for v in values[e*222:(e+1)*222]]))) for e in range(10)]
 logs[arm]=dict(actual_steps=2220,all_L_indices_equal_reused_stream=True,max_loss_sum_abs_error=maxerror,curves=curve,wall_seconds=done['wall_seconds'])
 byscene={};sp={}
 with np.load(out/'source/initial_source_readout/predictions.npz',allow_pickle=False) as before,np.load(out/'source_qualification/predictions.npz',allow_pickle=False) as after:
  if not np.array_equal(before['L_ids'],after['L_ids']) or not np.array_equal(np.asarray([v['physical_id'] for v in firstmeta]),before['L_ids']):raise ValueError('Actual training scene/physical IDs differ')
  sy=before['L_TX'];bp=before['L_actual_training_As'].argmax(-1);ap=after['L_actual_training_As'].argmax(-1)
  for scene in sorted(set(actual_scenes)):
   mask=actual_scenes==scene;b=bp[mask];a=ap[mask];t=sy[mask];byscene[scene]=dict(count=int(mask.sum()),initial_accuracy=float((b==t).mean()),final_accuracy=float((a==t).mean()),repair=int(((a==t)&(b!=t)).sum()),damage=int(((a!=t)&(b==t)).sum()))
  for key,truthkey in [('L_clean','L_TX'),('L_actual_training_As','L_TX'),('L_new','L_TX'),('V_clean','V_TX'),('V_satellite','V_TX'),('V_full_clean','V_full_TX')]:
   b=before[key].argmax(-1);a=after[key].argmax(-1);t=before[truthkey];sp[key]=dict(count=int(a.size),repair=int(((a==t)&(b!=t)).sum()),damage=int(((a!=t)&(b==t)).sum()),initial_accuracy=float((b==t).mean()),final_accuracy=float((a==t).mean()))
  if abs(sp['V_full_clean']['final_accuracy']-final['full_V_clean_accuracy'])>1e-12 or abs(sp['L_actual_training_As']['final_accuracy']-final['L_actual_training_As_accuracy'])>1e-12:raise ValueError('Source accuracy recount differs')
 source[arm]=dict(initial=initial,final=final,actual_training_As_by_scene=byscene);source_pairs[arm]=sp
frozen_source={a:read(frozen/(a+'-s2026092701')/'source_qualification/complete.json') for a in arms}
urban=lambda group,a:float(np.asarray(group[a]['source_accuracies'])[:,1:].mean());clean=lambda group,a:group[a]['full_V_clean_accuracy'];joint={a:source[a]['final'] for a in arms}
for a in arms:
 if joint[a]!=decision['joint_readouts'][a] or frozen_source[a]!=decision['frozen_readouts'][a]:raise ValueError('Frozen source decision inputs differ')
for style in ('FULL','LORA','TEMP'):
 a,b=style+'_As',style+'_NoAs';gain=100*(urban(joint,a)-urban(frozen_source,a));keep=100*(clean(joint,a)-clean(frozen_source,a));net=100*(clean(joint,a)-clean(joint,b));dfj=100*(urban(joint,a)-urban(joint,'FULL_As'));dff=100*(urban(joint,a)-urban(frozen_source,'FULL_As'));no=100*(urban(joint,b)-urban(frozen_source,b));q=decision['criteria'][style]
 for key,val in dict(joint_minus_same_frozen_As_urban_pp=gain,joint_minus_same_frozen_As_fullVclean_pp=keep,joint_As_minus_joint_NoAs_fullVclean_pp=net,urban_minus_FULL_joint_As_pp=dfj,urban_minus_FULL_frozen_As_pp=dff,NoAs_head_effect_urban_pp=no,As_specific_head_effect_urban_pp=gain-no).items():
  if abs(q[key]-val)>1e-10:raise ValueError('Predeclared source effect recount differs')
 if bool(gain>=.5 and keep>=-.2 and net>=-.2 and (style=='FULL' or min(dfj,dff)>=.5))!=q['qualified']:raise ValueError('Predeclared source criterion differs')
result=dict(status='SOURCE_REPLAY_VERIFIED',run_id=c['run_id'],decision=decision,source=source,source_parent_error_redistribution=source_pairs,logs=logs,states=states,initial_logits_max_abs_difference=initial_maxdiff,source_criterion_independently_recomputed=True,all_new_updates_replayed=13320,new_physical_simulations=0,model_seed_count=1,formal_scores_read=False,goal_complete=False)
if c['source_only']:
 print(json.dumps(result));raise SystemExit(0)
summary=read(run/'analysis.json');ids=None;predictions={};score_rows={}
for arm in arms:
 out=run/(arm+'_JOINT-s2026092701');sd=read(out/'scoring/complete.json')
 if sd['status']!='SCORED_COMPLETE' or not sd['truth_last'] or decision['timestamp']>=(out/'scoring/complete.json').stat().st_mtime:raise ValueError('Source decision/independent truth-last order differs')
 with np.load(out/'prediction/predictions.npz',allow_pickle=False) as f:
  if ids is None:ids=f['ids'].copy()
  if not np.array_equal(ids,f['ids']):raise ValueError('Formal paired IDs differ')
  predictions[arm]={v:f[v].copy() for v in c['views']}
 score_rows[arm]=read(out/'scoring/scores.json')['results']
from experiments.cvs_early_geometry.evaluate import truth_arrays
y,rx,day=truth_arrays(read(Path(c['truth'])),ids);confusion={}
def mask_for(dim,v):
 return np.ones(len(y),bool) if dim=='overall' else rx==v if dim=='receiver' else day==v if dim=='day' else y==int(v) if dim=='transmitter' else None
parent=Path(c['project'])/'runs'/c['reference_run']/'B0-s2026092701';pd=read(parent/'prediction/complete.json')
if pd['count']!=168000 or pd['truth_read'] or not (parent/'scoring/complete.json').is_file():raise ValueError('Reused original F frozen readout incomplete')
parent_pairs={}
with np.load(parent/'prediction/predictions.npz',allow_pickle=False) as f:
 if not np.array_equal(ids,f['ids']):raise ValueError('Parent observation IDs differ')
 for arm in arms:
  parent_pairs[arm]={}
  for view in c['views']:
   b=f[view];a=predictions[arm][view];parent_pairs[arm][view]=dict(count=len(y),repair=int(((a==y)&(b!=y)).sum()),damage=int(((a!=y)&(b==y)).sum()),net_pp=float(100*((a==y).mean()-(b==y).mean())))
for arm in arms:
 audit=[]
 for record in score_rows[arm]:
  dim,v=record['dimension'],record['stratum'];mask=mask_for(dim,v)
  if mask is None or not mask.any():raise ValueError('Unknown scoring stratum')
  p=predictions[arm][record['view']][mask];cm=np.bincount(y[mask]*6+p,minlength=36).reshape(6,6);acc=float(np.trace(cm)/cm.sum());den=cm.sum(0)+cm.sum(1);f1=float(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean())
  if abs(acc-record['accuracy'])>1e-12 or abs(f1-record['macro_f1'])>1e-12:raise ValueError('Independent confusion recount differs')
  audit.append(dict(view=record['view'],dimension=dim,stratum=v,count=int(cm.sum()),accuracy=acc,macro_f1=f1,confusion=cm.tolist()))
 confusion[arm]=audit
for key,records in summary['paired_comparisons'].items():
 a,b=key.split('_minus_')
 for r in records:
  mask=mask_for(r['dimension'],r['stratum']);pa=predictions[a][r['view']][mask];pb=predictions[b][r['view']][mask];t=y[mask]
  if int(((pa==t)&(pb!=t)).sum())!=r['repair'] or int(((pa!=t)&(pb==t)).sum())!=r['damage']:raise ValueError('Independent As/style pair differs')
for arm,records in summary['head_effect_pairs'].items():
 with np.load(frozen/(arm+'-s2026092701')/'prediction/predictions.npz',allow_pickle=False) as f:
  if not np.array_equal(ids,f['ids']):raise ValueError('Reused frozen-W paired IDs differ')
  for r in records:
   mask=mask_for(r['dimension'],r['stratum']);a=predictions[arm][r['view']][mask];b=f[r['view']][mask];t=y[mask];repair=int(((a==t)&(b!=t)).sum());damage=int(((a!=t)&(b==t)).sum());net=100*(repair-damage)/int(mask.sum())
   if repair!=r['repair'] or damage!=r['damage'] or abs(net-r['net_pp'])>1e-10:raise ValueError('Independent joint-minus-frozen head effect differs')
   if r['dimension']=='overall' and abs(net-summary['joint_minus_frozen_pp'][arm][r['view']])>1e-10:raise ValueError('Head effect overall differs')
for s in ('FULL','LORA','TEMP'):
 for view in c['views']:
  interaction=summary['joint_minus_frozen_pp'][s+'_As'][view]-summary['joint_minus_frozen_pp'][s+'_NoAs'][view]
  if abs(interaction-summary['head_As_interaction_pp'][s][view])>1e-10:raise ValueError('As-specific head interaction differs')
result.update(status='ANALYZED',formal_scores_read=True,summary=summary,parent_frozen_reference=dict(run_id=c['reference_run'],row_id='B0-s2026092701',scores=read(parent/'scoring/scores.json'),new_predictions=0),parent_error_redistribution=parent_pairs,confusion_recount=confusion,all_formal_confusions_recounted=sum(len(v) for v in confusion.values()),all_paired_strata_recounted=sum(len(v) for v in summary['paired_comparisons'].values()),all_head_effect_strata_recounted=sum(len(v) for v in summary['head_effect_pairs'].values()))
print(json.dumps(result))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--output',required=True);p.add_argument('--source-only',action='store_true');a=p.parse_args();rt=d.read(a.runtime);out=Path(a.output)
 if rt['run_id']!=d.RUN:raise ValueError('Wrong run runtime')
 compile(SCRIPT.replace('CONFIG','{}'),'<remote-analysis>','exec')
 c=dict(project=rt['project'],run_id=d.RUN,reference_run=d.REFERENCE_RUN,frozen_run=d.FROZEN_RUN,release=rt['project']+'/releases/'+d.RELEASE,truth=str(d.TRUTH).replace('PROJECT',rt['project']).replace('\\','/'),views=list(d.VIEWS),source_only=a.source_only)
 result=remote(SCRIPT.replace('CONFIG',repr(c)),rt);out.mkdir(parents=True,exist_ok=False);d.write(out/('source_analysis.json' if a.source_only else 'complete_analysis.json'),result)
 print(json.dumps(dict(status=result['status'],source=result['decision']['criteria'],preferred=result['decision']['preferred'],all_new_updates_replayed=result['all_new_updates_replayed'],formal_scores_read=result['formal_scores_read'],confusion_records=result.get('all_formal_confusions_recounted'),head_effect_records=result.get('all_head_effect_strata_recounted'),model_seed_count=1,goal_complete=False)))

if __name__=='__main__':main()
