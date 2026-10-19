"""Complete source replay, then optional independent formal confusion recount. No model fit."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.cvs_feature_adapters30.publish import remote
from experiments.cvs_feature_adapters30 import design as d

SCRIPT=r'''
import gzip,json,math,sys
from pathlib import Path
import numpy as np
import torch
c=CONFIG;run=Path(c['project'])/'runs'/c['run_id'];release=Path(c['release']);sys.path.insert(0,str(release))
from experiments.cvs_feature_adapters30.runner import bootstrap
bootstrap()
def read(p):return json.loads(p.read_text())
decision=read(run/'source_matrix_decision.json');bank=read(run/'observation_bank/complete.json');indices=np.load(run/'observation_bank/indices.npy',allow_pickle=False)
if decision['formal_scores_read'] or decision['current_cumulative_epochs']!=30 or bank['updates']!=2220 or not bank['all_six_share_actual_IQ'] or bank['physical_training_observations']!=284160:raise ValueError('Fixed source-only plan/stream differs')
arms=[s+'_'+a for s in ('FULL','LORA','TEMP') for a in ('NoAs','As')];source={};logs={};states={};first_logits=None;initial_maxdiff={}
for arm in arms:
 row=arm+'-s2026092701';out=run/row;cfg=read(out/'source/resolved_config.json');done=read(out/'source/completion.json');final=read(out/'source_qualification/complete.json');initial=read(out/'source/initial_source_readout/complete.json')
 if done['optimizer_steps']!=2220 or done['epoch']!=30 or done['U_used'] or done['EMA_used'] or not done['default_head_frozen'] or done['student_training_NN']!=568320 or done['parent_reference_NN']!=284160 or initial['NN']!=50400 or final['NN']!=50400:raise ValueError('Actual budget/fixed-W/L-only metadata differs')
 values=[];maxerror=0.;names=set(cfg['trainable_names'])
 with gzip.open(out/'source/step_metrics.jsonl.gz','rt') as f:
  for i,line in enumerate(f):
   v=json.loads(line);expected=v['clean_CE']+v['second_CE']+v['keep_KL'];maxerror=max(maxerror,abs(expected-v['total_loss']))
   if v['step']!=i+1 or v['epoch']!=i//222+1 or v['cumulative_epoch']!=21+i//222 or v['L_indices']!=indices[i].tolist() or v['bank_step']!=i+1 or v['As_consumed']!=(arm.endswith('_As')) or v['U_used'] or not v['head_frozen'] or v['student_NN']!=256 or v['parent_NN']!=128 or set(v['gradient_squared_by_parameter'])-names or any('.cls_head.' in n for n in v['gradient_squared_by_parameter']) or not all(math.isfinite(g) and g>=0 for g in v['gradient_squared_by_parameter'].values()):raise ValueError('Full actual training stream/gradient replay differs')
   values.append(v)
 if len(values)!=2220 or maxerror>2e-5:raise ValueError('Incomplete updates or three-term objective differs')
 initial_state=torch.load(out/'source/initialization.pt',map_location='cpu',weights_only=False)['model'];final_state=torch.load(out/'source/final_ssdg.pth',map_location='cpu',weights_only=False)['model'];changed=[k for k,v in initial_state.items() if not torch.equal(v,final_state[k])];head='id_backbone.encoder.core.id_backbone.cls_head.weight'
 if not changed or set(changed)-names or head in changed:raise ValueError('Adapted features missing or frozen state changed')
 states[arm]=dict(changed_parameters=changed,fixed_default_W_bit_equal=True,all_frozen_native_state_bit_equal=True,trainable_parameters=cfg['trainable_parameters'],total_parameters=cfg['total_parameters'])
 with np.load(out/'source/initial_source_readout/predictions.npz',allow_pickle=False) as f:
  current={k:f[k].copy() for k in ('L_ids','L_clean','L_actual_training_As','L_new','V_clean','V_satellite','V_full_clean')}
  if first_logits is None:first_logits=current
  if not np.array_equal(current['L_ids'],first_logits['L_ids']):raise ValueError('Initial source diagnostic IDs differ')
  initial_maxdiff[arm]=max(float(np.abs(current[k]-first_logits[k]).max()) for k in current if k!='L_ids')
 curve=[dict(epoch=e+1,**{k:float(np.mean([v[k] for v in values[e*222:(e+1)*222]])) for k in ('clean_CE','second_CE','keep_KL','total_loss','gradient_norm')}) for e in range(10)]
 logs[arm]=dict(actual_steps=2220,all_L_indices_equal_shared_stream=True,max_loss_sum_abs_error=maxerror,curves=curve,wall_seconds=done['wall_seconds'])
 source[arm]=dict(initial=initial,final=final)
urban=lambda a:np.asarray(source[a]['final']['source_accuracies'])[:,1:].mean();clean=lambda a:source[a]['final']['full_V_clean_accuracy'];recount={}
for style in ('FULL','LORA','TEMP'):
 a,b=style+'_As',style+'_NoAs';du=100*(urban(a)-urban(b));dc=100*(clean(a)-clean(b));df=100*(urban(a)-urban('FULL_As'));q=decision['criteria'][style]
 if abs(du-q['urban_As_minus_NoAs_pp'])>1e-10 or abs(dc-q['fullVclean_As_minus_NoAs_pp'])>1e-10 or abs(df-q['urban_minus_FULL_As_pp'])>1e-10 or bool(du>=.5 and dc>=-.2 and (style=='FULL' or df>=0))!=q['qualified']:raise ValueError('Predeclared source criterion recount differs')
 recount[style]=q
result=dict(status='SOURCE_REPLAY_VERIFIED',run_id=c['run_id'],decision=decision,source=source,logs=logs,states=states,initial_logits_max_abs_difference=initial_maxdiff,source_criterion_independently_recomputed=True,all_new_updates_replayed=13320,model_seed_count=1,formal_scores_read=False,goal_complete=False)
if c['source_only']:
 print(json.dumps(result));raise SystemExit(0)
summary=read(run/'analysis.json');ids=None;predictions={};score_rows={}
for arm in arms:
 out=run/(arm+'-s2026092701');sd=read(out/'scoring/complete.json')
 if sd['status']!='SCORED_COMPLETE' or not sd['truth_last'] or decision['timestamp']>=(out/'scoring/complete.json').stat().st_mtime:raise ValueError('Source decision/independent truth-last order differs')
 with np.load(out/'prediction/predictions.npz',allow_pickle=False) as f:
  if ids is None:ids=f['ids'].copy()
  if not np.array_equal(ids,f['ids']):raise ValueError('Formal paired IDs differ')
  predictions[arm]={v:f[v].copy() for v in c['views']}
 score_rows[arm]=read(out/'scoring/scores.json')['results']
from experiments.cvs_early_geometry.evaluate import truth_arrays
y,rx,day=truth_arrays(read(Path(c['truth'])),ids);confusion={}
for arm in arms:
 audit=[]
 for record in score_rows[arm]:
  dim,v=record['dimension'],record['stratum'];mask=np.ones(len(y),bool) if dim=='overall' else rx==v if dim=='receiver' else day==v if dim=='day' else y==int(v) if dim=='transmitter' else None
  if mask is None or not mask.any():raise ValueError('Unknown scoring stratum')
  p=predictions[arm][record['view']][mask];cm=np.bincount(y[mask]*6+p,minlength=36).reshape(6,6);acc=float(np.trace(cm)/cm.sum());den=cm.sum(0)+cm.sum(1);f1=float(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean())
  if abs(acc-record['accuracy'])>1e-12 or abs(f1-record['macro_f1'])>1e-12:raise ValueError('Independent full confusion recount differs')
  audit.append(dict(view=record['view'],dimension=dim,stratum=v,count=int(cm.sum()),accuracy=acc,macro_f1=f1,confusion=cm.tolist()))
 confusion[arm]=audit
for key,records in summary['paired_comparisons'].items():
 a,b=key.split('_minus_')
 for r in records:
  dim,v=r['dimension'],r['stratum'];mask=np.ones(len(y),bool) if dim=='overall' else rx==v if dim=='receiver' else day==v if dim=='day' else y==int(v);pa=predictions[a][r['view']][mask];pb=predictions[b][r['view']][mask];t=y[mask]
  if int(((pa==t)&(pb!=t)).sum())!=r['repair'] or int(((pa!=t)&(pb==t)).sum())!=r['damage']:raise ValueError('Independent paired repair/damage differs')
result.update(status='ANALYZED',formal_scores_read=True,summary=summary,confusion_recount=confusion,all_formal_confusions_recounted=sum(len(v) for v in confusion.values()),all_paired_strata_recounted=sum(len(v) for v in summary['paired_comparisons'].values()))
print(json.dumps(result))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--output',required=True);p.add_argument('--source-only',action='store_true');a=p.parse_args();rt=d.read(a.runtime);out=Path(a.output)
 if rt['run_id']!=d.RUN:raise ValueError('Wrong run runtime')
 c=dict(project=rt['project'],run_id=d.RUN,release=rt['project']+'/releases/'+d.RELEASE,truth=str(d.TRUTH).replace('PROJECT',rt['project']).replace('\\','/'),views=list(d.VIEWS),source_only=a.source_only)
 result=remote(SCRIPT.replace('CONFIG',repr(c)),rt);out.mkdir(parents=True,exist_ok=False);d.write(out/('source_analysis.json' if a.source_only else 'complete_analysis.json'),result)
 print(json.dumps(dict(status=result['status'],source=result['decision']['criteria'],preferred=result['decision']['preferred'],all_new_updates_replayed=result['all_new_updates_replayed'],formal_scores_read=result['formal_scores_read'],confusion_records=result.get('all_formal_confusions_recounted'),model_seed_count=1,goal_complete=False)))

if __name__=='__main__':main()
