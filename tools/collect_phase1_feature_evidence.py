"""Collect immutable Phase1 source/score aggregates; no training or target IQ.

Complete source stdout and every structured optimizer step are parsed once per
immutable row; cached audits are reused, never partial logs relabeled complete.
"""
import argparse,json
from pathlib import Path
from experiments.cvs_phase1_stack.publish import ssh
REMOTE=r'''
import sys,json,gzip,math,time,re
from pathlib import Path
import torch
torch.set_num_threads(1)
c=CONFIG;root=Path(c['project'])/'releases'/c['release'];sys.path[:0]=[str(root),str(root/'code')]
from importlib import import_module
d=import_module(c['package']+'.design');proof=import_module(c['package']+'.provenance').source_proof
cache=CACHE
result=dict(run_id=d.RUN,read_at=time.time(),release_commit=(root/'release_commit.txt').read_text().strip(),rows=[],history=[],source_audit_scope='Full stdout, all50 epoch records and all11100 optimizer-step records for completed sources; immutable cached row audits reused',truth_read=False,target_IQ_read=False,model_updates=0,head_fit_updates=0)
for row in d.rows():
 cfg=d.config(row);sid=row['row_id'];source=Path(cfg['baseline_reference']['output_root'] if cfg['baseline_reuse'] else cfg['output_root']);entry=dict(**row,source=str(source),config=cfg,source_complete=False,row_complete=False)
 if (source/'completion.json').is_file():
  proof(cfg);done=d.read(source/'completion.json');entry['source_complete']=True
  cached=next((x for x in cache.get('rows',[]) if x['row_id']==sid and x.get('source_complete') and x.get('config')==cfg and x.get('source_audit',{}).get('status')=='PASS'),None)
  if cached:entry['source_audit']=cached['source_audit'];entry['source_audit_reused']=True
  else:
   epochs=[json.loads(s) for s in (source/'epoch_metrics.jsonl').read_text().splitlines()]
   if len(epochs)!=50 or [int(x['epoch']) for x in epochs]!=list(range(1,51)):raise ValueError('Full50 epoch logs missing '+sid)
   steps=updates=utotal=uselected=hidden=nonfinite=0;byepoch={}
   with gzip.open(source/'step_metrics.jsonl.gz','rt',encoding='utf-8') as f:
    for line in f:
     v=json.loads(line);steps+=1;m=v['metrics'];ep=int(v['epoch'])
     if int(v['step'])!=steps or ep!=(steps-1)//222+1:raise ValueError('Optimizer step identity differs '+sid)
     loss=float(m['train/loss']);ce=float(m['train/loss_tx_labeled'])
     if not math.isfinite(loss) or not math.isfinite(ce):nonfinite+=1
     nonfinite+=int(m.get('train/skipped_nonfinite_loss',0) or 0)+int(m.get('train/skipped_nonfinite_grad',0) or 0)
     t=int(m.get('train/pseudo_total',0) or 0);a=int(m.get('train/pseudo_selected',0) or 0)
     if not 0<=a<=t:raise ValueError('Pseudo count invalid '+sid)
     utotal+=t;uselected+=a;hidden+=int(m.get('train/pseudo_truth_available',0) or 0)
     g=byepoch.setdefault(ep,dict(count=0,total_loss=0.,L_CE=0.,U_total=0,U_accepted=0));g['count']+=1;g['total_loss']+=loss;g['L_CE']+=ce;g['U_total']+=t;g['U_accepted']+=a
   if steps!=11100 or done['optimizer_steps']!=11100 or nonfinite or hidden:raise ValueError('Actual update/hidden truth/finite budget invalid '+sid)
   for ep,g in byepoch.items():
    if g['count']!=222 or g['U_total']!=(56700 if ep>=34 else 0):raise ValueError('Actual epoch L/U budget invalid '+sid)
   log=source.parent.parent/'logs'/f'{sid}.log'
   if cfg['baseline_reuse']:log=d.old.BASE/'logs'/f"{cfg['baseline_reference']['row_id']}.log"
   stdout=log.read_text(errors='replace');fatal=[s for s in stdout.splitlines() if re.search(r'Traceback \(most recent call last\)|CUDA out of memory|^RuntimeError:|^ValueError:|^FloatingPointError:',s)]
   if fatal:raise ValueError('Source full stdout fatal '+sid+repr(fatal))
   compact=[]
   for e in epochs:
    g=byepoch[int(e['epoch'])];compact.append(dict(epoch=int(e['epoch']),phase=e['phase'],lr=e['lr'],mean_L_CE=g['L_CE']/222,mean_total_loss=g['total_loss']/222,source_V_percent=e.get('val_tx_acc'),U_total=g['U_total'],U_accepted=g['U_accepted'],U_accept_rate=g['U_accepted']/g['U_total'] if g['U_total'] else None,epoch_seconds=e.get('epoch_time_s')))
   resolved=d.read(source/'resolved_config.json');entry['source_audit']=dict(status='PASS',read_at=time.time(),full_stdout_scanned=True,stdout_bytes=len(stdout.encode()),epochs=50,steps=11100,optimizer_EMA_updates=done['optimizer_steps'],nonfinite_or_skipped=nonfinite,U_truth_available=hidden,U_forward_samples=utotal,U_accepted=uselected,U_accept_rate=uselected/utotal if utotal else None,final_source=done['final_source_metrics'],final_L_CE=done['final_training_L_CE'],training_seconds=done['elapsed_seconds'],total_parameters=resolved.get('total_parameters'),feature_extractor=done['feature_extractor'],identity_input_policy=done['identity_input_policy'],curve=compact,baseline_reused=cfg['baseline_reuse'])
 if entry['source_complete']:
  epochs=[json.loads(s) for s in (source/'epoch_metrics.jsonl').read_text().splitlines()]
  for curve,e in zip(entry['source_audit']['curve'],epochs):
   curve['native_unlabeled_loss']=e.get('train_loss_unlabeled');curve['weighted_total_minus_L']=curve['mean_total_loss']-curve['mean_L_CE']
  entry['source_audit']['U_observation_draws']=entry['source_audit'].pop('U_forward_samples',entry['source_audit'].get('U_observation_draws'))
  entry['source_audit']['U_cost_definition']='One hidden U observation drawn per weak-teacher/strong-student pair; does not count the two network forwards as two physical draws'
 if (d.BASE/sid/'completion.json').is_file():
  from_module=import_module(c['package']+'.predict');from_module.preflight(sid)
  v=d.read(d.BASE/sid/'completion.json');scores=d.read(d.BASE/sid/'scores/metrics.json')
  if v['status']!='TRAINED_DEFAULT_PREDICTED_SCORED' or not v['truth_last'] or v['metric_records']!=18 or not scores['truth_last'] or len(scores['records'])!=18:raise ValueError('Fixed default predictions/score incomplete '+sid)
  for metric in scores['records']:
   cm=metric['confusion'];n=sum(map(sum,cm));acc=sum(cm[k][k] for k in range(6))/n;f1=sum(2*cm[k][k]/(sum(cm[k])+sum(z[k] for z in cm)) if sum(cm[k])+sum(z[k] for z in cm) else 0. for k in range(6))/6
   if n!=metric['query_count'] or abs(acc-metric['accuracy'])>1e-12 or abs(f1-metric['macro_f1'])>1e-12:raise ValueError('Raw confusion recount invalid '+sid)
  entry.update(row_complete=True,metric_records=scores['records'],default_prediction_count=168000,truth_last=True,raw_confusions_recounted=True,source_diagnostics=d.read(d.BASE/sid/'diagnostics/summary.json'))
 result['rows'].append(entry)
for path in sorted((d.BASE/'historical_time_diagnostics').glob('*/summary.json')):
 v=d.read(path)
 if v['target_access'] or v['head_fit_updates']!=0 or not v['model_params_buffers_unchanged']:raise ValueError('Historical source-only boundary invalid')
 result['history'].append(v)
if (d.BASE/'analysis.json').is_file():result['analysis']=d.read(d.BASE/'analysis.json')
if (d.BASE/'completion.json').is_file():result['completion']=d.read(d.BASE/'completion.json')
result.update(complete_source_rows=sum(x['source_complete'] for x in result['rows']),complete_scored_rows=sum(x['row_complete'] for x in result['rows']),all_rows_complete=all(x['row_complete'] for x in result['rows']))
print(json.dumps(result,allow_nan=False))
'''
def collect(package,output):
 from importlib import import_module
 d=import_module(package+'.design');output=Path(output);output.mkdir(parents=True,exist_ok=True);path=output/'full_evidence.json'
 prior=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
 # Cache only source-log audits; current provenance and prediction closure rerun.
 cache=dict(rows=[dict(row_id=r['row_id'],config=r['config'],source_complete=r['source_complete'],source_audit=r.get('source_audit')) for r in prior.get('rows',[]) if r.get('source_audit')])
 config=dict(project=d.PROJECT,release=d.RELEASE,package=package)
 payload=REMOTE.replace('CONFIG',repr(config)).replace('CACHE',repr(cache))
 wrapper="import subprocess\nr=subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input="+repr(payload)+",text=True,capture_output=True,env=dict(__import__('os').environ,CUDA_VISIBLE_DEVICES=''))\nprint(r.stdout)\nif r.returncode:raise RuntimeError(r.stderr)\n"
 result=json.loads(ssh(wrapper));path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:result[k] for k in ('run_id','complete_source_rows','complete_scored_rows','all_rows_complete')}));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--package',required=True,choices=['experiments.cvs_feature_sequence50','experiments.cvs_time_factorial50']);p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.package,a.output)
