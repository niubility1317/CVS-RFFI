"""Read-only full-log and artifact audit after training/prediction/scoring closure."""
import argparse
import json
from pathlib import Path
from experiments.cvs_selected_concat.prepare import PROJECT,RUN,RELEASE
from experiments.cvs_selected_concat.publish import ssh

REMOTE=r'''
import csv,json,math,subprocess,time
from collections import defaultdict
from pathlib import Path
project=Path(PROJECT);root=project/'runs'/RUN;release=project/'releases'/RELEASE
def read(p):return json.loads(p.read_text())
completion=read(root/'completion.json');marker=read(root/'scoring_p1_complete.json')
if completion.get('status')!='ANALYZED' or marker.get('status')!='SCORED_COMPLETE':raise ValueError('Experiment not closed')
launch=read(root/'launch.json');rows=[]
for row in launch['rows']:
 source=root/row['row_id']/'source';pred=root/row['row_id']/'prediction'
 epochs=[json.loads(x) for x in (source/'epoch_metrics.jsonl').read_text().splitlines()]
 csvrows=list(csv.DictReader((source/'epoch_metrics.csv').open()))
 logpath=Path(row['log']);stdout=[json.loads(x[6:]) for x in logpath.read_text().splitlines() if x.startswith('EPOCH ')]
 if len(epochs)!=200 or len(csvrows)!=200 or stdout!=epochs:raise ValueError('Structured/text epoch mismatch')
 agg=defaultdict(lambda:dict(steps=0,samples=0,satellite=0,clean=0.,sat_ce=0.,loss=0.,norm=0.))
 steps=0
 for line in (source/'step_metrics.jsonl').read_text().splitlines():
  v=json.loads(line);steps+=1;e=v['epoch'];g=agg[e];active=e>=80
  if v['step']!=steps or v['satellite_weight']!=(.68 if active else 0.) or v['clean_weight']!=1. or v['augmentation_active']!=active:raise ValueError('Step/augmentation schedule mismatch')
  if v['domain_backbone_active'] or v['extra_losses_active'] or v['pseudo_labels_active']:raise ValueError('Unexpected method state')
  if (v['satellite_ce'] is None)!= (not active) or v['satellite_samples']!=(v['source_samples'] if active else 0):raise ValueError('Satellite exposure mismatch')
  expected=v['clean_ce']+(.68*v['satellite_ce'] if active else 0.)
  if not math.isclose(v['total_loss'],expected,abs_tol=3e-6,rel_tol=3e-6):raise ValueError('CE weighting mismatch')
  if any(not math.isfinite(v[k]) for k in ['total_loss','clean_ce','gradient_norm','learning_rate']) or v['gradient_used_parameters']!=164225:raise ValueError('Gradient/finite mismatch')
  g['steps']+=1;g['samples']+=v['source_samples'];g['satellite']+=v['satellite_samples'];g['clean']+=v['clean_ce'];g['sat_ce']+=v['satellite_ce'] or 0.;g['loss']+=v['total_loss'];g['norm']+=v['gradient_norm']
 if steps!=10000 or sorted(agg)!=list(range(1,201)):raise ValueError('Full step budget missing')
 for i,v in enumerate(epochs,1):
  g=agg[i];active=i>=80
  if v['epoch']!=i or g['steps']!=50 or g['samples']!=6300 or g['satellite']!=(6300 if active else 0):raise ValueError('Epoch exposure mismatch')
  for key,total in [('clean_ce','clean'),('total_loss','loss'),('gradient_norm','norm')]+([('satellite_ce','sat_ce')] if active else []):
   if not math.isclose(v[key],g[total]/50,rel_tol=1e-8,abs_tol=1e-8) or not math.isclose(float(csvrows[i-1][key]),v[key],rel_tol=1e-8,abs_tol=1e-8):raise ValueError('Step/epoch/CSV aggregate mismatch')
  if v['source_val_count']!=27000 or not math.isclose(v['source_val_worst_rx'],min(v['source_val_rx_accuracy'].values())):raise ValueError('Source validation mismatch')
 initial=read(source/'initialization.json');resolved=read(source/'resolved_config.json');done=read(source/'completion.json');freeze=read(source/'source_selection.json');provenance=read(pred/'provenance.json');prediction=read(pred/'phase1_complete.json')
 if initial['checkpoint'] is not None or initial['ancestors'] or initial['checkpoint_sources'] or initial['target_access'] or initial['target_contact']:raise ValueError('Non scratch')
 if resolved['variant']!='residual_fusion' or resolved['augmentation_recipe']['scenes']!=['practical_mid','practical_low_urban'] or resolved['total_parameters']!=164225 or resolved['target_access'] or done['steps']!=10000:raise ValueError('Actual method mismatch')
 if done['target_access'] or done['target_evaluated'] or freeze['target_access'] or freeze['target_score_used'] or prediction['truth_read'] or prediction['count']!=168000 or provenance['status']!='VERIFIED':raise ValueError('Freeze/query provenance mismatch')
 rows.append(dict(row_id=row['row_id'],gpu=row['gpu'],epochs=epochs,resolved=resolved,initialization=initial,completion=done,freeze=freeze,profile=read(source/'resource_profile.json'),prediction_complete=prediction,provenance=provenance,
  audit=dict(status='PASS',step_records=steps,epoch_records=200,csv_records=200,stdout_epoch_records=200,clean_exposure=sum(v['samples'] for v in agg.values()),satellite_exposure=sum(v['satellite'] for v in agg.values()),high_training_augmentation=False),
  paths=dict(checkpoint=str(source/'last.pt'),steps=str(source/'step_metrics.jsonl'),stdout=str(logpath),prediction=str(pred/'phase1_predictions.npz')),
  worker_alive=Path('/proc/'+str(row['pid'])).exists(),source_alive=Path('/proc/'+str(resolved['pid'])).exists()))
print(json.dumps(dict(status='VERIFIED',read_at=time.time(),identity=dict(user=subprocess.check_output(['whoami'],text=True).strip(),host=subprocess.check_output(['hostname'],text=True).strip()),
 release_commit=(release/'release_commit.txt').read_text().strip(),completion=completion,scoring=marker,rows=rows,
 scored=read(root/'phase1_scored_results.json'),summary=read(root/'phase1_summary.json'),
 dispatcher_alive=Path('/proc/'+str(read(release/'submit.json')['pid'])).exists(),failure=(root/'failure.json').exists())))
'''

def collect(output):
    script=REMOTE
    for key,value in [('PROJECT',PROJECT),('RUN',RUN),('RELEASE',RELEASE)]:script=script.replace(key,repr(value))
    result=json.loads(ssh(script))
    if result['failure'] or len(result['rows'])!=4:raise ValueError('Failed/incomplete matrix')
    output.mkdir(parents=True,exist_ok=True)
    for name,value in [('completion_audit',result),('test_scores',result['scored']),('test_summary',result['summary'])]:
        p=output/(name+'.json')
        if p.exists():raise FileExistsError('Preserve collected evidence')
        p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED',rows=4,full_step_records=40000,epoch_records=800,score_records=result['scoring']['records'],source_pids_alive=[r['source_alive'] for r in result['rows']])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.output)
