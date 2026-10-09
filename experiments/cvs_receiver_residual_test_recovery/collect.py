"""Read final artifacts only; independently recount all exported confusion scores."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import tarfile
from experiments.cvs_receiver_residual_test_recovery import design as d
from experiments.cvs_receiver_residual_v2.publish import ssh

def verify(folder):
    done=json.loads((folder/'completion.json').read_text(encoding='utf-8'))
    scoring=json.loads((folder/'scoring_complete.json').read_text(encoding='utf-8'))
    if done['status']!='ANALYZED' or scoring['result_rows']!=1568 or scoring['truth_last'] is not True or scoring['independent_recount']!='VERIFIED':raise ValueError('Terminal coverage differs')
    rows=json.loads((folder/'scores.json').read_text(encoding='utf-8'))['results']
    keys=set();overall={};sub={};accuracies={};score_lookup={}
    for r in rows:
        key=(r['row_id'],r['view'],r['dimension'],r['stratum'])
        if key in keys:raise ValueError('Duplicate score row')
        keys.add(key);cm=r['confusion'];n=sum(map(sum,cm));correct=sum(cm[i][i] for i in range(6))
        score_lookup[key]=r
        f1=sum(2*cm[i][i]/(sum(cm[i])+sum(a[i] for a in cm)) if sum(cm[i])+sum(a[i] for a in cm)>0 else 0. for i in range(6))/6
        if n!=r['query_count'] or abs(correct/n-r['accuracy'])>1e-12 or abs(f1-r['macro_f1'])>1e-12:raise ValueError('Local independent metric recount differs')
        group=(r['row_id'],r['view'])
        if r['dimension']=='overall':
            if n!=168000:raise ValueError('Overall query count differs')
            overall[group]=cm;accuracies[r['arm'],r['model_seed'],r['view']]=r['accuracy']
        else:sub.setdefault((group,r['dimension']),[]).append(cm)
    if len(overall)!=112 or len(rows)!=1568:raise ValueError('Fixed row/view coverage differs')
    for group,cm in overall.items():
        for dim,count in [('receiver',7),('transmitter',6)]:
            cms=sub[group,dim]
            if len(cms)!=count or any(sum(m[i][j] for m in cms)!=cm[i][j] for i in range(6) for j in range(6)):
                raise ValueError('RX/TX partitions do not sum to overall confusion')
    day_rows=json.loads((folder/'day_scores.json').read_text(encoding='utf-8'))['results']
    if len(day_rows)!=448:raise ValueError('Day coverage differs')
    day_groups={}
    for r in day_rows:
        cm=r['confusion'];n=sum(map(sum,cm))
        f1=sum(2*cm[i][i]/(sum(cm[i])+sum(a[i] for a in cm)) if sum(cm[i])+sum(a[i] for a in cm)>0 else 0. for i in range(6))/6
        if n!=42000 or n!=r['query_count'] or abs(sum(cm[i][i] for i in range(6))/n-r['accuracy'])>1e-12 or abs(f1-r['macro_f1'])>1e-12:
            raise ValueError('Day independent recount differs')
        day_groups.setdefault((r['row_id'],r['view']),[]).append(cm)
    for group,cm in overall.items():
        cms=day_groups[group]
        if len(cms)!=4 or any(sum(m[i][j] for m in cms)!=cm[i][j] for i in range(6) for j in range(6)):
            raise ValueError('Day partitions differ')
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))['results']
    for r in summary:
        for metric in ('accuracy','macro_f1','worst_rx'):
            values=[]
            for seed in (2026092701,2026092702,2026092703,2026092704):
                rid=r['arm']+'-s'+str(seed)
                value=min(row['accuracy'] for key,row in score_lookup.items() if key[0]==rid and key[1]==r['view'] and key[2]=='receiver') if metric=='worst_rx' else score_lookup[rid,r['view'],'overall','ALL'][metric]
                values.append(value)
            mean=sum(values)/4;sd=math.sqrt(sum((x-mean)**2 for x in values)/3)
            if abs(mean-r[metric+'_mean'])>1e-12 or abs(sd-r[metric+'_sd'])>1e-12:raise ValueError('Full seed summary differs')
    paired=json.loads((folder/'paired_results.json').read_text(encoding='utf-8'))['comparisons']
    weights=dict(displacement_vs_baseline={'displacement':1,'baseline':-1},contribution_vs_baseline={'contribution':1,'baseline':-1},
        combined_vs_baseline={'combined':1,'baseline':-1},interaction={'combined':1,'displacement':-1,'contribution':-1,'baseline':1})
    if len(paired)!=28:raise ValueError('Paired contrast coverage differs')
    for r in paired:
        values=[100*sum(w*accuracies[arm,s,r['view']] for arm,w in weights[r['contrast']].items()) for s in (2026092701,2026092702,2026092703,2026092704)]
        mean=sum(values)/4;sd=math.sqrt(sum((x-mean)**2 for x in values)/3)
        if any(abs(a-b)>1e-12 for a,b in zip(values,r['seed_differences_pp'])) or abs(mean-r['mean_pp'])>1e-12 or abs(sd-r['sd_pp'])>1e-12:
            raise ValueError('Paired mean/SD recount differs')
    return dict(status='VERIFIED',score_rows=len(rows),overall_rows=len(overall),fixed_model_seeds=4,views=7,
        all_confusion_accuracy_f1_recount=True,all_RX_TX_day_partition_recount=True,day_rows=448,
        accuracy_F1_worstRX_seed_mean_sampleSD_recount=True,all_paired_contrasts_recount=True)


def main():
    script=r'''
import json,sys,tarfile
from pathlib import Path
project=Path(PROJECT);release=project/'releases'/RELEASE
if not (release/'completion.json').is_file() or json.loads((release/'completion.json').read_text())['status']!='ANALYZED':raise RuntimeError('Pipeline is not terminal ANALYZED')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
    for name in ('completion.json','owner.json','all_predictions_verified.json','dispatcher.stdout.log'):
        tar.add(release/name,arcname='control/'+name,recursive=False)
    marker=project/'releases'/PRED_RELEASE/'all_predictions_complete.json'
    tar.add(marker,arcname='control/all_predictions_complete.json',recursive=False)
    for run in RUNS:
        base=project/'runs'/run['run_id']
        pred=project/'runs'/run['prediction_run_id']
        names=['lineage.json','completion.json','scoring_complete.json','scores.json','scores.csv','summary.json','summary.csv',
            'paired_results.json','resources.json','analysis.md','day_scores.json','day_scores.csv','receiver_map.json','scorer.stdout.log']
        for name in names:tar.add(base/name,arcname=run['run_id']+'/'+name,recursive=False)
        names=['source_matrix_frozen.json','dispatcher.json','launch_predict.json','queue_state.json']
        for seed in (2026092701,2026092702,2026092703,2026092704):
            for arm in ('baseline','displacement','contribution','combined'):
                rid=arm+'-s'+str(seed)
                names+=[rid+'/prediction/'+n for n in ('provenance.json','resolved_config.json','complete.json')]
        for name in names:tar.add(pred/name,arcname=run['run_id']+'/'+name,recursive=False)
        state=dict(run_id=run['run_id'],prediction_run_id=run['prediction_run_id'],source_frozen_mtime=(pred/'source_matrix_frozen.json').stat().st_mtime,
            latest_prediction_mtime=max((pred/(a+'-s'+str(s))/'prediction/complete.json').stat().st_mtime for s in (2026092701,2026092702,2026092703,2026092704) for a in ('baseline','displacement','contribution','combined')),
            all32_predictions_marker_mtime=marker.stat().st_mtime,
            scoring_complete_mtime=(base/'scoring_complete.json').stat().st_mtime,completion_mtime=(base/'completion.json').stat().st_mtime)
        blob=(json.dumps(state)+'\n').encode();item=tarfile.TarInfo(run['run_id']+'/poststate.json');item.size=len(blob)
        import io;tar.addfile(item,io.BytesIO(blob))
'''
    code=script.replace('PROJECT',repr(d.PROJECT)).replace('PRED_RELEASE',repr(d.RELEASE)).replace('RELEASE',repr(d.SCORE_RELEASE)).replace('RUNS',repr(d.SCORE_RUNS))
    blob=ssh(code);output=d.ROOT/'local_artifacts'/d.SCORE_RELEASE;output.mkdir(exist_ok=True)
    (output/'final_results.tar.gz').write_bytes(blob);folder=output/'final_results';folder.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        for m in tar.getmembers():
            if not m.isfile() or not (folder/m.name).resolve().is_relative_to(folder.resolve()):raise ValueError('Unsafe result member')
        tar.extractall(folder)
    checks=[]
    for run in d.SCORE_RUNS:
        base=folder/run['run_id'];check=verify(base);post=json.loads((base/'poststate.json').read_text(encoding='utf-8'))
        if not post['source_frozen_mtime']<post['latest_prediction_mtime']<=post['all32_predictions_marker_mtime']<post['scoring_complete_mtime']<=post['completion_mtime']:
            raise ValueError('Artifact source-freeze/prediction/truth-last order differs')
        check.update(post);(base/'local_independent_recount.json').write_text(json.dumps(check,indent=2)+'\n',encoding='utf-8');checks.append(check)
    (output/'final_verified.json').write_text(json.dumps(dict(status='VERIFIED',runs=checks),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED',all_models=32,all_views=224,total_score_rows=3136,runs=checks)))


if __name__=='__main__':main()
