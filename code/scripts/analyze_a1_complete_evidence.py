"""Parse every captured text record and export complete comparable score tables."""
import argparse
import collections
import csv
import io
import json
import math
from pathlib import Path
import re
import statistics
import tarfile

SCENES=['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']

def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,required=True);a=ap.parse_args()
    docs={}; curves={}; parsed=[]; errors=[]; stdout={}; score_rows=[]; class_rows=[]
    for archive_name in ['evidence.tar.gz','extra_json.tar.gz']:
        with tarfile.open(a.folder/archive_name) as archive:
            for member in archive:
                if not member.isfile():continue
                name=member.name;raw=archive.extractfile(member).read(); text=raw.decode('utf-8-sig',errors='strict')
                if name=='inventory.json':
                    docs[archive_name+'/inventory.json']=json.loads(text);continue
                info={'path':name,'bytes':len(raw),'records':0,'parse_errors':[]}
                if name.endswith('.jsonl'):
                    records=[]
                    for i,line in enumerate(text.splitlines(),1):
                        if not line.strip():continue
                        try:records.append(json.loads(line))
                        except Exception as exc:info['parse_errors'].append({'line':i,'error':str(exc)})
                    info['records']=len(records)
                    if name.endswith('/metrics_epoch.jsonl'):
                        curves['/'.join(name.split('/')[1:3])]=records
                elif name.endswith('.csv'):
                    records=list(csv.DictReader(io.StringIO(text)));info['records']=len(records)
                    # Fully parse every CSV field; expose malformed line widths.
                    info['malformed_width_rows']=sum(None in row for row in records)
                elif name.endswith('.json'):
                    try:docs[name]=json.loads(text);info['records']=1
                    except Exception as exc:info['parse_errors'].append({'error':str(exc)})
                else:
                    lines=text.splitlines();info['records']=len(lines)
                    markers=collections.Counter()
                    findings=[]
                    for i,line in enumerate(lines,1):
                        for key,pattern in {'traceback':r'Traceback \(most recent call last\)',
                            'systemic_nonfinite':r'RC4_SYSTEMIC_NONFINITE_BATCH_GUARD',
                            'oom':r'CUDA out of memory|OutOfMemoryError',
                            'nonfinite_guard':r'NONFINITE|non-finite|nonfinite.*skip|skip.*nonfinite',
                            'warnings':r'Warning:|WARNING',
                            'killed':r'^Killed\b',
                            'runtime_error':r'^RuntimeError:'}.items():
                            if re.search(pattern,line):
                                markers[key]+=1
                                if key!='warnings':findings.append({'line':i,'kind':key,'text':line})
                    stdout[name]={'lines':len(lines),'markers':dict(markers),'findings':findings,
                        'epoch_end':[int(x) for x in re.findall(r'\[EPOCH-END\] E(\d+)',text)],
                        'initialization':[x for x in lines if 'init=scratch' in x or '[CONFIG-RUN]' in x],
                        'tail':lines[-12:]}
                parsed.append(info)
                if info['parse_errors']:errors.append(info)
    states={name.split('/')[1]:value for name,value in docs.items() if name.endswith('/pipeline_state.json')}
    summaries={}
    for key,records in curves.items():
        if not records:continue
        run,row=key.split('/',1);records.sort(key=lambda x:x.get('epoch',0));last=records[-1]
        epochs=[int(x['epoch']) for x in records]
        state=states.get(run,{}).get('rows',{}).get(row,{})
        source=lambda field:[x[field] for x in records if finite(x.get(field))]
        times=source('epoch_time_s')
        base_recent=[float(x['epoch_time_s'])-float(x.get('train_time_periodic_target_seconds',0) or 0)
                     for x in records[-10:] if finite(x.get('epoch_time_s'))]
        eval_times=[x['train_time_periodic_target_seconds'] for x in records
                    if finite(x.get('train_time_periodic_target_seconds')) and x['train_time_periodic_target_seconds']>0]
        skips=[{'epoch':x['epoch'],'grad':x.get('train_skipped_nonfinite_grad',0),
                'loss':x.get('train_skipped_nonfinite_loss',0)} for x in records
                if float(x.get('train_skipped_nonfinite_grad',0) or 0)>0 or float(x.get('train_skipped_nonfinite_loss',0) or 0)>0]
        telemetry={}
        for field in sorted(set().union(*(x.keys() for x in records))):
            if not any(tag in field for tag in ('r3_','response_','fisher_','loss_daot_','daot_objective','ecrs_cross')):continue
            values=[(x['epoch'],x[field]) for x in records if finite(x.get(field))]
            if not values:continue
            telemetry[field]={'latest':values[-1][1],'maximum':max(v for _,v in values),
                'first_positive_epoch':next((e for e,v in values if v>0),None),
                'positive_epochs':sum(v>0 for _,v in values)}
        summary={'run':run,'row':row,'epochs':epochs,'parsed_records':len(records),
            'duplicate_epochs':len(epochs)-len(set(epochs)),
            'missing_epochs':sorted(set(range(min(epochs),max(epochs)+1))-set(epochs)),
            'latest_epoch':max(epochs),'total_epochs':last.get('epochs'),
            'state':state,'total_epoch_seconds':sum(times),'recent_base_seconds':base_recent,
            'recent_base_median':statistics.median(base_recent) if base_recent else None,
            'recent_source_eval_median':statistics.median([
                float(x.get('train_muse_time_base_validation_s',0) or 0)+float(x.get('train_muse_time_heavy_source_validation_s',0) or 0)
                for x in records[-10:]]),
            'periodic_seconds':eval_times,'periodic_median':statistics.median(eval_times) if eval_times else None,
            'peak_mb':max(source('train_muse_peak_cuda_memory_mb'),default=None),
            'latest_loss':last.get('train_loss'),'latest_source_clean':last.get('val_tx_acc'),
            'latest_source_leo':last.get('stage_source_val_sat_mean_tx'),
            'best_source_clean':max(source('val_tx_acc'),default=None),
            'skipped_epochs':skips,'mechanism_telemetry':telemetry,
            'from_scratch':last.get('from_scratch'),'baseline_ckpt':last.get('baseline_ckpt')}
        summaries[key]=summary
    for name,value in docs.items():
        if not name.endswith('/score.json'):continue
        parts=name.split('/');run,row=parts[1:3]
        match=re.search(r'/E(\d+)/',name)
        epoch=int(match.group(1)) if match else summaries.get(run+'/'+row,{}).get('latest_epoch')
        metrics=value.get('metrics',{})
        if set(metrics)!=set(SCENES):raise ValueError(f'Scenario coverage mismatch: {name}')
        if value.get('record_count')!=672000:raise ValueError(f'Record count mismatch: {name}')
        result={'run':run,'row':row,'epoch':epoch,'kind':'periodic' if match else 'final',
            'records':value['record_count'],'score_path':name}
        for scene in SCENES:
            metric=metrics[scene]
            assert metric['total']==168000
            assert abs(metric['accuracy']-metric['correct']/metric['total'])<1e-12
            result[scene]=metric['accuracy']*100
            for cls,accuracy in metric['per_class_accuracy'].items():
                class_rows.append({'run':run,'row':row,'epoch':epoch,'kind':result['kind'],
                    'scenario':scene,'class':cls,'accuracy_pct':accuracy*100})
        result['leo_mean']=sum(result[s] for s in SCENES[1:])/3
        result['worst_scene']=min(result[s] for s in SCENES[1:])
        score_rows.append(result)
    score_rows.sort(key=lambda x:(x['run'],x['row'],x['epoch'] or 0))
    for filename,rows in [('all_target_scores.csv',score_rows),('all_target_per_class.csv',class_rows)]:
        with (a.folder/filename).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    curve_csv=[]
    for key,records in curves.items():
        run,row=key.split('/',1)
        curve_csv.extend({'run':run,'row':row,**{k:r.get(k) for k in ['epoch','epoch_time_s',
            'train_time_periodic_target_seconds','train_loss','val_tx_acc','stage_source_val_sat_mean_tx',
            'train_optimizer_step_applied','train_skipped_nonfinite_grad','train_skipped_nonfinite_loss','lr']}}
            for r in records)
    with (a.folder/'all_training_curves.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(curve_csv[0]));writer.writeheader();writer.writerows(curve_csv)
    selected_keys=['epoch','epoch_time_s','train_time_periodic_target_seconds','train_loss','train_loss_tx_labeled',
        'val_tx_acc','stage_source_val_sat_mean_tx','train_skipped_nonfinite_grad','train_skipped_nonfinite_loss',
        'train_optimizer_step_applied','train_muse_peak_cuda_memory_mb','lr']
    compact_curves={key:[{k:r.get(k) for k in selected_keys} for r in records] for key,records in curves.items()}
    result={'inventory':[docs[x+'/inventory.json'] for x in ['evidence.tar.gz','extra_json.tar.gz']],
        'parsed_files':parsed,'parse_errors':errors,'summaries':summaries,'states':states,
        'stdout':stdout,'scores':score_rows,'curves':compact_curves,
        'extra_json_paths':[name for name in docs if 'inventory.json' not in name]}
    (a.folder/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'files':len(parsed),'records':sum(x['records'] for x in parsed),'parse_errors':len(errors),
        'training_rows':len(summaries),'scores':len(score_rows),'per_class_rows':len(class_rows)}))
    for key,s in summaries.items():
        if 'a1_mechanism_periodic' in key or 'a1_r3_budget' in key or 'a1_r3_clean' in key:
            print(key,s['state'].get('status'),s['latest_epoch'],'base_s',s['recent_base_median'],
                'eval_s',s['periodic_median'],'skips',len(s['skipped_epochs']))
    print('RUN_STATES',json.dumps({k:v.get('status') for k,v in states.items()}))

if __name__=='__main__':main()
