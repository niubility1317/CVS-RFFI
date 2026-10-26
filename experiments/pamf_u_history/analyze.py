"""Complete structured-log parsing and physically paired source readouts."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import zlib
import numpy as np
from .history import METHODS
from .source import SEED


def records(path, *, complete):
    data=Path(path).read_bytes();closed=True
    if str(path).endswith('.gz'):
        stream=zlib.decompressobj(16+zlib.MAX_WBITS);data=stream.decompress(data);closed=stream.eof
    if complete and not closed:raise ValueError('Closed segment has incomplete gzip: '+str(path))
    if data and not data.endswith(b'\n'):
        if complete:raise ValueError('Closed segment has partial structured record')
        data=data.rsplit(b'\n',1)[0]+b'\n' if b'\n' in data else b''
    rows=[json.loads(line) for line in data.decode('utf-8-sig').splitlines() if line.strip()]
    return rows,closed


def dump_csv(path, rows):
    rows=list(rows)
    if not rows:return
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def step_summary(rows, offset):
    if [r['step'] for r in rows]!=list(range(1,len(rows)+1)):
        raise ValueError('Full sequential native step records required')
    result=[]
    for start in range(0,len(rows),222):
        block=rows[start:start+222];keys=('train/loss','UH_positive_loss','UH_negative_loss','UH_temporal_loss','UH_ramp','UH_negative_batch','UH_temporal_batch')
        row=dict(cumulative_epoch=offset+start//222+1,steps=len(block),complete_epoch=len(block)==222)
        for key in keys:
            values=[float(r['metrics'][key]) for r in block if r['metrics'].get(key) is not None]
            if values and not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite training metric: '+key)
            row[key.replace('/','_')+'_mean']=statistics.mean(values) if values else None
        for key in ('UH_seen','UH_accepted','UH_negative_seen','UH_temporal_seen'):
            row[key+'_last']=block[-1]['metrics'].get(key)
        result.append(row)
    return result


def paired_rows(candidate, control, truth, views, method, *, strata=None):
    if candidate.shape!=control.shape or candidate.shape!=(len(views),len(truth),6):
        raise ValueError('Same source samples, views and class space required')
    masks=[('overall','ALL',np.ones(len(truth),dtype=bool))]
    for name,values in (strata or {}).items():
        if len(values)!=len(truth):raise ValueError('Physical source stratum length mismatch')
        masks.extend((name,str(v),values==v) for v in np.unique(values))
    rows=[];a=candidate.argmax(-1);b=control.argmax(-1)
    for j,view in enumerate(views):
        for dimension,stratum,mask in masks:
            n=int(mask.sum());good=a[j]==truth;old=b[j]==truth
            fix=int((mask&good&~old).sum());harm=int((mask&~good&old).sum())
            rows.append(dict(method=method,view=view,dimension=dimension,stratum=stratum,n=n,
                candidate_correct=int((mask&good).sum()),control_correct=int((mask&old).sum()),
                candidate_accuracy=100*float(good[mask].mean()),control_accuracy=100*float(old[mask].mean()),
                delta_pp=100*(fix-harm)/n,wrong_to_right=fix,right_to_wrong=harm,net_fix=fix-harm,
                comparison='existing PLControl SIG_TIME_X3;new SIG_TIME single-view;architecture differs'))
    return rows


def analyze(evidence, output):
    root=Path(evidence);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    curves=[];steps=[];segments=[];states=[];error_rows=[]
    paths=[('shared_PL_prefix','prefix20',20,0)]
    for method in METHODS:
        base='pilot/F/F0-'+method+'-s'+str(SEED)+'/source/'
        paths.extend((method,base+'segment'+str(end),epochs,offset) for end,epochs,offset in ((30,10,20),(50,20,30)))
    for method,folder,epochs,offset in paths:
        p=root/folder
        if not (p/'step_metrics.jsonl.gz').exists():continue
        complete=(p/'complete.json').exists();logs,closed=records(p/'step_metrics.jsonl.gz',complete=complete)
        if complete and len(logs)!=epochs*222:raise ValueError('Closed segment does not cover complete registered budget')
        rows=step_summary(logs,offset)
        for row in rows:row.update(method=method,segment=offset+epochs)
        steps.extend(rows)
        epoch_logs,_=records(p/'metrics_epoch.jsonl',complete=complete)
        if complete and len(epoch_logs)!=epochs:raise ValueError('Closed segment epoch coverage differs')
        for row in epoch_logs:
            selected={k:row.get(k) for k in ('lr','epoch_time_s','train_loss','val_tx_acc')}
            curves.append(dict(method=method,cumulative_epoch=offset+row['epoch'],**selected))
        segments.append(dict(method=method,segment_end=offset+epochs,step_records=len(logs),epoch_records=len(epoch_logs),
                             segment_complete=complete,gzip_closed=closed,source=str(p.relative_to(root))))
        if (p/'u_history_state.json').exists():
            state=json.loads((p/'u_history_state.json').read_text(encoding='utf-8'));states.append(dict(method=method,segment_end=offset+epochs,**state))
    fatal=re.compile(r'Traceback \(most recent call last\)|CUDA out of memory|FloatingPointError:|RuntimeError:|^Killed\s*$',re.I|re.M)
    stdout=[]
    for path in sorted((root/'pilot/logs').glob('*.log')):
        text=path.read_text(encoding='utf-8',errors='replace');matches=list(fatal.finditer(text))
        stdout.append(dict(file=path.name,bytes=path.stat().st_size,lines=len(text.splitlines()),fatal_markers=len(matches)))
        error_rows.extend(dict(file=path.name,character_offset=m.start(),marker=m.group()) for m in matches)
    comparisons=[];baseline=root/'existing_PL_source/source_V_predictions.npz'
    if baseline.exists():
        with np.load(baseline,allow_pickle=False) as b:
            index=np.flatnonzero((b['methods']=='PLControl')&(b['seeds']==SEED))
            if len(index)!=1:raise ValueError('Exactly one existing completed ordinary-PL source comparator required')
            j=int(index[0]);truth=b['y'];views=list(b['views']);control=b['logits'][j]
            for method in METHODS:
                folder=root/'pilot/F'/('F0-'+method+'-s'+str(SEED))/'source_validation'
                pred=folder/'source_V_predictions.npz'
                if not pred.exists() or not (folder/'complete.json').exists():continue
                with np.load(pred,allow_pickle=False) as a:
                    if list(a['views'])!=views or not np.array_equal(a['ids'],b['ids']) or not np.array_equal(a['y'],truth):
                        raise ValueError('Source physical IDs/labels/view order differs from reused comparator')
                    comparisons.extend(paired_rows(a['logits'],control,truth,views,method,strata=dict(TX=truth,RX=b['rx'],day=b['day'])))
            baseline_info=dict(method='PLControl',model_seed=SEED,signature_views=3,source_V=len(truth),
                source='20261025-diagnostic-PAMF-RestoredPL-sourceShift-m9-r01',
                results=[dict(view=view,correct=int((control[k].argmax(-1)==truth).sum()),accuracy=100*float((control[k].argmax(-1)==truth).mean())) for k,view in enumerate(views)])
    else:baseline_info=None
    result=dict(status='PARTIAL_LIVE_SOURCE_ANALYSIS',
                segments=segments,history_states=states,stdout_full_scan=stdout,fatal_markers=error_rows,
                existing_control=baseline_info,new_final_source_rows=len({r['method'] for r in comparisons}),
                target_performance='UNTESTED',U_label_correctness='UNKNOWN',
                scope='All complete structured records parsed;live segments explicitly partial;no target or model updates')
    # Completion derives from final artifacts, not the expected number of strata.
    result['status']='SOURCE_ANALYSIS_COMPLETE' if result['new_final_source_rows']==3 else 'PARTIAL_LIVE_SOURCE_ANALYSIS'
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    dump_csv(out/'training_epoch_curves.csv',curves);dump_csv(out/'training_step_epoch_summary.csv',steps)
    dump_csv(out/'source_paired_vs_existing_PL.csv',comparisons);dump_csv(out/'stdout_fatal_markers.csv',error_rows)
    print(json.dumps(dict(status=result['status'],parsed_steps=sum(s['step_records'] for s in segments),
                          parsed_epochs=sum(s['epoch_records'] for s in segments),final_rows=result['new_final_source_rows'],fatal_markers=len(error_rows))))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.evidence,a.output)
