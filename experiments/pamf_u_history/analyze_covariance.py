"""Complete covariance training records and frozen, physically paired source results."""
import argparse
import json
from pathlib import Path
import re
import statistics
import numpy as np
from .analyze import records,dump_csv,paired_rows
from .covariance_source import RUN,METHODS,SEED

SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
VIEWS=('raw','G_raw',*(v for s in SCENES for v in (s,'G_'+s)))
COMPARATOR='completed matched SIG1 ordinaryPL seed2026092701;same original fullPAMF,source roles,defaulthead and E50 budget'


def source_rule(state,overall):
    by={r['view']:r for r in overall}
    if set(by)!=set(VIEWS) or len(overall)!=8:return dict(passed=False,incomplete=True)
    if any(r['n']!=27000 or r['net_fix']!=r['candidate_correct']-r['control_correct'] for r in overall):
        raise ValueError('Complete physically paired V27000 integer counts required')
    conditions=dict(exact_calls=state['calls']==6660,exact_U_exposures=state['seen']==1701000,
        all_U=state['unique_seen']==56700 and state['min_visits']==30 and state['max_visits']==30,
        covariance_all_U=state['covariance_U_seen']==state['seen'],
        clean_floor=by['raw']['net_fix']>=-27,
        raw_three_nondecreasing=sum(by[v]['net_fix'] for v in SCENES)>=0,
        G_three_nondecreasing=sum(by['G_'+v]['net_fix'] for v in SCENES)>=0,
        no_unauthorized_updates=not state['U_truth_read'] and not state['target_reads'] and not state['model_EMA_updates'])
    return dict(passed=all(conditions.values()),incomplete=False,conditions=conditions,
        raw_clean_delta_pp=100*by['raw']['net_fix']/27000,
        raw_three_delta_pp=100*sum(by[v]['net_fix'] for v in SCENES)/81000,
        G_three_delta_pp=100*sum(by['G_'+v]['net_fix'] for v in SCENES)/81000,
        scope='source-only preregistered necessary conditions;not formal target or active-goal success')


def paired(candidate,control,method):
    for k in ('ids','y','views','rx','day'):
        if not np.array_equal(candidate[k],control[k]):raise ValueError('Exact source physical/label/view/RX/day pairing required: '+k)
    a=candidate['logits'];b=control['logits'];y=candidate['y'];views=list(candidate['views'])
    if (a.shape!=b.shape or a.shape!=(len(views),len(y),6) or not np.isfinite(a).all() or not np.isfinite(b).all()
        or y.dtype.kind not in 'iu' or np.any((y<0)|(y>=6))):raise ValueError('Finite complete six-class defaulthead logits required')
    rows=paired_rows(a,b,y,views,method,strata=dict(TX=y,RX=candidate['rx'],day=candidate['day']))
    masks={('overall','ALL'):np.ones(len(y),bool)}
    for key,value in [('TX',y),('RX',candidate['rx']),('day',candidate['day'])]:
        masks.update({(key,str(s)):value==s for s in np.unique(value)})
    for row in rows:
        j=views.index(row['view']);mask=masks[(row['dimension'],row['stratum'])]
        margin=[]
        for z in (a[j],b[j]):
            wrong=z.copy();wrong[np.arange(len(y)),y]=-np.inf
            margin.append(z[np.arange(len(y)),y]-wrong.max(1))
        row.update(comparison=COMPARATOR,candidate_defaulthead_logit_margin=float(margin[0][mask].mean()),
            control_defaulthead_logit_margin=float(margin[1][mask].mean()),
            margin_interpretation='unscaled defaulthead logit margin;not feature geometry or scale-invariant evidence')
    agreement=[]
    for route,prefix in [('raw',''),('DURA-LU','G_')]:
        reference=views.index(prefix+'raw')
        for scene in SCENES:
            if prefix+scene not in views:continue
            j=views.index(prefix+scene)
            for label,z in [('candidate',a),('completed_PL',b)]:
                first=z[reference].argmax(1);second=z[j].argmax(1)
                for (dimension,stratum),mask in masks.items():
                    agreement.append(dict(method=method,model=label,readout=route,scene=scene,dimension=dimension,stratum=stratum,
                        n=int(mask.sum()),same_packet_decision_agreement=float((first[mask]==second[mask]).mean()),
                        both_correct=int((mask&(first==y)&(second==y)).sum()),
                        raw_correct_perturbed_wrong=int((mask&(first==y)&(second!=y)).sum()),
                        predicted_classes=int(len(np.unique(second[mask]))),
                        interpretation='higher agreement alone can reflect collapsed or jointly wrong predictions'))
    return rows,agreement


def analyze(evidence,output):
    root=Path(evidence);pilot=root/'pilot';out=Path(output);out.mkdir(parents=True,exist_ok=True)
    segments=[];curves=[];epoch_steps=[];errors=[];stdout=[];last_steps={};paired_all=[];agreements=[];decisions=[];prefix_info=None
    prefix=root/'prefix20'
    if (prefix/'step_metrics.jsonl.gz').exists():
        complete=(prefix/'complete.json').exists();logs,closed=records(prefix/'step_metrics.jsonl.gz',complete=complete)
        epochs,_=records(prefix/'metrics_epoch.jsonl',complete=complete)
        if [v['step'] for v in logs]!=list(range(1,len(logs)+1)) or complete and (len(logs)!=4440 or len(epochs)!=20):
            raise ValueError('Complete shared ordinaryPL E20 log budget required')
        for row in logs:
            call=max(0,row['step']-13*222);m=row['metrics']
            if (m['T1_L_packets']!=768 or m['T1_As_KL_weight']!=0 or m.get('T1_PL_seen',0)!=call*256-132*(call//222)):
                raise ValueError('Shared E20 must retain full L and original13L7LU plainPL exposure')
        for row in epochs:curves.append(dict(method='shared_PL_E20',cumulative_epoch=row['epoch'],**{k:row.get(k) for k in ('lr','epoch_time_s','train_loss','val_tx_acc')}))
        segments.append(dict(method='shared_PL_E20',segment_end=20,steps=len(logs),epochs=len(epochs),complete=complete,gzip_closed=closed))
        prefix_info=dict(cumulative_epochs=20,steps=len(logs),U_seen=logs[-1]['metrics'].get('T1_PL_seen',0),U_accepted=logs[-1]['metrics'].get('T1_PL_accepted',0),
            scope='shared inherited prefix counted once;not newly trained for this covariance run') if logs else None
    for method in METHODS:
        folder=pilot/'F'/('F0-'+method+'-s'+str(SEED))/'source'
        for end,length,offset in ((30,10,20),(50,20,30)):
            p=folder/('segment'+str(end));path=p/'step_metrics.jsonl.gz'
            if not path.exists():continue
            complete=(p/'complete.json').exists();steps,closed=records(path,complete=complete)
            if [v['step'] for v in steps]!=list(range(1,len(steps)+1)) or complete and len(steps)!=length*222:
                raise ValueError('Full sequential covariance optimizer budget required')
            epochs,_=records(p/'metrics_epoch.jsonl',complete=complete)
            if complete and len(epochs)!=length:raise ValueError('Complete native epoch records required')
            for row in epochs:curves.append(dict(method=method,cumulative_epoch=offset+row['epoch'],**{k:row.get(k) for k in ('lr','epoch_time_s','train_loss','val_tx_acc')}))
            for start in range(0,len(steps),222):
                batch=steps[start:start+222];row=dict(method=method,cumulative_epoch=offset+start//222+1,steps=len(batch),complete_epoch=len(batch)==222)
                for key in ('COV_loss','COV_PL_loss','COV_weight'):
                    values=[v['metrics'].get(key) for v in batch]
                    if any(v is None or not np.isfinite(v) for v in values):raise ValueError('Actual finite covariance metrics missing: '+key)
                    row[key+'_mean']=statistics.mean(values)
                for v in batch:
                    m=v['metrics'];call=(offset-20)*222+v['step'];seen=call*256-132*(call//222)
                    if (m['COV_calls']!=call or m['COV_seen']!=seen or m['COV_L_pairs']!=64*call
                        or m['COV_L_forwards_reused']!=5 or m['COV_extra_U_forwards']!=1
                        or m['T1_L_packets']!=768 or m['T1_As_KL_weight']!=0
                        or m['T1_PL_seen']!=v['step']*256-132*(v['step']//222)):
                        raise ValueError('Actual original L/PL,reused forwards or all-U continuity differs')
                row.update({k+'_last':batch[-1]['metrics'][k] for k in ('COV_calls','COV_seen','COV_L_pairs')});epoch_steps.append(row)
            if steps:
                m=steps[-1]['metrics'];last_steps[method]=dict(cumulative_step=offset*222+steps[-1]['step'],
                    cumulative_U_seen=396900+m['COV_seen'],**{k:v for k,v in m.items() if k.startswith('COV_') or k in ('T1_PL_seen','T1_PL_accepted')})
            segments.append(dict(method=method,segment_end=end,steps=len(steps),epochs=len(epochs),complete=complete,gzip_closed=closed))
    fatal=re.compile(r'Traceback \(most recent call last\)|CUDA out of memory|FloatingPointError:|RuntimeError:|^Killed\s*$',re.I|re.M)
    for p in sorted((pilot/'logs').glob('*.log')):
        content=p.read_text(encoding='utf-8',errors='replace');matches=list(fatal.finditer(content))
        stdout.append(dict(file=p.name,bytes=p.stat().st_size,lines=len(content.splitlines()),fatal_markers=len(matches)))
        errors.extend(dict(file=p.name,offset=m.start(),marker=m.group()) for m in matches)
    control=pilot/'baseline_source/source_V_predictions.npz'
    if control.exists() and (control.parent/'complete.json').exists():
        with np.load(control,allow_pickle=False) as baseline:
            for method in METHODS:
                p=pilot/'F'/('F0-'+method+'-s'+str(SEED));prediction=p/'source_validation/source_V_predictions.npz'
                if not prediction.exists() or not (prediction.parent/'complete.json').exists():continue
                done=json.loads((prediction.parent/'complete.json').read_text());state=json.loads((p/'source/covariance_state.json').read_text())
                if (done['status']!='COVARIANCE_SOURCE_EVALUATED' or done['source_V_seen']!=27000 or done['target_reads'] or done['U_truth_read']
                    or not done['parameters_and_registered_buffers_unchanged']):raise ValueError('Complete unchanged final F/G source readout required')
                with np.load(prediction,allow_pickle=False) as candidate:rows,pairs=paired(candidate,baseline,method)
                rule=source_rule(state,[r for r in rows if r['dimension']=='overall']);decisions.append(dict(method=method,**rule))
                paired_all.extend(rows);agreements.extend(pairs)
    result=dict(status='COVARIANCE_SOURCE_ANALYSIS_COMPLETE' if len(decisions)==4 else 'PARTIAL_LIVE_COVARIANCE_SOURCE_ANALYSIS',run_id=RUN,
        segments=segments,shared_prefix=prefix_info,latest_steps=last_steps,stdout_full_scan=stdout,fatal_markers=errors,source_decisions=decisions,
        comparator=COMPARATOR,comparator_retraining=0,target_performance='UNTESTED',U_TX_correctness='UNKNOWN',
        scope='all available structured records and complete stdout parsed;live segments explicitly partial;no model/IQ/target updates')
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    dump_csv(out/'training_epoch_curves.csv',curves);dump_csv(out/'training_step_epoch_summary.csv',epoch_steps)
    dump_csv(out/'source_paired_vs_matched_PL.csv',paired_all);dump_csv(out/'same_packet_decision_consistency.csv',agreements)
    print(json.dumps(dict(status=result['status'],steps=sum(v['steps'] for v in segments),epochs=sum(v['epochs'] for v in segments),latest={k:v['cumulative_step'] for k,v in last_steps.items()},decisions=decisions,fatal_markers=len(errors))))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.evidence,a.output)
