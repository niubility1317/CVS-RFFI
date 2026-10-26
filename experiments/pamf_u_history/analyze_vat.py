"""Full VAT source logs and the preregistered source-only continuation rule."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from .analyze import records,dump_csv,paired_rows
from .source import SEED
from .vat_source import RUN
from .vat_objective import RECIPE


def scientific_rule(state, overall):
    by={r['view']:r for r in overall};scenes=('practical_low_suburban','practical_mid_urban','practical_low_urban')
    complete=set(by)=={'raw','G_raw',*scenes,*('G_'+v for v in scenes)}
    if not complete:return dict(passed=False,incomplete=True)
    conditions=dict(exact_calls=state['calls']==6660,exact_U_exposures=state['seen']==1701000,
        all_U=state['unique_seen']==56700 and state['min_visits']==30 and state['max_visits']==30,
        all_U_VAT=state['VAT_seen']==state['seen'],nonzero_direction_fraction=state['nonzero_direction_fraction']>=.99,
        radius_bounded=state['max_relative_radius']<=RECIPE['relative_radius']*(1+1e-4),
        input_probe_state_protected=state['probe_parameter_grad_and_buffer_checks']>=1,
        clean_floor=by['raw']['delta_pp']>=-.5,
        raw_three_nondecreasing=sum(by[v]['net_fix'] for v in scenes)>=0,
        G_three_nondecreasing=sum(by['G_'+v]['net_fix'] for v in scenes)>=0,
        no_unauthorized_updates=not state['U_truth_read'] and not state['target_reads'] and not state['model_EMA_updates'])
    return dict(passed=all(conditions.values()),incomplete=False,conditions=conditions,
        raw_clean_delta_pp=by['raw']['delta_pp'],raw_three_delta_pp=sum(by[v]['delta_pp'] for v in scenes)/3,
        G_three_delta_pp=sum(by['G_'+v]['delta_pp'] for v in scenes)/3,
        scope='prespecified source necessary conditions only;not target success')


def analyze(evidence,output):
    root=Path(evidence);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    folder=root/'pilot/F/F0-PL-VAT-s2026092701/source';segments=[];curves=[];steps=[];last_step=None
    for name,epochs,offset,path in [('prefix20',20,0,root/'prefix20'),('segment30',10,20,folder/'segment30'),('segment50',20,30,folder/'segment50')]:
        if not (path/'step_metrics.jsonl.gz').exists():continue
        done=(path/'complete.json').exists();values,closed=records(path/'step_metrics.jsonl.gz',complete=done)
        if [v['step'] for v in values]!=list(range(1,len(values)+1)) or (done and len(values)!=epochs*222):
            raise ValueError('Complete sequential registered VAT update budget required')
        epoch_rows,_=records(path/'metrics_epoch.jsonl',complete=done)
        if done and len(epoch_rows)!=epochs:raise ValueError('Closed VAT segment epoch coverage differs')
        for row in epoch_rows:
            curves.append(dict(segment=name,cumulative_epoch=offset+row['epoch'],**{k:row.get(k) for k in ('lr','epoch_time_s','train_loss','val_tx_acc')}))
        if name!='prefix20':
            for row in values:
                m=row['metrics'];last_step=dict(cumulative_step=offset*222+row['step'],segment=name,**{k:v for k,v in m.items() if k.startswith('VAT_')})
                steps.append(last_step)
                for key in ('VAT_positive_loss','VAT_smooth_loss','VAT_max_relative_radius'):
                    if key in m and not np.isfinite(m[key]):raise ValueError('Nonfinite actual VAT metric')
        segments.append(dict(segment=name,step_records=len(values),epoch_records=len(epoch_rows),complete=done,gzip_closed=closed))
    errors=[]
    pattern=re.compile(r'Traceback \(most recent call last\)|CUDA out of memory|FloatingPointError:|RuntimeError:|^Killed\s*$',re.I|re.M)
    log=root/'pilot/logs/PL-VAT.log'
    if log.exists():errors=[dict(character_offset=m.start(),marker=m.group()) for m in pattern.finditer(log.read_text(encoding='utf-8',errors='replace'))]
    paired=[];state=None;rule=dict(passed=False,incomplete=True)
    state_path=folder/'vat_state.json';validation=folder.parent/'source_validation'
    if state_path.exists():state=json.loads(state_path.read_text(encoding='utf-8'))
    if state is not None and (validation/'complete.json').exists():
        header=json.loads((validation/'complete.json').read_text(encoding='utf-8'))
        if (header['status']!='VAT_SOURCE_EVALUATED' or header['source_V_seen']!=27000 or header['target_reads'] or header['U_truth_read']
                or not header['parameters_and_registered_buffers_unchanged']):raise ValueError('Complete frozen source V only required')
        with np.load(validation/'source_V_predictions.npz',allow_pickle=False) as a,np.load(root/'existing_PL_source/source_V_predictions.npz',allow_pickle=False) as b:
            ids=np.flatnonzero((b['methods']=='PLControl')&(b['seeds']==SEED))
            if len(ids)!=1 or not np.array_equal(a['ids'],b['ids']) or not np.array_equal(a['y'],b['y']) or not np.array_equal(a['views'],b['views']):
                raise ValueError('Exact physical source/readout pairing required')
            paired=paired_rows(a['logits'],b['logits'][int(ids[0])],a['y'],list(a['views']),'PL-VAT',strata=dict(TX=a['y'],RX=b['rx'],day=b['day']))
        rule=scientific_rule(state,[r for r in paired if r['dimension']=='overall'])
    result=dict(status='ANALYZED_SOURCE_ONLY' if not rule['incomplete'] else 'PARTIAL_LIVE_SOURCE_ANALYSIS',run_id=RUN,
        segments=segments,latest_step=last_step,state=state,source_rule=rule,full_stdout_error_markers=errors,
        target_performance='UNTESTED',comparator='completed X3 ordinary PL;new SIG1 architecture differs',
        scope='all complete structured records parsed;live segments explicitly partial;no target/model updates')
    if errors:result['source_rule']['passed']=False
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    dump_csv(out/'training_epoch_curves.csv',curves);dump_csv(out/'VAT_step_metrics.csv',steps);dump_csv(out/'source_paired_vs_existing_PL.csv',paired)
    print(json.dumps(dict(status=result['status'],segments=segments,latest_step=last_step,source_rule=rule,error_markers=len(errors))))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.evidence,a.output)
