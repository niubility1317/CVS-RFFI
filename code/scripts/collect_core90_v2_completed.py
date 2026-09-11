"""Read-only complete-log audit and independent frozen source scoring."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
import json
import math
from pathlib import Path
import re
import subprocess
import sys

SCENES=('clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak')

def records(path,live=False):
    text=path.read_text(encoding='utf-8') if path.exists() else ''
    lines=text.splitlines();result=[];partial=0
    for i,line in enumerate(lines):
        try:result.append(json.loads(line))
        except json.JSONDecodeError:
            if live and i==len(lines)-1 and not text.endswith('\n'):partial+=1
            else:raise
    return result,partial

def metric(cm):
    k=len(cm);n=sum(map(sum,cm));tp=[cm[i][i] for i in range(k)]
    support=[sum(row) for row in cm];pred=[sum(row[i] for row in cm) for i in range(k)]
    recall=[tp[i]/support[i] if support[i] else 0 for i in range(k)]
    precision=[tp[i]/pred[i] if pred[i] else 0 for i in range(k)]
    f1=[2*tp[i]/(support[i]+pred[i]) if support[i]+pred[i] else 0 for i in range(k)]
    return dict(count=n,correct=sum(tp),accuracy=sum(tp)/n,macro_recall=sum(recall)/k,macro_precision=sum(precision)/k,
        macro_f1=sum(f1)/k,worst_tx_accuracy=min(recall),confusion_matrix=cm,
        per_tx={str(i):dict(count=support[i],correct=tp[i],predicted_count=pred[i],precision=precision[i],recall=recall[i],f1=f1[i]) for i in range(k)})

def independently_score(folder):
    manifest=json.loads((folder/'source_prediction_manifest.json').read_text())
    assert manifest['complete'] and manifest['source_only'] and manifest['prediction_file_closed_before_scoring']
    assert manifest['truth_used_for_prediction'] is False
    # Read and validate every frozen prediction before opening truth.
    predictions,_=records(folder/'source_predictions.jsonl')
    seen=set();counts=Counter()
    for p in predictions:
        key=(p['scene'],p['sample_id']);assert key not in seen;seen.add(key);counts[p['scene']]+=1
        assert 0<=p['prediction']<manifest['num_classes']
        assert not {'truth','y','capture_group'}.intersection(p)
        assert math.isfinite(p['confidence']) and 0<=p['confidence']<=1
    assert dict(counts)==manifest['counts'] and set(counts)==set(SCENES)
    truth_rows,_=records(folder/'source_truth.jsonl');truth={r['sample_id']:r for r in truth_rows}
    assert len(truth)==len(truth_rows)
    for scene in SCENES:assert {p['sample_id'] for p in predictions if p['scene']==scene}==set(truth)
    k=manifest['num_classes'];matrices={};rx_matrices={};day_matrices={}
    def add(store,key,y,p):
        if key not in store:store[key]=[[0]*k for _ in range(k)]
        store[key][y][p]+=1
    for p in predictions:
        t=truth[p['sample_id']];assert p['rx_i']==t['rx_i'] and p['day_i']==t['day_i']
        add(matrices,p['scene'],t['truth'],p['prediction'])
        add(rx_matrices,(p['scene'],str(t['rx_i'])),t['truth'],p['prediction'])
        add(day_matrices,(p['scene'],str(t['day_i'])),t['truth'],p['prediction'])
    result={scene:metric(matrices[scene]) for scene in SCENES}
    saved=json.loads((folder/'source_scores.json').read_text());max_error=0.
    for scene in SCENES:
        result[scene]['per_rx']={rx:metric(cm) for (s,rx),cm in rx_matrices.items() if s==scene}
        result[scene]['per_day']={day:metric(cm) for (s,day),cm in day_matrices.items() if s==scene}
        for key in ('accuracy','macro_recall','macro_f1','worst_tx_accuracy'):
            max_error=max(max_error,abs(result[scene][key]-saved['scenes'][scene][key]))
        for rx,m in result[scene]['per_rx'].items():
            for key in ('accuracy','macro_recall','macro_f1','worst_tx_accuracy'):
                max_error=max(max_error,abs(m[key]-saved['scenes'][scene]['per_rx'][rx][key]))
        for tx,m in result[scene]['per_tx'].items():
            max_error=max(max_error,abs(m['recall']-saved['scenes'][scene]['per_tx_accuracy'][tx]))
        for day,m in result[scene]['per_day'].items():
            for key in ('accuracy','macro_recall','macro_f1','worst_tx_accuracy'):
                max_error=max(max_error,abs(m[key]-saved['scenes'][scene]['per_day'][day][key]))
    assert max_error<1e-12
    return dict(status='VERIFIED_INDEPENDENT_RECOMPUTATION',manifest=manifest,truth_count=len(truth),prediction_count=len(predictions),
        saved_metric_max_absolute_error=max_error,scenes=result)

def action_summary(actions):
    result=dict(records=len(actions),accepted=sum(a.get('accepted') is True for a in actions),
        rejected=sum(a.get('accepted') is not True for a in actions),
        nonfinite_losses=sum(not math.isfinite(a.get('loss',float('nan'))) for a in actions),
        solver_counts=dict(Counter(a.get('main_solver') for a in actions)),
        b8_impl_counts=dict(Counter(a.get('game_b8_impl') for a in actions)),
        action_counts=dict(Counter(a.get('action') for a in actions)))
    for key in ('field_evaluations','main_backward_evaluations','telemetry_backward_evaluations','committed_head_steps','sample_count','satellite_count','unlabeled_samples','pseudo_selected'):
        result[key]=sum(a.get(key,0) or 0 for a in actions)
    result['first_pseudo_epoch']=next((a['epoch'] for a in actions if a.get('pseudo_selected',0)>0),None)
    result['first_u_epoch']=next((a['epoch'] for a in actions if a.get('unlabeled_samples',0)>0),None)
    result['last_epoch']=actions[-1]['epoch'] if actions else None
    result['scenario_counts']=dict(sum((Counter(a.get('actual_scenario_counts',{})) for a in actions),Counter()))
    return result

def remote_collect(run):
    root=Path('/home/szu2070436088/2510044040/CV-SincNet');logs=root/'logs'/run
    state=json.loads((logs/'queue_status.json').read_text());release=Path(state['release'])
    matrix=json.loads((release/'code/configs'/run/'matrix.json').read_text())
    states={r['run_id']:kind for kind in ('completed','active','failed') for r in state[kind]}
    output=dict(schema='core90_v2_full_source_audit_v1',snapshot_started=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        queue=state,rows=[],source_only=True,target_evaluation=False)
    for row in matrix['runs']:
        out=Path(row['config']['output_dir']);kind=states.get(row['run_id'],'pending');live=kind=='active'
        record=next((r for key in ('completed','active','failed') for r in state[key] if r['run_id']==row['run_id']),None)
        result=dict(run_id=row['run_id'],row=row['row'],seed=row['seed'],queue_state=kind,
            process_alive=bool(record and (Path('/proc')/str(record['pid'])).exists()))
        if not out.exists():output['rows'].append(result);continue
        config=json.loads((out/'resolved_config.json').read_text());contract=config.pop('game_data_contract',{})
        role_ids=contract.pop('role_ids',{});contract['role_id_counts']={k:len(v) for k,v in role_ids.items()}
        result['config']=config;result['source_contract']=contract
        result['artifacts']={str(p.relative_to(out)):p.stat().st_size for p in out.rglob('*') if p.is_file()}
        result['epochs'],partial=records(out/'logs.jsonl',live)
        actions,ap=records(out/'game_actions.jsonl',live)
        result['partial_trailing_records']=partial+ap;result['actions']=action_summary(actions)
        result['action_stages']={label:action_summary([a for a in actions if lo<=a['epoch']<=hi]) for label,lo,hi in [('E1_79',1,79),('E80_130',80,130),('E131_200',131,200)]}
        result['stdout']=(logs/(row['run_id']+'.stdout.log')).read_text()
        result['stdout_markers']=[line for line in result['stdout'].splitlines() if re.search(r'Traceback|RuntimeError|OutOfMemory|out of memory|\bKilled\b|Warning|\bnan\b|\binf\b',line,re.I)]
        for filename in ('resource_summary.json','completion.json','backend_configuration.json'):
            if (out/filename).exists():result[filename]=json.loads((out/filename).read_text())
        if kind=='completed':
            assert len(result['epochs'])==200 and [e['epoch'] for e in result['epochs']]==list(range(1,201))
            assert result['completion.json']['status']=='SOURCE_ARTIFACTS_COMPLETE'
            result['independent_scores']=independently_score(out/'source_final_eval')
        output['rows'].append(result)
    completed={r['run_id'] for r in output['rows'] if r['queue_state']=='completed'}
    output['paired_predictions']=[]
    for baseline,variant in [('V2_A','V2_B'),('V2_A','V2_C'),('V2_B','V2_D')]:
        for seed in (392005,392006,392007):
            a=f'{baseline}_seed{seed}';b=f'{variant}_seed{seed}'
            if not {a,b}<=completed:continue
            pa,_=records(root/'runs'/run/a/'source_final_eval/source_predictions.jsonl')
            pb,_=records(root/'runs'/run/b/'source_final_eval/source_predictions.jsonl')
            mapping={(p['scene'],p['sample_id']):p for p in pa}
            assert len(mapping)==len(pb)
            tr,_=records(root/'runs'/run/a/'source_final_eval/source_truth.jsonl');truth={r['sample_id']:r['truth'] for r in tr}
            comparison={s:dict(prediction_disagreements=0,correct_to_wrong=0,wrong_to_correct=0,max_confidence_delta=0.) for s in SCENES}
            for p in pb:
                old=mapping[(p['scene'],p['sample_id'])];c=comparison[p['scene']];y=truth[p['sample_id']]
                c['prediction_disagreements']+=old['prediction']!=p['prediction']
                c['correct_to_wrong']+=old['prediction']==y and p['prediction']!=y
                c['wrong_to_correct']+=old['prediction']!=y and p['prediction']==y
                c['max_confidence_delta']=max(c['max_confidence_delta'],abs(old['confidence']-p['confidence']))
            output['paired_predictions'].append(dict(baseline=a,variant=b,scenes=comparison))
    output['snapshot_finished']=datetime.now(timezone(timedelta(hours=8))).isoformat()
    print(json.dumps(output,allow_nan=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--remote',action='store_true');p.add_argument('--run',default='core90_game_v2_20260911_r3');p.add_argument('--output')
    args=p.parse_args()
    if args.remote:return remote_collect(args.run)
    payload=Path(__file__).read_text(encoding='utf-8')
    result=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607',
        '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-','--remote','--run',args.run],input=payload,text=True,encoding='utf-8',capture_output=True,timeout=180)
    if result.returncode:raise RuntimeError(result.stderr)
    value=json.loads(result.stdout);out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(snapshot=value['snapshot_finished'],states=dict(Counter(r['queue_state'] for r in value['rows'])),
        epoch_records=sum(len(r.get('epochs',[])) for r in value['rows']),action_records=sum(r.get('actions',{}).get('records',0) for r in value['rows']),
        prediction_records=sum(r.get('independent_scores',{}).get('prediction_count',0) for r in value['rows'])),indent=2))

if __name__=='__main__':main()
