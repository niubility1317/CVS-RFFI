"""Read immutable completed scores and live progress; never open query/truth."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.standard_ce_baselines.publish import ssh
from experiments.standard_ce_baselines.common import PROJECT,RUN,EXTENSION_RUN

REMOTE=r'''
import datetime,json
from pathlib import Path
project=Path(PROJECT);rows=[];progress=[];states={}
def read(p):return json.loads(p.read_text())
for run in RUNS:
    folder=project/'runs'/run;state=read(folder/'pipeline_state.json');states[run]=state['status']
    for rid,entry in state['rows'].items():
        if entry['variant']=='native':continue
        source=folder/rid/'source'
        epoch=source/'epoch_metrics.jsonl'
        last=json.loads(epoch.read_text().splitlines()[-1]) if epoch.exists() and epoch.stat().st_size else None
        process=Path('/proc')/str(entry.get('pid',-1))
        progress.append(dict(run_id=run,row_id=rid,variant=entry['variant'],model_seed=entry['model_seed'],
            status=entry['status'],epoch=last.get('epoch') if last else None,steps=last.get('optimizer_steps_total',last.get('steps')) if last else None,
            process_alive=process.exists(),gpu=entry.get('gpu'),error=entry.get('error')))
        if entry['status']!='SCORED':continue
        prediction=folder/rid/'prediction';scores=folder/rid/'scores'
        done=read(source/'completion.json');frozen=read(source/'freeze.json')
        pred=read(prediction/'complete.json');scored=read(scores/'complete.json');value=read(scores/'scores.json')
        if done['epoch']!=200 or done['steps']!=10000 or frozen['status']!='FROZEN':raise ValueError('Incomplete fixed training')
        if pred['status']!='PREDICTIONS_COMPLETE' or pred['count']!=168000 or pred['truth_read'] is not False or pred['query_fit'] is not False or pred['views']!=['clean']:raise ValueError('Incomplete legal prediction')
        if scored['status']!='SCORED_COMPLETE' or scored['count']!=168000 or scored['truth_last'] is not True or value['truth_last'] is not True or len(value['results'])!=14:raise ValueError('Incomplete scoring')
        for metric in value['results']:
            if metric['model_seed']!=entry['model_seed'] or metric['variant']!=entry['variant']:raise ValueError('Wrong row score')
        init=read(source/'initialization.json')
        if init['scratch_only'] is not True or init['ancestors'] or init['target_access']:raise ValueError('Non-scratch row')
        profile=read(source/'resource_profile.json')
        inference=read(prediction/'resource_profile.json') if (prediction/'resource_profile.json').exists() else None
        resolved=read(source/'resolved_config.json')
        rows.append(dict(row=dict(run_id=run,row_id=rid,variant=entry['variant'],model_seed=entry['model_seed']),
            scores=value,resource_profile=profile,inference_profile=inference,
            training_config={k:resolved[k] for k in ('variant','epochs','steps_per_epoch','batch_size','lr','lr_min','weight_decay','split_seed','source_counts','precision','optimizer','augmentation','U_s_use','target_access','total_parameters','trainable_parameters')},
            completion={k:done.get(k) for k in ('epoch','steps','elapsed_seconds')},
            prediction_complete={k:pred.get(k) for k in ('count','inference_seconds','peak_cuda_allocated_bytes')},
            training_peak_cuda_allocated_bytes=last.get('peak_cuda_allocated_bytes') if last else None,
            classes=read(source/'source_contract.json')['classes']))
print(json.dumps(dict(read_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    states=states,progress=progress,rows=rows)))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous progress snapshots')
    script=REMOTE.replace('PROJECT',repr(PROJECT)).replace('RUNS',repr([RUN,EXTENSION_RUN]))
    value=json.loads(ssh(script));a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(read_at_utc=value['read_at_utc'],states=value['states'],
        new_progress=dict(Counter(r['status'] for r in value['progress'] if r['run_id']==EXTENSION_RUN)),
        scored_total=len(value['rows']))))

if __name__=='__main__':main()
