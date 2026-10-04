from pathlib import Path
import json,statistics
root=Path('E:/type10-7/code/snapshots/daot_practical_three_20260918_wt/local_artifacts/cvs_reference_stack_speed_20261004_r01')
d=json.loads((root/'full_log_scan.json').read_text(encoding='utf-8'))
out={'started':d['started'],'finished':d['finished'],'rows':[],'r1':[]}
for r in d['rows']:
    e=r['structured']['epoch_metrics.jsonl']['rows'];a=r['resolved_native_args.json'];phases={}
    for lo,hi in [(1,40),(41,79),(80,90),(91,130),(131,200)]:
        x=[x for x in e if lo<=x['epoch']<=hi];t=[x['epoch_time_s'] for x in x if isinstance(x.get('epoch_time_s'),(int,float))]
        phases[f'{lo}-{hi}']={'epochs':len(t),'mean_s':statistics.mean(t) if t else None,'median_s':statistics.median(t) if t else None}
    last=e[-1];out['rows'].append(dict(row=r['row_id'],epoch=last['epoch'],steps=r['steps']['count'],phases=phases,last10_mean_s=statistics.mean([x['epoch_time_s'] for x in e[-10:]]),args={k:v for k,v in a.items() if any(s in k for s in ['batch','heavy_eval','num_workers','gradient_snapshot','runtime_fast','validation_reuse','incremental','daot_efficiency'])},last_metrics={k:v for k,v in last.items() if any(s in k for s in ['acc','loss','grad','pseudo','skipped','successful'])},files=r['files'],stdout=r['stdout']))
for r in d['r1_rows']:
    e=r['epochs'];keys=[k for k in e[-1] if any(s in k for s in ['seconds','time_s','duration','updates','steps'])]
    out['r1'].append(dict(row=r['row_id'],count=len(e),timing_keys=keys,last={k:e[-1][k] for k in keys},mean={k:statistics.mean([x[k] for x in e if isinstance(x.get(k),(int,float))]) for k in keys if any(isinstance(x.get(k),(int,float)) for x in e)},completion=r['completion']))
(root/'full_log_summary.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'steps':sum(r['steps'] for r in out['rows']),'rows':[{k:r[k] for k in ['row','epoch','steps','phases','last10_mean_s']} for r in out['rows']],'args':out['rows'][0]['args'],'r1':out['r1'][:2]},indent=2))
