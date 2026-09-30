"""Full-log execution audit; no model/performance selection or sampled parsing."""
import argparse
import csv
import io
import json
import math
import re
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--output',required=True); args=p.parse_args()
out=Path(args.output); audit=out/'full_log_audit.json'
if audit.exists(): raise FileExistsError(audit)
rows=[json.loads(line) for line in (out/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
csv_rows=list(csv.DictReader(io.StringIO((out/'steps.csv').read_text(encoding='utf-8'))))
stdout=out.with_suffix('.stdout.log').read_text(encoding='utf-8')
stdout_lines=stdout.splitlines(); stdout_json=[json.loads(line) for line in stdout_lines if line.startswith('{')]
assert len(rows)==len(csv_rows)
assert len(stdout_json)==len(rows)+2,'Missing startup, step or completion stdout records'
errors=[line for line in stdout_lines if re.search(r'Traceback|RuntimeError|CUDA out of memory|(^|\W)(nan|inf|killed)(\W|$)',line,re.I)]
assert not errors,errors
stats={}
for row in rows:
    key=row['stage']; stage=stats.setdefault(key,{'steps':0,'epochs':{},'summary_epochs':[], 'grad_min':None,'grad_max':None})
    if row.get('summary',False):
        stage['summary_epochs'].append(row['epoch'])
        assert math.isfinite(row['loss']) and math.isfinite(row['seconds'])
        continue
    stage['steps']+=1; epoch=stage['epochs'].setdefault(str(row['epoch']),[])
    epoch.append(row['loss'])
    assert math.isfinite(row['loss']) and math.isfinite(row['grad_norm'])
    stage['grad_min']=row['grad_norm'] if stage['grad_min'] is None else min(stage['grad_min'],row['grad_norm'])
    stage['grad_max']=row['grad_norm'] if stage['grad_max'] is None else max(stage['grad_max'],row['grad_norm'])
for stage in stats.values():
    assert len(stage['epochs'])==len(stage['summary_epochs'])
    stage['epochs']={epoch:{'steps':len(values),'loss_mean':sum(values)/len(values),'loss_first':values[0],'loss_last':values[-1],
                            'loss_min':min(values),'loss_max':max(values)} for epoch,values in stage['epochs'].items()}
result={'status':'VERIFIED','structured_rows':len(rows),'csv_rows':len(csv_rows),'stdout_lines':len(stdout_lines),
        'all_logs_read_fully':True,'error_markers':errors,'stages':stats,
        'convergence_verdict':'NOT_ESTABLISHED: two epochs verify execution, not convergence',
        'source_validation':'N/A: absent in diagnostic; no query used for selection'}
audit.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'VERIFIED','structured_rows':len(rows),'csv_rows':len(csv_rows),'stdout_lines':len(stdout_lines),'stages':list(stats)}))
