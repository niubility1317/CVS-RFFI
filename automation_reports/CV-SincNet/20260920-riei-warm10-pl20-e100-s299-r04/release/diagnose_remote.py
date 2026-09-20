import json,re
from pathlib import Path
import numpy as np
base=Path('/home/szu2070436088/2510044040/STAR-RFFI/STAR-test/补实验')
root=base/'20260920-riei-original-concat-pl-s299-e200-r03'
run=next((root/'runs').iterdir())
epochs=[json.loads(x) for x in (run/'metrics_epoch.jsonl').read_text().splitlines()]
batches=[json.loads(x) for x in (run/'batches.jsonl').read_text().splitlines()]
stdout=next((root/'logs').glob('*.log')).read_text(errors='replace')
final=json.loads((run/'final_last_epoch/student/metrics.json').read_text())
independent={}
for name in ['source_clean','source_sg','target_clean','target_sg']:
    d=run/'final_last_epoch/student';pred=np.load(d/(name+'_probabilities.npy')).argmax(1);truth=np.load(d/(name+'_truth.npy'))
    independent[name]=dict(n=len(truth),accuracy=float((pred==truth).mean()))
curve=[dict(epoch=e['epoch'],**{k:v['accuracy'] for k,v in e.get('evaluation',{}).items()}) for e in epochs if 'evaluation' in e]
windows=[]
for start,end in [(1,10),(11,50),(51,99),(100,120),(181,200)]:
    rows=[r for r in batches if start<=r['epoch']<=end]
    windows.append(dict(start=start,end=end,batches=len(rows),means={k:float(np.mean([r[k] for r in rows])) for k in ['ce_clean','ce_sg','rx_ce','mi','ie','pseudo_ce','weighted_pseudo_ce','selected','selected_clean','selected_sg']},
                        acceptance=sum(r['selected'] for r in rows)/sum(r['unlabeled_views'] for r in rows)))
print(json.dumps(dict(status='COMPLETE',epochs=len(epochs),batches=len(batches),curve=curve,windows=windows,final=final,independent=independent,
    errors=re.findall(r'^.*(?:Traceback|Error|nonfinite|NaN|CUDA out of memory).*$' ,stdout,re.M),
    config=json.loads((run/'resolved_config.json').read_text())),ensure_ascii=False))
