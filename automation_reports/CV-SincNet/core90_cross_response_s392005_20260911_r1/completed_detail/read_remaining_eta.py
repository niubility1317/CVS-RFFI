import json
import time
from pathlib import Path
root=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/core90_cross_response_s392005_20260911_r1')
state=json.loads((root/'pipeline.json').read_text())
rows=[]
for job in state['jobs']:
    if job['status']!='RUNNING':
        continue
    metrics=[json.loads(line) for line in (Path(job['output'])/'metrics_epoch.jsonl').read_text().splitlines() if line.strip()]
    window=metrics[-10:]
    seconds=sum(m['epoch_time_s'] for m in window)/len(window)
    rows.append(dict(variant=job['variant'],epoch=metrics[-1]['epoch'],phase=metrics[-1]['phase'],
        recent10_seconds_per_epoch=seconds,remaining_training_hours=(200-metrics[-1]['epoch'])*seconds/3600,
        latest_update_fraction=metrics[-1].get('train_optimizer_step_applied'),
        latest_nonfinite_grad_fraction=metrics[-1].get('train_skipped_nonfinite_grad')))
print(json.dumps(dict(time=time.time(),rows=rows)))
