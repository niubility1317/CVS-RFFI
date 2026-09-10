import json
from pathlib import Path
root = Path('/home/szu2070436088/2510044040/CV-SincNet')
run = 'core90_cross_response_s392005_20260911_r1'
state = json.loads((root/'runs'/run/'pipeline.json').read_text())
rows = []
for job in state['jobs']:
    path = Path(job['output'])
    metrics = [json.loads(line) for line in (path/'metrics_epoch.jsonl').read_text().splitlines() if line.strip()]
    log = Path(job['log']).read_text(errors='replace')
    rows.append(dict(variant=job['variant'], status=job['status'], epochs=len(metrics),
        first_joint_epoch=next((m['epoch'] for m in metrics if m.get('train_cross_response_response_joint_open', 0)>0), None),
        min_update=min(m.get('train_optimizer_step_applied', 0) for m in metrics),
        nonfinite_loss_skips=sum(m.get('train_skipped_nonfinite_loss', 0) for m in metrics),
        nonfinite_grad_skips=sum(m.get('train_skipped_nonfinite_grad', 0) for m in metrics),
        max_shared_response=max(m.get('train_cross_response_response_grad_shared', 0) for m in metrics),
        max_front_response=max(m.get('train_cross_response_response_grad_identity_front', 0) for m in metrics),
        max_tail_response=max(m.get('train_cross_response_response_grad_identity_tail', 0) for m in metrics),
        max_decision=max(m.get('train_cross_response_decision_identity_grad_norm', 0) for m in metrics),
        errors=[line for line in log.splitlines() if any(t in line for t in ('Traceback', 'Error:', 'CUDA out of memory'))],
        cross_records=[{k:v for k,v in m.items() if 'cross_response' in k or k=='epoch'} for m in metrics]))
print(json.dumps(dict(status=state['status'], rows=rows)))
