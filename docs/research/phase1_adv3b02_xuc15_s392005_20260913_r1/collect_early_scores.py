"""Read existing independent scores and coverage metadata; never predict or read truth."""
import json
import time
from pathlib import Path

project = Path('/home/szu2070436088/2510044040/CV-SincNet')
root = project / 'runs/phase1_adv3b02_xuc13_early_eval_s392005_20260913_r1'
state = json.loads((root/'evaluation_state.json').read_text())
out = {'time':time.time(), 'state':state, 'scores':{}, 'prediction_mtimes':{}, 'score_mtimes':{}, 'logs':{}}
for rid, row in state['rows'].items():
    pred = root/rid/'predictions.json'
    score = root/rid/'score.json'
    if row['status'] in ('PREDICTIONS_FIXED','SCORED'):
        out['prediction_mtimes'][rid] = pred.stat().st_mtime
    if row['status'] == 'SCORED':
        value = json.loads(score.read_text())
        assert value['record_count'] == 672000
        assert set(value['metrics']) == {'clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak'}
        assert all(v['total'] == 168000 for v in value['metrics'].values())
        assert value['prediction_path'] == str(pred)
        out['scores'][rid] = value
        out['score_mtimes'][rid] = score.stat().st_mtime
for log in (project/'logs'/root.name).glob('*.log'):
    text = log.read_text(errors='replace')
    out['logs'][log.name] = {'bytes':log.stat().st_size, 'fatal': any(s in text for s in ('Traceback (most recent call last)', 'CUDA out of memory'))}
if state['status'] == 'COMPLETE':
    assert len(out['scores']) == 13
    fixed = json.loads((root/'predictions_complete.json').read_text())
    assert fixed['record_count'] == 8736000
    assert max(out['prediction_mtimes'].values()) <= fixed['created'] <= min(out['score_mtimes'].values())
    out['verification'] = 'VERIFIED_COMPLETE_13_ROWS_4_SCENARIOS_TRUTH_LAST'
else:
    out['verification'] = 'IN_PROGRESS'
print(json.dumps(out, ensure_ascii=False))
