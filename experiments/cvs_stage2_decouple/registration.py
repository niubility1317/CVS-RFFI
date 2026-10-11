"""Update the existing run registration; never read metrics for training decisions."""
import time
from pathlib import Path

from . import design as d


def record(status, row_id=None, **values):
    import json
    from experiments.cvs_receiver_residual_capacity4.control import lock
    with lock(d.BASE / 'registration.lock'):
        item = dict(at=time.time(), status=status, **values)
        if row_id is not None:
            item['row_id'] = row_id
        with (d.BASE / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
        path = d.BASE / 'experiment.json'
        if path.is_file():
            spec = d.read(path)
            if row_id is None and (status in ('RUNNING', 'FAILED', 'ANALYZED') or 'model_seed' not in values):
                spec['status'] = status
            else:
                for row in spec.get('rows', []):
                    if row['row_id'] == row_id:
                        row.update(status=status, latest_event=item)
                        break
            spec['updated_at_unix'] = item['at']
            d.write(path, spec)


def initialize():
    template = d.ROOT / 'experiments/cvs_stage2_decouple/experiment.json'
    if not template.is_file():
        raise FileNotFoundError('Preregistration is required before dispatch')
    spec = d.read(template)
    if spec['run_id'] != d.RUN or {r['row_id'] for r in spec['rows']} != {r['row_id'] for r in d.rows()}:
        raise ValueError('Registered run or fixed matrix differs')
    spec['status'] = 'RUNNING'
    spec['execution_commit'] = (d.ROOT / 'release_commit.txt').read_text().strip()
    spec['code']['commit'] = spec['execution_commit']
    spec['execution']['release_commit'] = spec['execution_commit']
    d.write(d.BASE / 'experiment.json', spec)
    report = d.ROOT / 'experiments/cvs_stage2_decouple/report.md'
    if report.is_file():
        (d.BASE / 'report.md').write_text(report.read_text(encoding='utf-8'), encoding='utf-8')


def finalize_report():
    # Called only from the terminal controller after independent scoring.
    source = d.BASE / 'analysis.md'
    if not source.is_file():
        raise FileNotFoundError('Final scored report missing')
    path = d.BASE / 'report.md'
    prior = path.read_text(encoding='utf-8') if path.is_file() else ''
    path.write_text(prior + '\n\n' + source.read_text(encoding='utf-8'), encoding='utf-8')
