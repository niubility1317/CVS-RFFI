"""Read every currently flushed source log record; never open target data."""
import argparse
import json
import pathlib
import statistics
import zlib

RUN = '20261110-phase1-pamf-branch-relations-f0-s2026092701-m4-r01'
PARENT = '20261022-phase1-T1-DURA-F-ablation-binding-m63-r01'


def epoch_rows(path, offset):
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding='utf-8').splitlines(keepends=True):
        if line.endswith('\n') and line.strip():
            value = json.loads(line)
            records.append(dict(value, cumulative_epoch=value['epoch'] + offset))
    return records


def complete_steps(path):
    if not path.is_file():
        return [], False
    decoder = zlib.decompressobj(31)
    raw = decoder.decompress(path.read_bytes())
    lines = raw.split(b'\n')
    if lines[-1]:
        lines.pop()  # Current write may stop between records.
    return [json.loads(line)['metrics'] for line in lines if line], decoder.eof


def summarize_steps(steps):
    result = dict(records=len(steps))
    for key in ('branch_L_cross_anchors', 'branch_U_cross_anchors', 'branch_L_loss', 'branch_U_loss'):
        vals = [float(m[key]) for m in steps if key in m]
        if vals:
            result[key] = dict(records=len(vals), positive=sum(x > 0 for x in vals),
                               minimum=min(vals), maximum=max(vals), mean=statistics.fmean(vals))
    for key in ('HPL_seen', 'HPL_accepted', 'T1_PL_seen', 'T1_PL_accepted'):
        vals = [m[key] for m in steps if key in m]
        if vals:
            result[key] = vals[-1]
    result['nonfinite_skip_records'] = sum(bool(m.get('train/skipped_nonfinite_loss', 0)
                                               or m.get('train/skipped_nonfinite_grad', 0)) for m in steps)
    return result


def analyze(project):
    project = pathlib.Path(project)
    root = project / 'runs' / RUN
    state = json.loads((root / 'pipeline_state.json').read_text())
    parents = {}
    for arm in ('F20', 'F21'):
        folder = project / 'runs' / PARENT / (arm + '-s2026092701') / 'source'
        epochs, records = [], {}
        for end, offset in ((20, 0), (30, 20), (50, 30)):
            part = folder / ('segment' + str(end))
            epochs.extend(epoch_rows(part / 'metrics_epoch.jsonl', offset))
            values, closed = complete_steps(part / 'step_metrics.jsonl.gz')
            records[str(end)] = dict(summary=summarize_steps(values), stream_closed=closed)
        parents[arm] = dict(epochs=epochs, steps=records, stdout='not located in source segment directory')
    results = {}
    for row, arm in (('sup_l', 'F20'), ('pl_l', 'F21'), ('pl_u', 'F21'), ('pl_lu', 'F21')):
        rid = row + '-s2026092701'
        folder = root / 'F' / rid / 'source'
        epochs, steps = [], {}
        for end, offset in ((30, 20), (50, 30)):
            part = folder / ('segment' + str(end))
            epochs.extend(epoch_rows(part / 'metrics_epoch.jsonl', offset))
            values, closed = complete_steps(part / 'step_metrics.jsonl.gz')
            steps[str(end)] = dict(summary=summarize_steps(values), stream_closed=closed)
        old = {r['cumulative_epoch']: r for r in parents[arm]['epochs']}
        aligned = [dict(epoch=r['cumulative_epoch'], new=r['val_tx_acc'], old=old[r['cumulative_epoch']]['val_tx_acc'],
                        delta_pp=r['val_tx_acc'] - old[r['cumulative_epoch']]['val_tx_acc'])
                   for r in epochs if r['cumulative_epoch'] in old and r.get('val_tx_acc') is not None
                   and old[r['cumulative_epoch']].get('val_tx_acc') is not None]
        log = root / 'logs' / ('train__' + rid + '.log')
        text = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
        errors = [marker for marker in ('Traceback (most recent call last)', 'CUDA out of memory',
                   'FloatingPointError', 'CHECKPOINT_DATA_CONTRACT_MISMATCH') if marker in text]
        results[row] = dict(status=state['jobs']['train__' + rid]['status'], source_epochs=aligned,
                            steps=steps, stdout_characters=len(text), error_markers=errors)
    compact = {arm: dict(epochs=[dict(epoch=r['cumulative_epoch'], val_tx_acc=r.get('val_tx_acc'))
                                  for r in value['epochs']],
                         steps=value['steps'], stdout=value['stdout']) for arm, value in parents.items()}
    return dict(run=RUN, scope='all complete currently flushed records; ongoing source training only',
                phase1_target_reads=0, status=state['status'], parents=compact, rows=results)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project', required=True)
    a = p.parse_args()
    print(json.dumps(analyze(a.project), allow_nan=False))
