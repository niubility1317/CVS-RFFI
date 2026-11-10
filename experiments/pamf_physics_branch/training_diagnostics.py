"""Summarize every finalized source training step for the fixed PTBR pilot."""
import argparse
import json
from pathlib import Path
import statistics

from experiments.pamf_branch_training.source_diagnostics import complete_steps, epoch_rows


def stats(values):
    return dict(count=len(values), positive=sum(v > 0 for v in values),
                minimum=min(values), maximum=max(values), mean=statistics.fmean(values))


def summarize(root):
    root = Path(root)
    state = json.loads((root / 'pipeline_state.json').read_text(encoding='utf-8'))
    rows = {}
    for tag in ('beta0', 'beta05'):
        rid = 'ptbr-' + tag + '-s2026092701'
        if state['jobs']['train__' + rid]['status'] != 'COMPLETE':
            raise ValueError('Both source models must have completed before full-log diagnosis')
        source = root / 'F' / rid / 'source'
        epochs, segments = [], {}
        for end, offset in ((30, 20), (50, 30)):
            path = source / ('segment' + str(end))
            steps, closed = complete_steps(path / 'step_metrics.jsonl.gz')
            expected = (end - offset) * 222
            if not closed or len(steps) != expected:
                raise ValueError('Source step stream incomplete: ' + str(path))
            epochs.extend(epoch_rows(path / 'metrics_epoch.jsonl', offset))
            keys = ('physical_L_valid_anchors', 'physical_U_valid_anchors',
                    'physical_L_loss', 'physical_U_loss', 'physical_L_distance_correction',
                    'physical_U_distance_correction', 'physical_L_samples', 'physical_U_samples')
            keys += tuple('physical_' + role + '_direction_' + branch for role in ('L', 'U')
                          for branch in ('time', 'frequency', 'pa', 'path', 'reference'))
            summary = {}
            for key in keys:
                values = [float(m[key]) for m in steps if key in m]
                if len(values) != expected:
                    raise ValueError('Source metric coverage missing: ' + key)
                summary[key] = stats(values)
            skips = sum(bool(m.get('train/skipped_nonfinite_loss', 0)
                             or m.get('train/skipped_nonfinite_grad', 0)) for m in steps)
            exposure = json.loads((path / 'physical_state.json').read_text(encoding='utf-8'))
            segments[str(end)] = dict(step_count=expected, stream_closed=closed,
                                      nonfinite_skip_records=skips, metrics=summary, exposure=exposure)
        if len(epochs) != 30:
            raise ValueError('Expected full E21–E50 epoch records')
        logfile = (root / 'logs' / ('train__' + rid + '.log')).read_text(encoding='utf-8', errors='replace')
        errors = [s for s in ('Traceback (most recent call last)', 'CUDA out of memory', 'FloatingPointError',
                              'CHECKPOINT_DATA_CONTRACT_MISMATCH') if s in logfile]
        rows[tag] = dict(epochs=[dict(epoch=e['cumulative_epoch'], source_V_accuracy=e.get('val_tx_acc')) for e in epochs],
                         segments=segments, stdout_characters=len(logfile), error_markers=errors)
    return dict(status='COMPLETE_SOURCE_LOG_DIAGNOSIS', source_only=True, target_reads=0, rows=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    print(json.dumps(summarize(args.root), allow_nan=False))
