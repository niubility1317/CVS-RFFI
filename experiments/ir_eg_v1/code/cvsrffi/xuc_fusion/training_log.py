"""Read-only CVS-style text and compact AI telemetry from accepted records.

This module deliberately has no torch/model/dataset dependency. It never runs
an extra forward pass, reads U truth, or changes the optimization transaction.
"""
import csv
import json
import math
from collections import Counter
from pathlib import Path

SCHEMA = 'cvs_training_epoch_v1'


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value):
    if value is None:
        return 'N/A'
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def flatten(value, prefix=''):
    """Keep scalars only; physical IDs, confusion matrices and arrays stay raw."""
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f'{prefix}.{key}' if prefix else str(key)
            result.update(flatten(item, name))
    elif not isinstance(value, (list, tuple)):
        result[prefix] = value
    return result


def aggregate(records, key):
    groups = {}
    for row in records:
        for name, value in flatten(row.get(key, {})).items():
            groups.setdefault(name, []).append(value)
    output = {}
    for name, values in groups.items():
        if all(isinstance(v, bool) for v in values):
            output[name + '.true_count'] = sum(values)
            output[name + '.observed_count'] = len(values)
        elif any(number(v) for v in values):
            valid = [v for v in values if number(v)]
            output[name + '.mean'] = sum(valid) / len(valid)
            output[name + '.min'] = min(valid)
            output[name + '.max'] = max(valid)
            output[name + '.last'] = values[-1] if number(values[-1]) else None
            output[name + '.observed_count'] = len(valid)
        elif all(v is None or isinstance(v, str) for v in values):
            output[name + '.last'] = values[-1]
    return output


def epoch_metrics(summary, records, *, row_id, epochs, previous_elapsed):
    """Each scalar declares its reduction; absent measurements remain null."""
    if len(records) != summary['accepted']:
        raise ValueError('epoch record count differs from accepted summary')
    if any(not row.get('accepted') for row in records):
        raise ValueError('rejected steps cannot enter accepted epoch telemetry')
    val = summary.get('source_validation', {})
    grouped = val.get('group_metrics', {})
    overall = grouped.get('overall', {})
    elapsed = summary['elapsed_seconds']
    metrics = dict(schema=SCHEMA, row_id=row_id, epoch=summary['epoch'], epochs=epochs,
        accepted_steps=summary['accepted'], total_step=summary['total_step'],
        epoch_seconds=elapsed-previous_elapsed, elapsed_seconds=elapsed,
        train_loss_mean=summary['mean_loss'], train_accuracy=None,
        source_val_accuracy=val.get('accuracy'), source_val_ce=val.get('ce'),
        source_val_macro_f1=overall.get('macro_f1'), source_val_count=val.get('total'),
        target_evaluated=False, target_accuracy=None)
    for name, value in summary.get('terms', {}).items():
        metrics['loss_terms.' + name + '.mean'] = value
    for key in ('learning_rates', 'weights', 'daot_rc4', 'solver_telemetry', 'identity_supervision_exposure'):
        metrics.update({key + '.' + name: value for name, value in aggregate(records, key).items()})
    for key in ('grad_norm', 'cuda_peak_memory_bytes', 'field_evaluations'):
        values = [r.get(key) for r in records if number(r.get(key))]
        metrics[key + '.mean'] = sum(values)/len(values) if values else None
        metrics[key + '.max'] = max(values) if values else None
    metrics['scenario_counts'] = dict(Counter(str(r.get('scenario')) for r in records))
    metrics['cg_status_counts'] = dict(Counter((r.get('solver_telemetry') or {}).get('cg_status')
        for r in records if (r.get('solver_telemetry') or {}).get('cg_status') is not None))
    metrics['missing_reasons'] = dict(train_accuracy='not recorded by this trainer; not reconstructed from loss',
        target_accuracy='source-only training; target not evaluated')
    return metrics


def pairs(metrics, prefix, suffix='.mean'):
    return ' '.join(f'{key[len(prefix):-len(suffix)]}={fmt(value)}'
        for key, value in sorted(metrics.items()) if key.startswith(prefix) and key.endswith(suffix)) or 'N/A'


def format_epoch(metrics):
    m = metrics
    rows = ['=' * 120,
        f"[EPOCH-BEGIN] E{m['epoch']:03d}/{m['epochs']:03d} | row={m['row_id']} | time={m['epoch_seconds']:.1f}s | elapsed={m['elapsed_seconds']:.1f}s",
        f"[TRAIN] accepted={m['accepted_steps']} total_step={m['total_step']} loss={fmt(m['train_loss_mean'])} grad_norm_mean={fmt(m['grad_norm.mean'])} grad_norm_max={fmt(m['grad_norm.max'])} train_acc=N/A(not_recorded)",
        '[LR] ' + pairs(m, 'learning_rates.', '.last'),
        '[LOSS-ORIGIN-MEAN] ' + pairs(m, 'loss_terms.'),
        '[WEIGHTS-LAST] ' + pairs(m, 'weights.', '.last'),
        f"[VAL-SOURCE] acc={fmt(m['source_val_accuracy'])} macro_f1={fmt(m['source_val_macro_f1'])} ce={fmt(m['source_val_ce'])} n={fmt(m['source_val_count'])}",
        '[DAOT-RC4-MEAN] ' + pairs(m, 'daot_rc4.'),
        '[DAOT-RC4-COUNTS] ' + pairs(m, 'daot_rc4.', '.true_count'),
        '[IR-COUNTS] ' + pairs(m, 'solver_telemetry.', '.true_count'),
        '[IR-MEAN] ' + pairs(m, 'solver_telemetry.'),
        '[IR-CG] ' + json.dumps(m['cg_status_counts'], sort_keys=True),
        '[EXPOSURE-MEAN] ' + pairs(m, 'identity_supervision_exposure.'),
        '[SATELLITE] ' + json.dumps(m['scenario_counts'], sort_keys=True),
        f"[RESOURCE] cuda_peak_bytes={fmt(m['cuda_peak_memory_bytes.max'])} fields_per_step={fmt(m['field_evaluations.mean'])}",
        '[EVAL-POLICY] target=NOT_EVALUATED source_V=read_only accuracy_unit=fraction',
        f"[EPOCH-END] E{m['epoch']:03d}/{m['epochs']:03d}"]
    return '\n'.join(rows)


class TrainingLog:
    def __init__(self, output, config, *, previous_elapsed=0., emit=print):
        self.output = Path(output)
        self.row_id = config['xuc_row']['id']
        self.epochs = config['epochs']
        self.previous_elapsed = previous_elapsed
        self.rows = []
        self.stdout = emit
        self.text_path = self.output/'training.log'
        self.text_path.open('x', encoding='utf-8').close()
        self.path = self.output/'training_metrics.jsonl'
        self.path.open('x', encoding='utf-8').close()
        config_path = self.output/'training_config.json'
        with config_path.open('x', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2, default=str)
            f.write('\n')
        self.emit('[CVS-LOG] detailed_text=enabled compact_jsonl=training_metrics.jsonl csv=training_metrics.csv')
        self.emit('[CONFIG-BEGIN] resolved runtime parameters; see training_config.json')
        for key, value in sorted(config.items()):
            if key != 'ir_import_audit':
                self.emit('[CONFIG] ' + key + '=' + json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
        self.emit('[CONFIG-END]')

    def emit(self, text):
        with self.text_path.open('a', encoding='utf-8') as f:
            f.write(text+'\n')
        self.stdout(text)

    def step(self, record, steps_per_epoch):
        step = record.get('accepted_main_step', record['step']+1)
        if step % 50 != 0:
            return
        batch = (step-1) % steps_per_epoch+1
        self.emit(f"[TRAIN-STEP] E{record['execution_epoch']:03d}/{self.epochs:03d} batch={batch}/{steps_per_epoch} accepted_step={step} loss={fmt(record['loss'])} grad_norm={fmt(record['grad_norm'])} lr={json.dumps(record.get('learning_rates', {}),sort_keys=True)}")

    def epoch(self, summary, records):
        metrics = epoch_metrics(summary, records, row_id=self.row_id, epochs=self.epochs,
                                previous_elapsed=self.previous_elapsed)
        self.previous_elapsed = summary['elapsed_seconds']
        # Null is intentional; neither missing training accuracy nor target metrics
        # are fabricated. Source terms retain their native normalization semantics.
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False, sort_keys=True)+'\n')
        self.rows.append(flatten(metrics))
        columns = ['schema', 'row_id', 'epoch', 'epochs', 'accepted_steps', 'total_step']
        columns += sorted(set().union(*(r.keys() for r in self.rows))-set(columns))
        temporary = self.output/'training_metrics.csv.tmp'
        with temporary.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)
        temporary.replace(self.output/'training_metrics.csv')
        self.emit(format_epoch(metrics))
        return metrics
