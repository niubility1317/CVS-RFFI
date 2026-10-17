"""Read-only complete-log collection for the fixed full-source reference.

Runs independently of the frozen training release. No IQ, checkpoint, teacher,
truth or model is opened. Closed results are copied only from existing scores.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import itertools
import json
import math
from pathlib import Path
import re
import sys

ARMS = ('NATIVE', 'REPEAT', 'CONCAT')
SEED = 2026092701


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else None


def records(path, state):
    state.update(exists=path.is_file(), records=0, partial=False, error=None)
    if not state['exists']:
        return
    opener = gzip.open if path.suffix == '.gz' else open
    try:
        with opener(path, 'rt', encoding='utf8') as stream:
            for line in stream:
                if not line.endswith('\n'):
                    state['partial'] = True
                    break
                if not line.strip():
                    continue
                value = json.loads(line)
                state['records'] += 1
                yield value
    except (EOFError, gzip.BadGzipFile, json.JSONDecodeError, OSError) as exc:
        state.update(partial=True, error=type(exc).__name__)


class Moments:
    def __init__(self):
        self.n = 0
        self.mean = 0.
        self.m2 = 0.
        self.lo = None
        self.hi = None
        self.first = None
        self.last = None
        self.nonfinite = 0

    def add(self, value):
        value = float(value)
        if not math.isfinite(value):
            self.nonfinite += 1
            return
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.lo = value if self.lo is None else min(self.lo, value)
        self.hi = value if self.hi is None else max(self.hi, value)
        if self.first is None:
            self.first = value
        self.last = value

    def result(self):
        return dict(count=self.n, mean=self.mean if self.n else None,
                    std=math.sqrt(max(0., self.m2 / self.n)) if self.n else None,
                    minimum=self.lo, maximum=self.hi, first=self.first,
                    last=self.last, nonfinite=self.nonfinite)


def flatten(value, prefix=''):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, prefix + '/' + str(key) if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from flatten(item, prefix + '/' + str(index))
    elif isinstance(value, (int, float)):
        yield prefix, value


def numeric_logs(path, metrics_key=None):
    state = {}
    total = defaultdict(Moments)
    epochs = defaultdict(lambda: defaultdict(Moments))
    epoch_counts = Counter()
    step_order = dict(first=None, last=None, consecutive=True)
    for item in records(path, state):
        step = item.get('step')
        if step is not None:
            if step_order['first'] is None:
                step_order['first'] = step
            if step_order['last'] is not None and step != step_order['last'] + 1:
                step_order['consecutive'] = False
            step_order['last'] = step
        epoch = int(item.get('epoch', ((step - 1) // 222 + 1) if step else 0))
        epoch_counts[epoch] += 1
        for key, value in flatten(item.get(metrics_key, {}) if metrics_key else item):
            total[key].add(value)
            epochs[epoch][key].add(value)
    state.update(step_order=step_order, epoch_record_counts=dict(sorted(epoch_counts.items())),
                 totals={k:v.result() for k,v in sorted(total.items())},
                 epochs={str(e):{k:v.result() for k,v in sorted(m.items())}
                         for e,m in sorted(epochs.items())})
    return state


def sequence(path):
    state = {}
    packets = Counter()
    rx_day = Counter()
    unique = defaultdict(set)
    bad_roles = Counter()
    role_records = Counter()
    for item in records(path, state):
        role = item['role']
        role_records[role] += 1
        if role not in ('L', 'U'):
            bad_roles[role] += 1
        for value in item['ids']:
            physical = tuple(value)
            if len(physical) != 5:
                raise ValueError('Five-field physical source ID required')
            packets[role] += 1
            rx_day[(role, physical[1], physical[2])] += 1
            unique[role].add(physical)
    state.update(role_records=dict(role_records), packets=dict(packets), unique_physical={k:len(v) for k,v in unique.items()},
                 RX_day=[dict(role=k[0], RX=k[1], day=k[2], count=n)
                         for k,n in sorted(rx_day.items())], unexpected_roles=dict(bad_roles))
    return state


def compare_sequences(left, right):
    sa, sb = {}, {}
    compared = 0
    first_difference = None
    lengths_differ = False
    sentinel = object()
    for a,b in itertools.zip_longest(records(left, sa), records(right, sb), fillvalue=sentinel):
        if a is sentinel or b is sentinel:
            lengths_differ = True
            continue
        compared += 1
        if a != b and first_difference is None:
            first_difference = compared
    return dict(compared_records=compared, first_differing_record=first_difference,
                record_counts_differ=lengths_differ,
                same_complete_sequence=first_difference is None and not lengths_differ
                and sa['exists'] and sb['exists'] and compared > 0
                and not sa['partial'] and not sb['partial'], left=sa, right=sb)


def stdout_scan(path):
    patterns = dict(traceback=r'Traceback \(most recent call last\)',
                    warning=r'\bwarning\b', OOM=r'out of memory|CUDA OOM',
                    killed=r'\bkilled\b', resume=r'\bresum(?:e|ing|ed)\b',
                    epoch_end=r'\[EPOCH-END\]', view_complete=r'VIEW_COMPLETE',
                    row_scored=r'ROW_SCORED', config=r'config|arguments|Namespace\(')
    compiled = {k:re.compile(v,re.I) for k,v in patterns.items()}
    counts = Counter()
    examples = defaultdict(list)
    lines = 0
    if path.is_file():
        with path.open(encoding='utf8', errors='replace') as stream:
            for lines,line in enumerate(stream,1):
                for key, pattern in compiled.items():
                    if pattern.search(line):
                        counts[key] += 1
                        if len(examples[key]) < 5:
                            examples[key].append(dict(line=lines,text=line.strip()[:240]))
    return dict(exists=path.is_file(), lines=lines, marker_counts=dict(counts),
                marker_examples=dict(examples))


def collect(root, closed=False):
    root = Path(root)
    result = dict(schema='F_FULLSOURCE_COMPLETE_LOG_COLLECTION_V1',
                  run_id=root.name, collection_mode='closed' if closed else 'live_source_only',
                  IQ_reads=0, checkpoint_reads=0, truth_reads=0, model_forwards=0,
                  source_RX_holdout=None, rows={})
    completion = read_json(root/'completion.json')
    if closed and (completion or {}).get('status') != 'ARTIFACTS_COMPLETE':
        raise ValueError('All fixed rows must finish before closed result collection')
    for arm in ARMS:
        rid = arm + '-s' + str(SEED)
        base = root/rid
        source = base/'source'
        row = dict(progress=read_json(source/'reference_progress.json'),
                   initialization=read_json(source/'initialization.json'),
                   native_args=read_json(source/'actual_native_args.json'),
                   resolved_config=read_json(source/'resolved_config.json'),
                   source_completion=read_json(source/'completion.json'),
                   training_result=read_json(source/'training_result.json'),
                   ordinary_actual=read_json(source/'ordinary_augmentation_actual.json'),
                   actual_channel=read_json(source/'actual_channel.json'),
                   stdout=stdout_scan(root/'logs'/(rid+'.stdout.log')))
        contract = read_json(source/'source_contract.json')
        row['source_role_counts'] = {k:len(v) for k,v in contract['role_ids'].items()} if contract else None
        epoch_state = {}
        row['epoch_metrics'] = list(records(source/'epoch_metrics.jsonl',epoch_state))
        row['epoch_file'] = epoch_state
        row['step_metrics'] = numeric_logs(source/'step_metrics.jsonl.gz','metrics')
        row['reference_steps'] = numeric_logs(source/'reference_steps.jsonl.gz')
        row['unit_steps'] = numeric_logs(source/'unit_steps.jsonl.gz')
        row['training_sequence'] = sequence(source/'training_sequence.jsonl.gz')
        if closed:
            for name,expected in (('step_metrics',4440),('reference_steps',4440),('unit_steps',4440)):
                item = row[name]
                if item['records'] != expected or item['partial']:
                    raise ValueError(arm + ': incomplete full log ' + name)
            if len(row['epoch_metrics']) != 20 or row['epoch_file']['partial']:
                raise ValueError(arm + ': incomplete epoch curve')
            row['prediction'] = read_json(base/'prediction/complete.json')
            row['score'] = read_json(base/'scores/metrics.json')
            row['row_completion'] = read_json(base/'completion.json')
            if (row['row_completion'] or {}).get('status') != 'TRAINED_DEFAULT_PREDICTED_SCORED':
                raise ValueError(arm + ': incomplete seven-view score')
        result['rows'][arm] = row
    result['sampling_comparisons'] = {}
    for a,b in (('CONCAT','REPEAT'),('REPEAT','NATIVE'),('CONCAT','NATIVE')):
        result['sampling_comparisons'][a+'-'+b] = compare_sequences(
            root/(a+'-s'+str(SEED))/'source/training_sequence.jsonl.gz',
            root/(b+'-s'+str(SEED))/'source/training_sequence.jsonl.gz')
    if closed:
        result['matrix_completion'] = completion
        result['result'] = read_json(root/'result.json')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('run_root')
    parser.add_argument('--closed', action='store_true')
    args = parser.parse_args(argv)
    json.dump(collect(Path(args.run_root),args.closed), sys.stdout,
              ensure_ascii=False,allow_nan=False)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
