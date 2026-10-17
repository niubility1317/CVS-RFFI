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

ARMS = ('B10', 'B01', 'B11')
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


"""All persisted logs and fixed paired scores;no new IQ/checkpoint/rawtruth/model access."""
from pathlib import Path
import json,math

def json_ready(value,path='',missing=None):
 if missing is None:missing=[]
 if isinstance(value,float) and not math.isfinite(value):missing.append(path);return None
 if isinstance(value,dict):return {k:json_ready(v,path+'/'+str(k),missing) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [json_ready(v,path+'/'+str(i),missing) for i,v in enumerate(value)]
 return value


def source_logs(source,stdout,expected=2220,require_detail=False):
 source=Path(source);state={};epochs=list(records(source/'epoch_metrics.jsonl',state));steps=numeric_logs(source/'step_metrics.jsonl.gz','metrics');seq=sequence(source/'training_sequence.jsonl.gz')
 if state['partial'] or [x['epoch'] for x in epochs]!=list(range(1,11)) or steps['records']!=expected or steps['partial'] or not steps['step_order']['consecutive'] or seq['partial'] or seq['unexpected_roles']:raise ValueError('Closed ten-epoch logs required')
 if seq['unique_physical']!={'L':6300,'U':56700} or seq['packets']!={'L':284160,'U':567000}:raise ValueError('Full original source coverage differs')
 for key,value in (('train/optimizer_step_applied',1),('train/pseudo_truth_available',0),('train/skipped_nonfinite_grad',0),('train/skipped_nonfinite_loss',0),('concat/L_forwards',4),('concat/As_packets',128),('concat/SupCon_weight',.1)):
  x=steps['totals'][key]
  if x['minimum']!=value or x['maximum']!=value:raise ValueError('Actual source update/U/view rule differs:'+key)
 detail={k:numeric_logs(source/(k+'.jsonl.gz')) for k in ('reference_steps','unit_steps')}
 if require_detail and any(x['records']!=expected or x['partial'] for x in detail.values()):raise ValueError('Missing candidate detailed source records')
 return dict(epoch_metrics=epochs,step_metrics=steps,training_sequence=seq,source_role_counts={k:len(v) for k,v in read_json(source/'source_contract.json')['role_ids'].items()},native_args=read_json(source/'actual_native_args.json'),initialization=read_json(source/'initialization.json'),EMA_initialization=read_json(source/'EMA_initialization.json'),optimizer_initialization=read_json(source/'optimizer_initialization.json'),source_completion=read_json(source/'completion.json'),training_result=read_json(source/'training_result.json'),progress=read_json(source/'reference_progress.json'),actual_channel=read_json(source/'actual_channel.json'),ordinary_actual=read_json(source/'ordinary_augmentation_actual.json'),native_terminal=read_json(source/'phase1_terminal_status.json'),stdout=stdout_scan(Path(stdout)),**detail)
