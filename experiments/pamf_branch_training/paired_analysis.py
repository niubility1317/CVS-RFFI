"""Read-only, same-seed comparison of complete truth-last score artifacts."""
import argparse
import csv
import json
from pathlib import Path

from .plan import VIEWS, WEAK_VIEWS


def merge_disjoint(first, second):
    """Combine confusion matrices before deriving micro accuracy and macro F1."""
    a, b = first['confusion'], second['confusion']
    n = len(a)
    if not n or len(b) != n or any(len(r) != n for r in a + b):
        raise ValueError('Square matched confusion matrices required')
    cm = [[int(a[i][j]) + int(b[i][j]) for j in range(n)] for i in range(n)]
    count = sum(map(sum, cm))
    if count != first['count'] + second['count'] or not count:
        raise ValueError('Confusion counts do not match disjoint populations')
    accuracy = sum(cm[i][i] for i in range(n)) / count
    f1 = []
    for i in range(n):
        denom = sum(cm[i]) + sum(cm[j][i] for j in range(n))
        f1.append(2 * cm[i][i] / denom if denom else 0.)
    return dict(count=count, accuracy=accuracy, macro_f1=sum(f1) / n, confusion=cm)


def cells(data, *, weak=False):
    if data.get('status') != 'SCORED' or data.get('truth_last') is not True:
        raise ValueError('Complete independent truth-last scoring required')
    out = {}
    allowed = WEAK_VIEWS if weak else VIEWS
    for r in data['results']:
        if r['dimension'] != 'overall' or r['view'] not in allowed:
            continue
        key = (r['test'], r['readout'], r['view'])
        if key in out:
            raise ValueError('Duplicate scored cell')
        out[key] = r
    if not out:
        raise ValueError('No comparable overall scores')
    if not weak:
        # The F0 unseen-RX target is the disjoint union of these two populations.
        for readout in {k[1] for k in out}:
            for view in allowed:
                a, b = ('cross_rx', readout, view), ('cross_both', readout, view)
                if a in out and b in out:
                    out['combined_target', readout, view] = merge_disjoint(out[a], out[b])
    return out


def compare(candidate, baseline, candidate_name, baseline_name, family):
    common = sorted(candidate.keys() & baseline.keys())
    rows = []
    for test, readout, view in common:
        a, b = candidate[test, readout, view], baseline[test, readout, view]
        if a['count'] != b['count']:
            raise ValueError('Matched target cell count differs')
        rows.append(dict(candidate=candidate_name, baseline=baseline_name, family=family, test=test,
                         readout=readout, view=view, count=a['count'],
                         candidate_accuracy=100 * a['accuracy'], baseline_accuracy=100 * b['accuracy'],
                         delta_accuracy_pp=100 * (a['accuracy'] - b['accuracy']),
                         candidate_macro_f1=100 * a['macro_f1'], baseline_macro_f1=100 * b['macro_f1'],
                         delta_macro_f1_pp=100 * (a['macro_f1'] - b['macro_f1'])))
    if not rows:
        raise ValueError('No matched score cells')
    if family == 'practical':
        for test, readout in sorted({(r['test'], r['readout']) for r in rows}):
            selected = [r for r in rows if r['test'] == test and r['readout'] == readout and r['view'] in VIEWS[1:]]
            if len(selected) != 6:
                continue
            mean = {key: sum(r[key] for r in selected) / 6 for key in
                    ('candidate_accuracy', 'baseline_accuracy', 'delta_accuracy_pp', 'candidate_macro_f1',
                     'baseline_macro_f1', 'delta_macro_f1_pp')}
            rows.append(dict(candidate=candidate_name, baseline=baseline_name, family=family, test=test,
                             readout=readout, view='six_practical_mean', count=selected[0]['count'], **mean))
    return rows


def run(manifest, output):
    spec = json.loads(Path(manifest).read_text(encoding='utf-8-sig'))
    scored = {}
    for name, paths in spec.items():
        scored[name] = {family: cells(json.loads(Path(path).read_text(encoding='utf-8-sig')), weak=family == 'weak')
                        for family, path in paths.items()}
    result = []
    for name in ('sup_l', 'pl_l', 'pl_u', 'pl_lu'):
        for baseline in (('baseline_sup',) if name == 'sup_l' else
                         ('baseline_pl', 'natural_repl', 'shuffled_ssl')):
            for family in sorted(scored[name].keys() & scored[baseline].keys()):
                result.extend(compare(scored[name][family], scored[baseline][family], name, baseline, family))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    return len(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print('matched_cells', run(args.manifest, args.output))
