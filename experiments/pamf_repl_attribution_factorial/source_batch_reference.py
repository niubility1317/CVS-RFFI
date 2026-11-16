"""Compare L-L, U-U and L-U source gradients at fixed E20 parameters."""
from itertools import combinations
import json
import sys
from experiments.cric_dura_comparison.io import read
from experiments.pamf_repl_weight import source_adamw_diagnostic as original
from experiments.pamf_ssl_research.source_branch_gradients import _relation

OFFSETS = tuple(range(12))
_snapshots = []


class ReferenceObjective(original.DiagnosticObjective):
    def guide(self, labeled, student):
        named = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
        with original.capture_guard_derivatives() as values:
            result = super().guide(labeled, student)
        if len(values) != 2:
            raise ValueError('Exactly the original complete L and raw U gradients are required')
        _snapshots.append(dict(named=[(n, p.detach()) for n, p in named],
            L=tuple(None if g is None else g.clone() for g in values[0]),
            U=tuple(None if g is None else g.clone() for g in values[1])))
        return result


def summarize(records):
    output = {}
    for setting in sorted({r['setting'] for r in records}):
        rows = [r for r in records if r['setting'] == setting]
        if len({r['offset'] for r in rows}) != len(rows) or len(rows) < 2:
            raise ValueError('Distinct source batches required')
        names = [(n, tuple(p.shape)) for n, p in rows[0]['named']]
        if any([(n, tuple(p.shape)) for n, p in r['named']] != names for r in rows):
            raise ValueError('Gradient parameter identities differ')
        groups = {}
        for group in original.GROUPS:
            def measure(a, b, left, right):
                return dict(left_offset=a['offset'], right_offset=b['offset'],
                    **_relation(a['named'], a[left], b[right], group))
            pairs = list(combinations(rows, 2))
            groups[group] = dict(
                L_vs_L=[measure(a,b,'L','L') for a,b in pairs],
                U_vs_U=[measure(a,b,'U','U') for a,b in pairs],
                L_vs_U_same_batch=[measure(r,r,'L','U') for r in rows],
                L_vs_U_cross_batch=[measure(a,b,'L','U') for a in rows for b in rows if a is not b])
        output[setting] = groups
    return output


def collect(runtime):
    if _snapshots:
        raise ValueError('A single source diagnostic execution per process is required')
    old = original.DiagnosticObjective, original.BATCH_OFFSETS
    original.DiagnosticObjective, original.BATCH_OFFSETS = ReferenceObjective, OFFSETS
    try:
        result = original.collect(runtime)
    finally:
        original.DiagnosticObjective, original.BATCH_OFFSETS = old
    if len(result['rows']) != len(_snapshots):
        raise ValueError('Every source batch must retain exactly one gradient pair')
    for record, row in zip(_snapshots, result['rows']):
        record.update(setting=row['setting'], offset=row['batch_offset'])
    result['batch_references'] = summarize(_snapshots)
    _snapshots.clear()
    result.update(status='SOURCE_BATCH_REFERENCE_COMPLETE', fixed_offsets=list(OFFSETS),
        interpretation='Within each setting the same source-only E20 parameters initialize every '
            'batch. Batch composition, random augmentation and forward buffers may differ. '
            'L-L, U-U and L-U relations are source gradient comparisons, not independent model '
            'seeds, real AdamW-history updates, or proof of target performance.')
    return result


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.pamf_repl_attribution_factorial.source_batch_reference RUNTIME_JSON')
    print(json.dumps(collect(read(sys.argv[1])), ensure_ascii=False, allow_nan=False), flush=True)
