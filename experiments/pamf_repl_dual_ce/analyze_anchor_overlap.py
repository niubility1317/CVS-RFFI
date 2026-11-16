"""Classify full-L and ordinary-CE conflicts before the actual step guard."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path

GROUPS=('t2','cls_head','other')
CATEGORIES=('both','full_only','ce_only','neither','undefined')


def category(full,ce):
    if full is None or ce is None:return 'undefined'
    if not all(math.isfinite(v) for v in (full,ce)):raise ValueError('Finite anchor directions required')
    left,right=full < -1e-6,ce < -1e-6
    return 'both' if left and right else 'full_only' if left else 'ce_only' if right else 'neither'


def analyze(source):
    source=Path(source);done=json.loads((source/'complete.json').read_text(encoding='utf-8'))
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100
            or done['target_access'] or done['U_truth_read']):
        raise ValueError('Complete source-only E50 trajectory required')
    output={g:{c:Counter() for c in CATEGORIES} for g in GROUPS}
    for end,expected in ((30,2220),(50,4440)):
        folder=source/f'segment{end}';seg=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        if (seg['status']!='SOURCE_FROZEN' or seg['updates']!=expected or seg['target_access']
                or seg['U_truth_read']):raise ValueError('Complete source segment required')
        n=0
        with gzip.open(folder/'step_guard_metrics.jsonl.gz','rt',encoding='utf-8') as stream:
            for n,line in enumerate(stream,1):
                row=json.loads(line)
                if row['step']!=n:raise ValueError('Ordered step records required')
                for group in GROUPS:
                    d=row['groups'][group];before=d['anchor_cosines_before'];after=d['anchor_cosines_after']
                    if not d['actual_parameter_check'] or not d['norm_nonexpansion']:
                        raise ValueError('Actual step constraints required')
                    if any(v is not None and (not math.isfinite(v) or v < -1e-6)
                           for v in (after['L_full'],after['L_clean'])):
                        raise ValueError('Protected anchor conflict')
                    raw=float(d['raw_increment_norm']);applied=float(d['applied_increment_norm'])
                    if not all(math.isfinite(v) and v>=0 for v in (raw,applied)):
                        raise ValueError('Finite actual increment norms required')
                    c=output[group][category(before['L_full'],before['L_clean'])]
                    c['steps']+=1;c['zero_extra']+=int(applied==0);c['quantization_fallback']+=int(d['quantization_fallback'])
                    c['raw_norm_sum']+=raw;c['applied_norm_sum']+=applied;c['retention_sum']+=float(d['retention'])
        if n!=expected:raise ValueError('Complete segment steps required')
    return dict(status='ANALYZED_DUAL_ANCHOR_OVERLAP',row_id=done['config']['row_id'],steps=6660,
        groups={g:{c:dict(values) for c,values in rows.items()} for g,rows in output.items()},
        target_inputs_read=False,U_truth_read=False,
        interpretation='Batch-level, actual incremental AdamW descent relative to same-state L-only; '
        'zero or undefined directions are separate; not a causal counterfactual or sample label.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args()
    print(json.dumps(analyze(a.source),ensure_ascii=False,allow_nan=False))
