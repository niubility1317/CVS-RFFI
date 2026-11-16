"""Post-freeze diagnostic: relate each blind-U batch's precision to step conflicts.

The caller supplies truth in the independent scoring process.  No labels or
diagnostic statistics are returned to a trainer or used for selection.
"""
import gzip
import json


def analyze(observations,guard_metrics,*,truth,step_offset,source_completion):
    if (source_completion.get('status')!='SOURCE_FROZEN' or source_completion.get('epoch')!=50
            or source_completion.get('target_access') is not False or source_completion.get('U_truth_read') is not False):
        raise ValueError('A completed source-only E50 must be frozen before U truth diagnosis')
    with gzip.open(observations,'rt',encoding='utf-8') as source:
        predictions=[json.loads(line) for line in source]
    with gzip.open(guard_metrics,'rt',encoding='utf-8') as source:
        guards=[json.loads(line) for line in source]
    if not predictions or len(predictions)!=len(guards):
        raise ValueError('Complete fixed paired observations and guard steps required')
    names=('t2','cls_head','other')
    result={g:{'conflict':dict(steps=0,seen=0,accepted=0,accepted_correct=0),
               'nonconflict':dict(steps=0,seen=0,accepted=0,accepted_correct=0),
               'undefined':dict(steps=0,seen=0,accepted=0,accepted_correct=0)} for g in names}
    for index,(pseudo,guard) in enumerate(zip(predictions,guards),1):
        if pseudo['step']!=step_offset+index or guard['step']!=index:
            raise ValueError('Exact same-step pairing required')
        arrays=[pseudo[k] for k in ('physical_ids','pseudo','accepted')]
        if not arrays[0] or not all(len(x)==len(arrays[0]) for x in arrays):
            raise ValueError('Aligned pseudo decisions required')
        accepted=correct=0
        for sid,label,selected in zip(*arrays):
            if sid not in truth or not isinstance(selected,bool):raise ValueError('Unknown U ID or acceptance')
            if selected:
                accepted+=1;correct+=int(label==truth[sid])
        for group in names:
            cosine=guard['groups'][group]['anchor_cosines_before']['L_full']
            status='undefined' if cosine is None else 'conflict' if cosine < -1e-6 else 'nonconflict'
            bucket=result[group][status]
            bucket['steps']+=1;bucket['seen']+=len(arrays[0]);bucket['accepted']+=accepted
            bucket['accepted_correct']+=correct
    return dict(steps=len(predictions),groups=result,truth_used_only_after_freeze=True,
                interpretation='Descriptive association across repeated source exposures; '
                'conflict refers to extra AdamW motion before parameter-step guard, not a causal effect.')


def analyze_frozen_run(source_root,output):
    from pathlib import Path
    from experiments.cric_dura_comparison.io import read,write
    from .u_scoring import load_truth
    root=Path(source_root);done=read(root/'complete.json')
    _,truth=load_truth(done['config'],source_completion=done)
    results={}
    for epoch,offset in ((30,4440),(50,6660)):
        segment=root/f'segment{epoch}'
        results[str(epoch)]=analyze(segment/'U_pseudo_observations.jsonl.gz',
            segment/'step_guard_metrics.jsonl.gz',truth=truth,step_offset=offset,source_completion=done)
    write(output,dict(status='ANALYZED_U_CONFLICT_DIAGNOSTIC',segments=results,
                      training_or_selection_use=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();analyze_frozen_run(args.source,args.output)
