"""Post-freeze U pseudo-label confusion, confidence and unique-ID diagnostics."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path

BINS=('0.95-0.97','0.97-0.99','0.99-0.999','0.999-1.0')


def confidence_bin(value):
    # Tensor-side threshold 0.95 is rounded to float32 before comparison.
    if not math.isfinite(value) or not .95-1e-7<=value<=1.:
        raise ValueError('Accepted confidence must be finite and at least 0.95')
    return BINS[0] if value<.97 else BINS[1] if value<.99 else BINS[2] if value<.999 else BINS[3]


def aggregate(records,*,truth,receivers,num_classes):
    if not truth or set(truth)!=set(receivers) or num_classes<2:
        raise ValueError('Exact U truth and source receiver maps required')
    confusion=[[0]*num_classes for _ in range(num_classes)]
    bins={key:Counter() for key in BINS};by_receiver_class={};individual={};seen=accepted=correct=steps=0
    for row in records:
        steps+=1
        arrays=[row[k] for k in ('physical_ids','pseudo','confidence','accepted')]
        if not arrays[0] or not all(len(a)==len(arrays[0]) for a in arrays):raise ValueError('Aligned U batch required')
        for sid,pseudo,confidence,selected in zip(*arrays):
            if (sid not in truth or not isinstance(pseudo,int) or not 0<=pseudo<num_classes
                    or not isinstance(selected,bool) or not isinstance(confidence,(int,float))
                    or not math.isfinite(confidence) or not 0<=confidence<=1):
                raise ValueError('Valid blind U observation required')
            actual=truth[sid]
            if not isinstance(actual,int) or not 0<=actual<num_classes:raise ValueError('Valid U truth required')
            seen+=1
            if not selected:continue
            bucket=confidence_bin(confidence);accepted+=1;right=int(pseudo==actual);correct+=right
            confusion[actual][pseudo]+=1;bins[bucket]['accepted']+=1;bins[bucket]['correct']+=right
            key=f'{receivers[sid]}:{actual}';by_receiver_class.setdefault(key,Counter()).update(accepted=1,correct=right)
            individual.setdefault(sid,Counter()).update(accepted=1,wrong=1-right)
    if not accepted:raise ValueError('At least one accepted U exposure required')
    wrong_ids=[v for v in individual.values() if v['wrong']]
    unique=dict(accepted=len(individual),ever_wrong=len(wrong_ids),
        always_wrong_when_accepted=sum(v['wrong']==v['accepted'] for v in wrong_ids),
        mixed_correct_and_wrong=sum(v['wrong']<v['accepted'] for v in wrong_ids),
        wrong_exposures=sum(v['wrong'] for v in wrong_ids),
        max_wrong_exposures_per_id=max((v['wrong'] for v in wrong_ids),default=0))
    return dict(steps=steps,overall=dict(seen=seen,accepted=accepted,accepted_correct=correct),
        confusion=confusion,confidence_bins={k:dict(v) for k,v in bins.items()},
        receiver_true_class={k:dict(v) for k,v in by_receiver_class.items()},unique=unique)


def validate_marginals(result,score,num_classes):
    matrix=result['confusion']
    for class_id in range(num_classes):
        true=Counter();pseudo=Counter()
        for segment in score['segments']:
            true.update(segment['per_true_class'].get(str(class_id),{}))
            pseudo.update(segment['per_pseudo_class'].get(str(class_id),{}))
        diagonal=matrix[class_id][class_id]
        if (sum(matrix[class_id])!=true['accepted'] or diagonal!=true['accepted_correct']
                or sum(row[class_id] for row in matrix)!=pseudo['accepted']
                or diagonal!=pseudo['accepted_correct']):
            raise ValueError('U confusion does not match independent true/pseudo class marginals')


def analyze(source):
    import numpy as np
    from experiments.pamf_repl_step_guard.u_scoring import load_truth
    root=Path(source).resolve()
    if root.name!='source' or root.parent.parent.name!='F':raise ValueError('Registered source trajectory required')
    done=json.loads((root/'complete.json').read_text(encoding='utf-8'));cfg=done['config'];rid=cfg['row_id']
    if (rid!=root.parent.name or done['status']!='SOURCE_FROZEN' or done['epoch']!=50
            or done['updates']!=11100 or done['target_access'] or done['U_truth_read']):
        raise ValueError('Complete source-only E50 freeze required')
    score_root=root.parents[2]/'readouts'/rid/'U_diagnostic'
    complete=json.loads((score_root/'complete.json').read_text(encoding='utf-8'))
    score=json.loads((score_root/'scores.json').read_text(encoding='utf-8'))
    if (complete['status']!='SCORED_U_DIAGNOSTIC' or complete['truth_last'] is not True
            or complete['source_epoch']!=50 or score['training_or_selection_use'] is not False):
        raise ValueError('Existing independent U diagnostic required')
    ids,truth=load_truth(cfg,source_completion=done)
    with np.load(Path(cfg['source_cache'])/'U_s_metadata.npz',allow_pickle=False) as meta:
        if 'y' in meta.files or [str(x) for x in meta['ids'].tolist()]!=ids:raise ValueError('Blind U metadata required')
        receivers=dict(zip(ids,[int(x) for x in meta['rx_i']]))
    def records():
        for end,offset,count in ((30,4440,2220),(50,6660,4440)):
            path=root/f'segment{end}'/'U_pseudo_observations.jsonl.gz';n=0
            with gzip.open(path,'rt',encoding='utf-8') as stream:
                for n,line in enumerate(stream,1):
                    value=json.loads(line)
                    if value['step']!=offset+n:raise ValueError('Complete ordered frozen U steps required')
                    yield value
            if n!=count:raise ValueError('Complete U trajectory required')
    result=aggregate(records(),truth=truth,receivers=receivers,num_classes=cfg['num_classes'])
    if result['overall']!=score['overall'] or result['unique']['accepted']!=score['unique_accepted']:
        raise ValueError('Independent U score and unique acceptance must match')
    validate_marginals(result,score,cfg['num_classes'])
    result.update(status='ANALYZED_FROZEN_U_CONFUSION',row_id=rid,truth_last=True,training_or_selection_use=False,
        interpretation='Repeated exposures and unique samples are distinct. Always-wrong means only '
        'among accepted appearances. Source U diagnostic only; no training, target or selection use.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args()
    print(json.dumps(analyze(a.source),ensure_ascii=False,allow_nan=False))
