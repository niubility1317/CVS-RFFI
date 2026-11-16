"""Summarize existing post-freeze U score by true/pseudo class and epoch."""
import argparse
from collections import Counter
import json
from pathlib import Path

KEYS=('seen','accepted','accepted_correct')


def summarize(score,*,num_classes):
    if score['status']!='SCORED_U_DIAGNOSTIC' or score['training_or_selection_use'] is not False:
        raise ValueError('Existing independent diagnostic-only U score required')
    if len(score['segments'])!=2 or num_classes<2:raise ValueError('Both original source segments required')
    aggregates={key:{} for key in ('per_epoch','per_true_class','per_pseudo_class')}
    for segment in score['segments']:
        if (segment['role']!='U_s' or segment['truth_used_only_for_offline_diagnostic'] is not True
                or segment['training_or_selection_use'] is not False):
            raise ValueError('Source U diagnostic-only segments required')
        for dimension,values in aggregates.items():
            for key,counts in segment[dimension].items():
                if any(not isinstance(counts[k],int) or counts[k]<0 for k in KEYS):
                    raise ValueError('Nonnegative integer exposure counts required')
                if not counts['accepted_correct']<=counts['accepted']<=counts['seen']:
                    raise ValueError('Invalid accepted/correct counts')
                values.setdefault(key,Counter()).update({k:counts[k] for k in KEYS})
    if set(aggregates['per_epoch'])!={str(i) for i in range(21,51)}:
        raise ValueError('Complete epochs 21 through 50 required')
    for dim in ('per_true_class','per_pseudo_class'):
        if set(aggregates[dim])!={str(i) for i in range(num_classes)}:
            raise ValueError('Every registered class required')
    overall=score['overall'];wrong=overall['accepted']-overall['accepted_correct']
    output={}
    for dim,values in aggregates.items():
        sums=Counter()
        for counts in values.values():sums.update(counts)
        if any(sums[k]!=overall[k] for k in KEYS):raise ValueError('Aggregated U counts must close')
        output[dim]={}
        for key,counts in sorted(values.items(),key=lambda x:int(x[0])):
            errors=counts['accepted']-counts['accepted_correct']
            output[dim][key]=dict(**{k:counts[k] for k in KEYS},accepted_wrong=errors,
                acceptance=counts['accepted']/counts['seen'] if counts['seen'] else None,
                accuracy=counts['accepted_correct']/counts['accepted'] if counts['accepted'] else None,
                wrong_exposure_share=errors/wrong if wrong else None)
    return dict(status='ANALYZED_U_CLASS_EXPOSURES',overall=overall,dimensions=output,
        unique_U_total=score['unique_U_total'],unique_accepted=score['unique_accepted'],
        training_or_selection_use=False,
        interpretation='Repeated exposures, not unique errors. True classes and pseudo classes are separate '
        'marginals, not a confusion matrix. Frozen U truth diagnostic only; no training or selection use.')


def analyze(source):
    root=Path(source);done=json.loads((root/'complete.json').read_text(encoding='utf-8'))
    if (root.name!='source' or root.parent.parent.name!='F' or done['status']!='SOURCE_FROZEN'
            or done['epoch']!=50 or done['updates']!=11100 or done['target_access'] or done['U_truth_read']):
        raise ValueError('Registered source-only E50 freeze required')
    cfg=done['config'];rid=cfg['row_id']
    if rid!=root.parent.name:raise ValueError('Exact source row required')
    folder=root.parents[2]/'readouts'/rid/'U_diagnostic'
    status=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    if (status['status']!='SCORED_U_DIAGNOSTIC' or status['truth_last'] is not True
            or status['source_epoch']!=50 or status['training_or_selection_use'] is not False):
        raise ValueError('Post-freeze diagnostic-only U scorer required')
    result=summarize(json.loads((folder/'scores.json').read_text(encoding='utf-8')),num_classes=cfg['num_classes'])
    result['row_id']=rid
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args()
    print(json.dumps(analyze(a.source),ensure_ascii=False,allow_nan=False))
