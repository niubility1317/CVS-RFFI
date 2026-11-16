"""Analyze every closed training, step-guard and blind pseudo-decision record."""
from collections import Counter,defaultdict
import argparse
import gzip
from itertools import zip_longest
import json
import math
from pathlib import Path
from experiments.cric_dura_comparison.io import read


GROUPS=('t2','cls_head','other')
FIELDS=('train/loss_tx_labeled','HPL_loss','HPL_base_loss','HPL_confidence')+tuple(
    f'HPL_GP_{group}_{key}' for group in GROUPS for key in
    ('L_norm','U_norm','applied_U_norm','dot_before','dot_after','conflict','capped'))


def records(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def analyze(source):
    source=Path(source);done=read(source/'complete.json')
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100
            or done['target_access'] or done['U_truth_read']):
        raise ValueError('Frozen source-only complete trajectory required')
    epochs=[];hist=Counter();unique_seen=set();unique_accepted=set();seen=accepted=0
    total_groups={g:Counter() for g in GROUPS}
    for end,offset,steps in ((30,20,2220),(50,30,4440)):
        folder=source/f'segment{end}';state=read(folder/'repl_state.json')
        if state['calls']!=steps or state['U_truth_read'] or state['U_parameter_step_guard']['calls']!=steps:
            raise ValueError('Incomplete original U or actual step guard')
        epoch_rows=[json.loads(x) for x in (folder/'metrics_epoch.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        if len(epoch_rows)!=end-offset:raise ValueError('Incomplete source validation records')
        sums=defaultdict(Counter);counts=defaultdict(Counter);segment_seen=segment_accepted=0;n=0
        paths=[folder/'step_metrics.jsonl.gz',folder/'step_guard_metrics.jsonl.gz',folder/'U_pseudo_observations.jsonl.gz']
        for n,items in enumerate(zip_longest(*(records(path) for path in paths)),1):
            step,guard,pseudo=items
            if (None in items or step['step']!=n or guard['step']!=n or pseudo['step']!=offset*222+n):
                raise ValueError('Training, guard and U observation step alignment differs')
            arrays=[pseudo[k] for k in ('physical_ids','pseudo','confidence','accepted')]
            if not all(len(v)==len(arrays[0]) for v in arrays):raise ValueError('U batch fields differ')
            epoch=offset+(n-1)//222+1;m=step['metrics']
            segment_seen+=len(arrays[0]);segment_accepted+=sum(arrays[3])
            if int(m['HPL_seen'])!=segment_seen or int(m['HPL_accepted'])!=segment_accepted:
                raise ValueError('Online accepted exposure counter mismatch')
            for sid,label,selected in zip(arrays[0],arrays[1],arrays[3]):
                unique_seen.add(sid)
                if selected:unique_accepted.add(sid);hist[str(label)]+=1
            for key in FIELDS:
                value=float(m[key])
                if not math.isfinite(value):raise ValueError('Nonfinite logged source metric')
                sums[epoch][key]+=value;counts[epoch][key]+=1
            for group in GROUPS:
                value=guard['groups'][group]
                if not value['actual_parameter_check'] or not value['norm_nonexpansion']:
                    raise ValueError('Actual step projection constraint failed')
                before=value['anchor_cosines_before']['L_full'];after=value['anchor_cosines_after']['L_full']
                key='step_'+group+'_';sums[epoch][key+'retention']+=float(value['retention'])
                counts[epoch][key+'retention']+=1
                for name,condition in (('conflict_before',before is not None and before < -1e-6),
                                       ('conflict_after',after is not None and after < -1e-6),
                                       ('after_undefined',after is None),('fallback',value['quantization_fallback'])):
                    sums[epoch][key+name]+=int(condition);counts[epoch][key+name]+=1
                    total_groups[group][name]+=int(condition)
        if n!=steps or segment_seen!=state['seen'] or segment_accepted!=state['accepted']:
            raise ValueError('Complete source segment U exposure required')
        seen+=segment_seen;accepted+=segment_accepted
        for i,row in enumerate(epoch_rows,1):
            epoch=offset+i
            if row['epoch']!=i or counts[epoch]['train/loss_tx_labeled']!=222:
                raise ValueError('Complete per-epoch training records required')
            epochs.append(dict(epoch=epoch,source_val_accuracy=row['val_tx_acc'],
                means={k:sums[epoch][k]/v for k,v in counts[epoch].items()}))
    if sum(hist.values())!=accepted:raise ValueError('Accepted pseudo histogram mismatch')
    return dict(status='ANALYZED_COMPLETE_SOURCE_LOGS',row_id=done['config']['row_id'],steps=6660,
        U_seen=seen,U_accepted=accepted,U_acceptance=accepted/seen,accepted_pseudo_class_counts=dict(hist),
        unique_U_seen=len(unique_seen),unique_U_accepted=len(unique_accepted),groups={g:dict(v) for g,v in total_groups.items()},
        epochs=epochs,U_truth_read=False,target_inputs_read=False,
        interpretation='Repeated exposures differ from distinct U IDs; no U accuracy is inferred without independent truth scoring.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args()
    print(json.dumps(analyze(a.source),ensure_ascii=False,allow_nan=False))
