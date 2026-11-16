"""Complete blind-U source segment and sampled loss-component gradient analysis."""
import argparse
from collections import Counter,defaultdict
import gzip
from itertools import zip_longest
import json
import math
from pathlib import Path

GROUPS=('t2','cls_head','other')
PARTS=('shallow','joint','extrap_delta')


def component_stat(value):
    defined=value['L_norm']>0 and value['U_norm']>0
    cosine=value['cosine'] if defined else 0.
    return dict(defined=int(defined),conflict=int(defined and cosine < -1e-6),cosine_sum=cosine)


def records(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def analyze(folder,*,offset,steps):
    folder=Path(folder)
    done=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    state=json.loads((folder/'repl_state.json').read_text(encoding='utf-8'))
    if (done['status']!='SOURCE_FROZEN' or done['updates']!=steps or done['target_access']
            or done['U_truth_read'] or state['U_truth_read'] or state['calls']!=steps):
        raise ValueError('A complete source-only frozen segment is required')
    epochs=[json.loads(x) for x in (folder/'metrics_epoch.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(epochs)*222!=steps:raise ValueError('Every source validation epoch required')
    sums=defaultdict(Counter);parts={part:{g:Counter() for g in GROUPS} for part in PARTS}
    groups={g:Counter() for g in GROUPS};unique_seen=set();unique_accepted=set();seen=accepted=n=samples=0
    paths=(folder/'step_metrics.jsonl.gz',folder/'U_pseudo_observations.jsonl.gz')
    for n,(step,pseudo) in enumerate(zip_longest(*(records(p) for p in paths)),1):
        if step is None or pseudo is None or step['step']!=n or pseudo['step']!=offset*222+n:
            raise ValueError('Training steps and blind U observations must align')
        arrays=[pseudo[k] for k in ('physical_ids','pseudo','confidence','accepted')]
        if not all(len(a)==len(arrays[0]) for a in arrays):raise ValueError('Blind U fields do not align')
        seen+=len(arrays[0]);accepted+=sum(arrays[3]);unique_seen.update(arrays[0])
        unique_accepted.update(sid for sid,chosen in zip(arrays[0],arrays[3]) if chosen)
        m=step['metrics'];epoch=offset+(n-1)//222+1
        if int(m['HPL_seen'])!=seen or int(m['HPL_accepted'])!=accepted:
            raise ValueError('Online U counters do not match observations')
        ce=float(m['train/loss_tx_labeled'])
        if not math.isfinite(ce):raise ValueError('Finite labeled CE required')
        sums[epoch]['CE']+=ce;sums[epoch]['steps']+=1
        for group in GROUPS:
            for key in ('L_norm','U_norm','applied_U_norm','conflict','capped'):
                value=float(m[f'HPL_GP_{group}_{key}'])
                if not math.isfinite(value):raise ValueError('Finite group telemetry required')
                groups[group][key]+=value
        if m.get('HPL_component_sample')==1:
            samples+=1
            for part in PARTS:
                for group in GROUPS:
                    prefix=f'HPL_component_{part}_{group}_'
                    value={k:float(m[prefix+k]) for k in ('L_norm','U_norm','dot','cosine')}
                    if not all(math.isfinite(v) for v in value.values()):
                        raise ValueError('Finite sampled component telemetry required')
                    parts[part][group].update(component_stat(value))
                    parts[part][group]['U_norm_sum']+=value['U_norm']
    if n!=steps or seen!=state['seen'] or accepted!=state['accepted'] or samples!=steps//100:
        raise ValueError('Complete source steps, U observations and component samples required')
    return dict(status='ANALYZED_CLOSED_SOURCE_SEGMENT',row_id=done['config']['row_id'],offset=offset,steps=steps,
        U_seen=seen,U_accepted=accepted,U_acceptance=accepted/seen,unique_U_seen=len(unique_seen),
        unique_U_accepted=len(unique_accepted),component_samples=samples,
        epochs=[dict(epoch=offset+i+1,labeled_CE=sums[offset+i+1]['CE']/222,source_val_accuracy=row['val_tx_acc'])
                for i,row in enumerate(epochs)],
        groups={g:dict(v) for g,v in groups.items()},parts={p:{g:dict(v) for g,v in values.items()}
                                                          for p,values in parts.items()},
        U_truth_read=False,target_inputs_read=False,
        interpretation='Repeated exposures, 100-step component samples, and zero-norm directions marked undefined.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--offset',type=int,required=True)
    p.add_argument('--steps',type=int,required=True);a=p.parse_args()
    print(json.dumps(analyze(a.folder,offset=a.offset,steps=a.steps),allow_nan=False))
