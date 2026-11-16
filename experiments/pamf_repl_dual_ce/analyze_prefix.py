"""Read the fixed E31–E35 source prefix of a live dual-anchor run."""
import argparse
from collections import Counter,defaultdict
import gzip
from itertools import islice,zip_longest
import json
import math
from pathlib import Path

GROUPS=('t2','cls_head','other')


def records(path,steps):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        yield from (json.loads(line) for line in islice(stream,steps))


def analyze(folder,*,offset,steps):
    folder=Path(folder)
    if offset!=30 or steps!=1110:raise ValueError('Only the fixed E31–E35 prefix is registered')
    ancestor=json.loads((folder.parent/'segment30'/'complete.json').read_text(encoding='utf-8'))
    if (ancestor['status']!='SOURCE_FROZEN' or ancestor['epoch']!=10 or ancestor['updates']!=2220
            or ancestor['target_access'] or ancestor['U_truth_read']):
        raise ValueError('Matching source-only E30 ancestor required')
    epochs=[json.loads(x) for x in (folder/'metrics_epoch.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(epochs)<5 or [r['epoch'] for r in epochs[:5]]!=list(range(1,6)):
        raise ValueError('All first five source validation epochs required')
    sums=defaultdict(Counter);groups={g:Counter() for g in GROUPS};seen=accepted=n=0
    unique_seen=set();unique_accepted=set()
    paths=[folder/'step_metrics.jsonl.gz',folder/'step_guard_metrics.jsonl.gz',folder/'U_pseudo_observations.jsonl.gz']
    for n,(step,guard,pseudo) in enumerate(zip_longest(*(records(p,steps) for p in paths)),1):
        if (step is None or guard is None or pseudo is None or step['step']!=n or guard['step']!=n
                or pseudo['step']!=offset*222+n):
            raise ValueError('Training, actual parameter step and blind U prefixes must align')
        arrays=[pseudo[k] for k in ('physical_ids','pseudo','confidence','accepted')]
        if not all(len(a)==len(arrays[0]) for a in arrays):raise ValueError('Blind U fields do not align')
        seen+=len(arrays[0]);accepted+=sum(arrays[3]);unique_seen.update(arrays[0])
        unique_accepted.update(sid for sid,chosen in zip(arrays[0],arrays[3]) if chosen)
        m=step['metrics'];epoch=offset+(n-1)//222+1
        if int(m['HPL_seen'])!=seen or int(m['HPL_accepted'])!=accepted:
            raise ValueError('Blind U counters do not match observations')
        ce=float(m['train/loss_tx_labeled'])
        if not math.isfinite(ce):raise ValueError('Finite labeled CE required')
        sums[epoch]['CE']+=ce;sums[epoch]['steps']+=1
        for group in GROUPS:
            d=guard['groups'][group]
            if not d['actual_parameter_check'] or not d['norm_nonexpansion']:
                raise ValueError('Actual parameter-step checks required')
            groups[group]['fallback']+=int(d['quantization_fallback'])
            for anchor,label in (('L_full','full'),('L_clean','ce')):
                for stage in ('before','after'):
                    cosine=d['anchor_cosines_'+stage][anchor]
                    if cosine is not None and not math.isfinite(cosine):raise ValueError('Finite anchor cosine required')
                    conflict=cosine is not None and cosine < -1e-6
                    groups[group][f'{label}_{stage}_conflict']+=int(conflict)
                    groups[group][f'{label}_{stage}_undefined']+=int(cosine is None)
                    if stage=='after' and conflict:raise ValueError('Protected anchor conflict found')
            for key in ('L_norm','U_norm','applied_U_norm','conflict','capped'):
                value=float(m[f'HPL_GP_{group}_{key}'])
                if not math.isfinite(value):raise ValueError('Finite original gradient telemetry required')
                groups[group][key+'_sum']+=value
            for key in ('raw_increment_norm','applied_increment_norm','retention'):
                value=float(d[key])
                if not math.isfinite(value):raise ValueError('Finite actual update telemetry required')
                groups[group][key+'_sum']+=value
    if n!=steps or any(sums[e]['steps']!=222 for e in range(31,36)):
        raise ValueError('A complete E31–E35 aligned prefix is required')
    return dict(status='ANALYZED_LIVE_SOURCE_PREFIX_NOT_FROZEN',row_id=ancestor['config']['row_id'],
        first_epoch=31,last_epoch=35,steps=steps,U_seen=seen,U_accepted=accepted,U_acceptance=accepted/seen,
        unique_U_seen=len(unique_seen),unique_U_accepted=len(unique_accepted),
        epochs=[dict(epoch=offset+i+1,labeled_CE=sums[offset+i+1]['CE']/222,source_val_accuracy=row['val_tx_acc'])
                for i,row in enumerate(epochs[:5])],groups={g:dict(v) for g,v in groups.items()},
        U_truth_read=False,target_inputs_read=False,
        interpretation='Live E31–E35 prefix, not E50 or clean; CE is ordinary labeled CE, U counts are repeated exposures.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--offset',type=int,required=True)
    p.add_argument('--steps',type=int,required=True);a=p.parse_args()
    print(json.dumps(analyze(a.folder,offset=a.offset,steps=a.steps),allow_nan=False))
