"""Read a fixed complete prefix of a live, truth-blind source segment."""
import argparse
from collections import Counter,defaultdict
import gzip
from itertools import islice,zip_longest
import json
import math
from pathlib import Path

GROUPS=('t2','cls_head','other')
PARTS=('shallow','joint','extrap_delta')


def records(path,steps):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        yield from (json.loads(line) for line in islice(stream,steps))


def analyze(folder,*,offset,steps):
    folder=Path(folder)
    if offset!=30 or steps!=2220:raise ValueError('Only the fixed E31–E40 prefix is registered')
    ancestor=json.loads((folder.parent/'segment30'/'complete.json').read_text(encoding='utf-8'))
    if (ancestor['status']!='SOURCE_FROZEN' or ancestor['epoch']!=10 or ancestor['updates']!=2220
            or ancestor['target_access'] or ancestor['U_truth_read']):
        raise ValueError('Matching source-only E30 ancestor required')
    epoch_rows=[json.loads(x) for x in (folder/'metrics_epoch.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(epoch_rows)<10 or [r['epoch'] for r in epoch_rows[:10]]!=list(range(1,11)):
        raise ValueError('All first ten source validation epochs required')
    sums=defaultdict(Counter);groups={g:Counter() for g in GROUPS}
    parts={p:{g:Counter() for g in GROUPS} for p in PARTS};seen=accepted=n=samples=0
    unique_seen=set();unique_accepted=set()
    paths=(folder/'step_metrics.jsonl.gz',folder/'U_pseudo_observations.jsonl.gz')
    for n,(step,pseudo) in enumerate(zip_longest(*(records(p,steps) for p in paths)),1):
        if step is None or pseudo is None or step['step']!=n or pseudo['step']!=offset*222+n:
            raise ValueError('Exact aligned prefix steps and blind U records required')
        arrays=[pseudo[k] for k in ('physical_ids','pseudo','confidence','accepted')]
        if not all(len(a)==len(arrays[0]) for a in arrays):raise ValueError('Blind U fields do not align')
        seen+=len(arrays[0]);accepted+=sum(arrays[3]);unique_seen.update(arrays[0])
        unique_accepted.update(sid for sid,chosen in zip(arrays[0],arrays[3]) if chosen)
        m=step['metrics'];epoch=offset+(n-1)//222+1
        if int(m['HPL_seen'])!=seen or int(m['HPL_accepted'])!=accepted:
            raise ValueError('Blind U observations do not match online counters')
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
                    values={k:float(m[prefix+k]) for k in ('L_norm','U_norm','dot','cosine')}
                    if not all(math.isfinite(v) for v in values.values()):raise ValueError('Finite component sample required')
                    defined=values['L_norm']>0 and values['U_norm']>0
                    parts[part][group]['defined']+=int(defined)
                    parts[part][group]['conflict']+=int(defined and values['cosine'] < -1e-6)
                    if defined:parts[part][group]['cosine_sum']+=values['cosine']
                    parts[part][group]['U_norm_sum']+=values['U_norm']
    if n!=steps or samples!=22 or any(sums[e]['steps']!=222 for e in range(31,41)):
        raise ValueError('Complete fixed prefix and 22 component samples required')
    return dict(status='ANALYZED_LIVE_SOURCE_PREFIX_NOT_FROZEN',row_id=ancestor['config']['row_id'],
        first_epoch=31,last_epoch=40,steps=steps,U_seen=seen,U_accepted=accepted,U_acceptance=accepted/seen,
        unique_U_seen=len(unique_seen),unique_U_accepted=len(unique_accepted),component_samples=samples,
        epochs=[dict(epoch=offset+i+1,labeled_CE=sums[offset+i+1]['CE']/222,source_val_accuracy=row['val_tx_acc'])
                for i,row in enumerate(epoch_rows[:10])],groups={g:dict(v) for g,v in groups.items()},
        parts={p:{g:dict(v) for g,v in values.items()} for p,values in parts.items()},
        target_inputs_read=False,U_truth_read=False,
        interpretation='Live E31–E40 prefix only, not E50 or clean; repeated U exposures and 100-step component samples.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--offset',type=int,required=True)
    p.add_argument('--steps',type=int,required=True);a=p.parse_args()
    print(json.dumps(analyze(a.folder,offset=a.offset,steps=a.steps),allow_nan=False))
