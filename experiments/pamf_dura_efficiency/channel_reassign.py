"""Reassign source-only linear channel/noise tuples, keeping clean IQ unchanged."""
import numpy as np
from experiments.pamf_restoration.models import complex_iq,pair


def matched_donors(positions,rx,day,labels,rng):
    keys=list(zip(*(labels[k] for k in ('scene','state','lock'))))
    groups={}
    for index,key in enumerate(keys):groups.setdefault(key,[]).append(index)
    groups={key:np.asarray(value,np.int64) for key,value in groups.items()}
    kept=[];within=[];cross=[];excluded=[]
    for index in positions:
        pool=groups[keys[index]]
        same=pool[(rx[pool]==rx[index])&(day[pool]==day[index])&(pool!=index)]
        other=pool[rx[pool]!=rx[index]]
        if not len(same) or not len(other):excluded.append(index);continue
        kept.append(index);within.append(rng.choice(same));cross.append(rng.choice(other))
    return {name:np.asarray(value,np.int64) for name,value in dict(positions=kept,within=within,cross=cross,excluded=excluded).items()}


def reassign(clean_destination,clean_donor,received_donor,action_donor):
    noise=complex_iq(received_donor)-action_donor(complex_iq(clean_donor))
    signal=action_donor(complex_iq(clean_destination))
    return pair(signal+noise),pair(noise),pair(signal)


def qualifies(rows):
    return len(rows)==2 and all(row['count']>=480 and row['IQ_ratio']>=1.10 and row['action_ratio']>=1.10
        and row['reference_unchanged'] and row['noise_error_max']<=1e-10 for row in rows)
