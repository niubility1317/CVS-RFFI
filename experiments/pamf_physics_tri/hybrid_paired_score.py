"""Independent truth-last corrections/damage for frozen structural pairs."""
from pathlib import Path
import numpy as np
from experiments.cric_dura_comparison.io import read,write


def net_stats(y,base,other,rx,day):
    a=base==y;b=other==y;rows=[]
    groups=[('overall','ALL',np.ones(len(y),dtype=bool))]
    for dim,values in (('receiver',rx),('day',day),('transmitter',y)):
        groups.extend((dim,str(v),values==v) for v in np.unique(values))
    for dim,stratum,mask in groups:
        repairs=int((~a & b & mask).sum());damage=int((a & ~b & mask).sum())
        rows.append(dict(dimension=dim,stratum=stratum,count=int(mask.sum()),repairs=repairs,damage=damage,net=repairs-damage))
    return rows


def score(c):
    for folder in c['predictions'].values():
        done=read(Path(folder)/'complete.json')
        if done['status']!='ALL_PREDICTIONS_COMPLETE' or done['truth_read']:
            raise ValueError('All paired predictions must be frozen before truth access')
    rows=[]
    for task in ('cross_rx','cross_both'):
        arrays={}
        for arm,folder in c['predictions'].items():
            with np.load(Path(folder)/task/'predictions.npz',allow_pickle=False) as f:
                arrays[arm]={k:f[k].copy() for k in f.files}
        base_arm=c.get('base_arm','A');compare_arms=c.get('compare_arms',['B','C','D'])
        ids=arrays[base_arm]['ids']
        if any(not np.array_equal(x['ids'],ids) for x in arrays.values()):
            raise ValueError('Paired prediction physical IDs differ')
        truth=read(c['tests'][task]['truth'])
        y=np.asarray([truth[str(i)]['label'] for i in ids],dtype=np.int64)
        rx=np.asarray([str(truth[str(i)]['receiver']) for i in ids]);day=np.asarray([str(truth[str(i)]['day']) for i in ids])
        for route in c['readouts']:
            for view in c['views']:
                key=route+'__'+view;a=arrays[base_arm][key]
                for arm in compare_arms:
                    rows.extend(dict(task=task,readout=route,view=view,arm=arm,**v)
                                for v in net_stats(y,a,arrays[arm][key],rx,day))
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    write(out/'paired_net.json',dict(status='SCORED',truth_last=True,rows=rows))
    write(out/'complete.json',dict(status='PAIRED_SCORED_COMPLETE',truth_last=True,rows=len(rows)))
