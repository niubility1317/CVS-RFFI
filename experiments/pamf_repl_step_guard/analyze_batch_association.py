"""Truth-blind batch associations from the complete prefix of three live logs."""
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import zlib


def available_records(path):
    """Read completed JSON lines from an open or closed single-member gzip."""
    data=zlib.decompressobj(31).decompress(Path(path).read_bytes())
    return [json.loads(line) for line in data.split(b'\n')[:-1] if line]


def analyze(segment,*,step_offset):
    segment=Path(segment)
    metrics,guards,pseudos=(available_records(segment/path) for path in
        ('step_metrics.jsonl.gz','step_guard_metrics.jsonl.gz','U_pseudo_observations.jsonl.gz'))
    n=min(map(len,(metrics,guards,pseudos)))
    if not n:raise ValueError('No aligned complete source steps')
    tuples=[];pseudo_hist={'conflict':Counter(),'nonconflict':Counter()};counts=Counter()
    for i,(metric,guard,pseudo) in enumerate(zip(metrics[:n],guards[:n],pseudos[:n]),1):
        if metric['step']!=i or guard['step']!=i or pseudo['step']!=step_offset+i:
            raise ValueError('Exact training and observation step alignment required')
        cosine=guard['groups']['t2']['anchor_cosines_before']['L_full']
        if cosine is None:status='undefined'
        else:status='conflict' if cosine < -1e-6 else 'nonconflict'
        counts[status]+=1
        pred,accept=pseudo['pseudo'],pseudo['accepted']
        if len(pred)!=len(accept):raise ValueError('Aligned blind pseudo decisions required')
        if status!='undefined':pseudo_hist[status].update(str(y) for y,keep in zip(pred,accept) if keep)
        tuples.append((float(metric['metrics']['train/loss_tx_labeled']),status))
    epochs=n//222; bins=[[0,0,0.] for _ in range(5)]
    for epoch in range(epochs):
        rows=sorted(tuples[epoch*222:(epoch+1)*222])
        for i in range(5):
            for ce,status in rows[i*222//5:(i+1)*222//5]:
                if status=='undefined':continue
                bins[i][0]+=1;bins[i][1]+=int(status=='conflict');bins[i][2]+=ce
    a,b=pseudo_hist['conflict'],pseudo_hist['nonconflict'];sa,sb=sum(a.values()),sum(b.values());js=None
    if sa and sb:
        js=0.
        for k in set(a)|set(b):
            x,y=a[k]/sa,b[k]/sb;mid=(x+y)/2
            js+=.5*((x*math.log2(x/mid) if x else 0)+(y*math.log2(y/mid) if y else 0))
    return dict(aligned_steps=n,full_epochs=epochs,step_offset=step_offset,conflict_steps=dict(counts),
        CE_quintiles=[dict(steps=count,conflicts=conflicts,mean_CE=total/count if count else None,
                           conflict_rate=conflicts/count if count else None) for count,conflicts,total in bins],
        accepted_pseudo_histograms={k:dict(v) for k,v in pseudo_hist.items()},pseudo_class_js_bits=js,
        U_truth_read=False,target_reads=False,
        interpretation='Training-batch associations on a live complete prefix; repeated U exposures are not independent.')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--segment',required=True);p.add_argument('--step-offset',type=int,required=True)
    args=p.parse_args();print(json.dumps(analyze(args.segment,step_offset=args.step_offset),allow_nan=False))
