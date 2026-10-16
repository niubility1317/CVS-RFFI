"""Independent post-process of all frozen SOURCE predictions; no model or target."""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from . import design as d


class Group:
    def __init__(self):
        self.cm=[[0]*6 for _ in range(6)];self.n=0;self.margin=0.;self.masks=defaultdict(lambda:[0,0]);self.repair=0;self.broken=0
    def add(self,r):
        y,p=r['TX'],r['prediction'];ok=p==y;self.n+=1;self.cm[y][p]+=1;self.margin+=r['margin']
        for key in ('clean_correct','C0_clean_correct','common_clean_correct'):
            if r.get(key):self.masks[key][0]+=1;self.masks[key][1]+=ok
        full=r.get('full_prediction',p)==y;self.repair+=ok and not full;self.broken+=full and not ok
    def result(self):
        tp=sum(self.cm[i][i] for i in range(6));f1=[]
        for k in range(6):
            den=sum(self.cm[k])+sum(row[k] for row in self.cm)
            f1.append(2*self.cm[k][k]/den if den else 0.)
        return dict(count=self.n,accuracy=tp/self.n,macro_f1=sum(f1)/6,confusion=self.cm,mean_cosine_margin=self.margin/self.n,
            retention={k:dict(denominator=v[0],correct=v[1],rate=v[1]/v[0] if v[0] else None) for k,v in self.masks.items()},
            repaired_vs_full=self.repair,broken_vs_full=self.broken)


def quality_bin(db):return '<5' if db<5 else '5..15' if db<15 else '>=15'
def cfo_bin(hz):return '<100' if abs(hz)<100 else '100..1000' if abs(hz)<1000 else '>=1000'


def harvest(base):
    base=Path(base);done=d.read(base/'completion.json')
    if done['status']!='SOURCE_F_CROSSRX_PRACTICAL_DIAGNOSTIC_COMPLETE' or done['target_access']:raise ValueError('Complete legal source diagnostic required')
    all_results={};predictions={};truth={};gradient_results={};channel_stats={}
    for row in d.rows():
        a=row['arm'];root=base/row['row_id'];groups=defaultdict(Group);features={};channels={};seen=set();predictions[a]={}
        with gzip.open(root/'packets.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);pid=r['physical_id'];key=(r['scene'],r['condition'],pid)
                if key in seen:raise ValueError('Duplicate physical observation in condition')
                seen.add(key);truth[pid]=r['TX'];predictions[a][key]=r['prediction']
                prefix=(r['scene'],r['condition']);groups[prefix+('all','all')].add(r)
                if r['scene']!='clean':
                    m=r['full_channel']
                    for dimension,value in [('state',m['state']),('lock',m['receiver_lock_status']),('quality',quality_bin(m['quality_snr_db'])),
                        ('CFO_Hz',cfo_bin(m['output_frequency_hz'])),('AGC_limited',str(m['agc_limited'])),('TX',str(r['TX'])),('day',str(r['day']))]:
                        groups[prefix+(dimension,value)].add(r)
                    for k,v in r['channel'].items():
                        if isinstance(v,(int,float)) and not isinstance(v,bool):
                            s=channels.setdefault(prefix+(k,),[0,0.,float('inf'),-float('inf')]);s[0]+=1;s[1]+=v;s[2]=min(s[2],v);s[3]=max(s[3],v)
                for name,stats in r['features'].items():
                    for metric,value in stats.items():
                        if isinstance(value,(int,float)):
                            s=features.setdefault(prefix+(name,metric),[0,0.]);s[0]+=1;s[1]+=value
        if len(seen)!=84420 or len({k[2] for k in seen})!=1260:raise ValueError('Complete predefined source physical matrix required')
        for scene in ('clean',*d.SCENES):
            for condition in (('clean',) if scene=='clean' else d.CONDITIONS):
                if groups[(scene,condition,'all','all')].n!=1260:raise ValueError('A predefined condition is incomplete')
        results=[dict(scene=k[0],condition=k[1],dimension=k[2],stratum=k[3],**g.result()) for k,g in sorted(groups.items())]
        all_results[a]=dict(completion=d.read(root/'completion.json'),metrics=results,
            feature_means=[dict(scene=k[0],condition=k[1],feature=k[2],metric=k[3],count=s[0],mean=s[1]/s[0]) for k,s in sorted(features.items())])
        channel_stats[a]=[dict(scene=k[0],condition=k[1],metric=k[2],count=s[0],mean=s[1]/s[0],minimum=s[2],maximum=s[3]) for k,s in sorted(channels.items())]
        gradients={};calls=0
        with gzip.open(root/'gradients.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);calls+=1
                if len(r['parameters'])!=14:raise ValueError('Actual original parameter paths changed')
                for p in r['parameters']:
                    for k in ('clean_norm','condition_norm','cosine'):
                        if p[k] is not None:
                            s=gradients.setdefault((r['scene'],r['condition'],p['parameter'],k),[0,0.]);s[0]+=1;s[1]+=p[k]
        if calls!=1340:raise ValueError('Complete fixed diagnostic derivative calls required')
        gradient_results[a]=[dict(scene=k[0],condition=k[1],parameter=k[2],metric=k[3],batches=s[0],mean=s[1]/s[0]) for k,s in sorted(gradients.items())]
    if predictions['C0'].keys()!=predictions['C1'].keys() or len(truth)!=1260:raise ValueError('Both fixed models must share physical matrix')
    paired=[]
    for scene in ('clean',*d.SCENES):
        for condition in (('clean',) if scene=='clean' else d.CONDITIONS):
            keys=[k for k in predictions['C0'] if k[:2]==(scene,condition)]
            repaired=sum(predictions['C0'][k]!=truth[k[2]] and predictions['C1'][k]==truth[k[2]] for k in keys)
            broken=sum(predictions['C0'][k]==truth[k[2]] and predictions['C1'][k]!=truth[k[2]] for k in keys)
            paired.append(dict(scene=scene,condition=condition,count=len(keys),C1_repaired_C0=repaired,C1_broke_C0=broken,
                C1_minus_C0_accuracy_pp=100*(repaired-broken)/len(keys)))
    result=dict(status='ANALYZED_SOURCE_ONLY',source_only=True,target_access=False,completion=done,rows=all_results,paired_model_differences=paired,
        channel_statistics=channel_stats,gradient_statistics=gradient_results,
        interpretation='Descriptive frozen source RX1 L1260/one model seed; counterfactual effects are not additive or training efficacy. C1 is paired practical CE versus C0 matched-forward ordinary IQ; source counterfactuals do not establish Ag efficacy or causal training attribution. No target/seed claim.')
    d.write(base/'analysis'/'summary.json',result)
    columns=['model','scene','condition','dimension','stratum','count','accuracy','macro_f1','mean_cosine_margin','repaired_vs_full','broken_vs_full']
    with (base/'analysis'/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for a,results in all_results.items():
            for r in results['metrics']:w.writerow(dict(model=a,**{k:r[k] for k in columns if k!='model'}))
    return dict(status=result['status'],rows=list(all_results),unique_source_physical_packets=1260,paired_full=[v for v in paired if v['condition'] in ('full','clean')])
