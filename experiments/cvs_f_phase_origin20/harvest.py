import csv,gzip,json
from collections import defaultdict
from pathlib import Path
from . import design as d
from experiments.cvs_f_practical_source20.harvest import Group,quality_bin

def harvest(base):
    base=Path(base);done=d.read(base/'completion.json')
    if done['status']!='SOURCE_F_PHASE_ORIGIN_COMPLETE' or done['target_access']:raise ValueError('Complete frozen source factor required')
    result={};predictions={};labels={};raws={};pairs={};changes={}
    for row in d.rows():
        arm=row['arm'];root=base/row['row_id'];groups=defaultdict(Group);seen=set();predictions[arm]={};raws[arm]={};paired=defaultdict(lambda:[0,0]);features={};cached_changes=0
        with gzip.open(root/'packets.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);pid=r['physical_id'];key=(r['scene'],r['condition'],pid)
                if key in seen:raise ValueError('Duplicate source physical factor observation')
                seen.add(key);labels[pid]=r['TX'];predictions[arm][key]=r['prediction'];prefix=key[:2]
                groups[prefix+('all','all')].add(r)
                if r['condition']=='raw':
                    raws[arm][pid]=r['prediction'];cached_changes+=r['prediction']!=r['original_cached_clean_prediction']
                else:
                    raw=r['raw_prediction']==r['TX'];ok=r['prediction']==r['TX'];paired[prefix][0]+=not raw and ok;paired[prefix][1]+=raw and not ok
                    full=r['full_channel']
                    for dimension,value in [('state',full['state']),('lock',full['receiver_lock_status']),('quality',quality_bin(full['quality_snr_db'])),('TX',str(r['TX'])),('day',str(r['day']))]:
                        groups[prefix+(dimension,value)].add(r)
                for name,stats in r['features'].items():
                    for metric,value in stats.items():
                        if isinstance(value,(int,float)):
                            z=features.setdefault(prefix+(name,metric),[0,0.]);z[0]+=1;z[1]+=value
        if len(seen)!=23940 or len(raws[arm])!=1260:raise ValueError('Complete fixed source matrix required')
        for scene in d.SCENES:
            for condition in d.CONDITIONS:
                if groups[(scene,condition,'all','all')].n!=1260:raise ValueError('Incomplete factor row')
        metrics=[dict(scene=k[0],condition=k[1],dimension=k[2],stratum=k[3],**g.result()) for k,g in sorted(groups.items())]
        pairs[arm]=[dict(scene=k[0],condition=k[1],repaired_raw=z[0],broken_raw=z[1],accuracy_change_pp=100*(z[0]-z[1])/1260) for k,z in sorted(paired.items())]
        result[arm]=dict(completion=d.read(root/'completion.json'),metrics=metrics,cached_original_clean_prediction_mismatches=cached_changes,
            feature_means=[dict(scene=k[0],condition=k[1],feature=k[2],metric=k[3],count=z[0],mean=z[1]/z[0]) for k,z in sorted(features.items())])
    if predictions['A0'].keys()!=predictions['W30'].keys() or len(labels)!=1260:raise ValueError('Both frozen models share source matrix')
    a0_correct={pid for pid,p in raws['A0'].items() if p==labels[pid]}
    for arm in result:
        effects=[]
        raw_acc=sum(p==labels[pid] for pid,p in raws[arm].items())/1260
        raw_ret=sum(raws[arm][pid]==labels[pid] for pid in a0_correct)/len(a0_correct)
        for scene in d.SCENES:
            scores={c:sum(predictions[arm][(scene,c,pid)]==y for pid,y in labels.items())/1260 for c in d.CONDITIONS}
            retention={c:sum(predictions[arm][(scene,c,pid)]==labels[pid] for pid in a0_correct)/len(a0_correct) for c in d.CONDITIONS}
            effects.append(dict(scene=scene,raw_accuracy=raw_acc,accuracies=scores,accuracy_effect_pp={c:100*(p-raw_acc) for c,p in scores.items()},
                accuracy_interaction_pp=100*(scores['phase_delay']-scores['phase']-scores['delay']+raw_acc),
                A0_fixed_raw_correct_count=len(a0_correct),raw_A0_retention=raw_ret,A0_retentions=retention,
                A0_retention_effect_pp={c:100*(p-raw_ret) for c,p in retention.items()},
                A0_retention_interaction_pp=100*(retention['phase_delay']-retention['phase']-retention['delay']+raw_ret)))
        changes[arm]=effects
    small=all(abs(v)<1. for x in changes['A0'] for name in ('accuracy_effect_pp','A0_retention_effect_pp') for v in x[name].values())
    summary=dict(status='ANALYZED_SOURCE_ONLY',source_only=True,target_access=False,completion=done,rows=result,paired_raw_changes=pairs,factor_effects=changes,
        predefined_interpretation=dict(A0_all_effects_abs_lt_1pp=small,abandon_major_factor_hypothesis=small,automatic_training=False,W30_promotion=False),
        interpretation='Frozen SOURCE1260/one seed only; delay is repeated same waveform across6scenes, not6replications. Synthetic factors are not practical replacements or training efficacy.')
    d.write(base/'analysis/summary.json',summary)
    columns=['model','scene','condition','dimension','stratum','count','accuracy','macro_f1','mean_cosine_margin']
    with (base/'analysis/metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for arm,z in result.items():
            for r in z['metrics']:w.writerow(dict(model=arm,**{k:r[k] for k in columns if k!='model'}))
    return dict(status=summary['status'],factor_effects=changes,predefined_interpretation=summary['predefined_interpretation'])
