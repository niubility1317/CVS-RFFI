"""Read completed fixed predictions' scores; never select or retrain models."""
import csv
import statistics
from pathlib import Path
from experiments.standard_ce_baselines.common import PROJECT,RUN,SEEDS,EXTENSION_SEEDS,read,write


def aggregate(records):
    groups={}
    keys=set()
    for r in records:
        key=(r['variant'],r['model_seed'],r['dimension'],r['stratum'])
        if key in keys:raise ValueError('Duplicate seed/stratum')
        keys.add(key)
        groups.setdefault((r['variant'],r['dimension'],r['stratum']),[]).append(r)
    result=[]
    for (variant,dimension,stratum),rows in groups.items():
        if {r['model_seed'] for r in rows}!=set(SEEDS+EXTENSION_SEEDS):raise ValueError('Incomplete eight-seed group')
        if len({r['query_count'] for r in rows})!=1:raise ValueError('Different test sample counts')
        result.append(dict(variant=variant,dimension=dimension,stratum=stratum,actual_seed_count=8,
            query_count_per_seed=rows[0]['query_count'],
            **{metric+suffix:fn(r[metric] for r in rows) for metric in ('accuracy','macro_f1')
               for suffix,fn in (('_mean',statistics.mean),('_sample_sd',statistics.stdev))}))
    return result


def csv_write(path,rows):
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def combine(spec):
    root=Path(spec['runtime_root']);old=Path(PROJECT)/'runs'/RUN
    oldstate=read(old/'pipeline_state.json');newstate=read(root/'pipeline_state.json')
    variants={r['variant'] for r in spec['rows']};records=[]
    for folder,state,seeds in ((old,oldstate,SEEDS),(root,newstate,EXTENSION_SEEDS)):
        for variant in sorted(variants):
            for seed in seeds:
                rid=variant+'-s'+str(seed)
                if state['rows'].get(rid,{}).get('status')!='SCORED':raise ValueError('Unscored row: '+rid)
                path=folder/rid
                done=read(path/'source/completion.json')
                if done['epoch']!=200 or done['steps']!=10000:raise ValueError('Wrong budget')
                complete=read(path/'scores/complete.json')
                if complete['status']!='SCORED_COMPLETE':raise ValueError('Incomplete scoring')
                scores=read(path/'scores/scores.json')['results']
                if len(scores)!=14:raise ValueError('Missing overall/RX/TX result')
                for s in scores:records.append(dict(variant=variant,model_seed=seed,**s))
    groups=aggregate(records)
    overall={(r['variant'],r['model_seed']):r for r in records if r['dimension']=='overall'}
    paired=[];worst=[]
    for variant in sorted(variants):
        lows=[min(r['accuracy'] for r in records if r['variant']==variant and r['model_seed']==s and r['dimension']=='receiver') for s in SEEDS+EXTENSION_SEEDS]
        worst.append(dict(variant=variant,actual_seed_count=8,mean=statistics.mean(lows),sample_sd=statistics.stdev(lows)))
        for s in SEEDS+EXTENSION_SEEDS:
            paired.append(dict(variant=variant,control='residual_fusion',model_seed=s,
                accuracy_difference_pp=100*(overall[variant,s]['accuracy']-overall['residual_fusion',s]['accuracy']),
                macro_f1_difference_pp=100*(overall[variant,s]['macro_f1']-overall['residual_fusion',s]['macro_f1'])))
    # Config comparison proves that the extension changed only model seed and outputs.
    for row in spec['rows']:
        variant=row['variant'];a=read(old/(variant+'-s2026092701')/'source/resolved_config.json')
        b=read(root/row['row_id']/'source/resolved_config.json')
        for k in ('model_seed','loader_seed','output_root','device','pid','cwd','python','commit','hardware','torch_version'):
            a.pop(k,None);b.pop(k,None)
        if a!=b:raise ValueError('Training configuration changed: '+variant)
    write(root/'combined_eight_seed.json',dict(status='ANALYZED',network_count=13,seeds=list(SEEDS+EXTENSION_SEEDS),
        old_run=RUN,new_run=spec['run_id'],groups=groups,worst_receiver=worst,paired=paired,records=records,target_feedback_forbidden=True))
    csv_write(root/'combined_eight_seed.csv',groups)
    csv_write(root/'combined_per_seed.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in records])
    csv_write(root/'combined_paired.csv',paired)
    return groups
