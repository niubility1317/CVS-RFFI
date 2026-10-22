"""Truth-last score-only matched F-PL x G-data-permission contrasts."""
from collections import defaultdict
import csv
from pathlib import Path
import statistics
from .io import read,write
from .analyze_core import verified_metrics,key
from .external_dura import VIEWS
from .gfactor_scope import SEEDS,PROTOCOL_IDS

NAMES=('Sup+G-L','Sup+G-LU','PL+G-L','PL+G-LU')


def run(c):
    expected={(p,s) for p in PROTOCOL_IDS for s in SEEDS};cells=c['cells']
    if len(cells)!=51 or {(v['protocol_id'],v['model_seed']) for v in cells}!=expected:
        raise ValueError('Incomplete17-contract/three-seed permission factorial')
    cache={};values={};pairs=[];checked=0;costs=[];G_cache_cost={}
    for cell in cells:
        pid=cell['protocol_id'];seed=cell['model_seed'];raw_by_F={};baseline_keys=None
        if set(cell['records'])!=set(NAMES):raise ValueError('Incomplete Sup/PL x G-L/LU cell')
        for name,r in cell['records'].items():
            path=r['scores']
            if path not in cache:
                if read(r['completion'])['status']!='SCORED_COMPLETE':raise ValueError('Unfinished independent score')
                obj=read(path)
                if obj.get('status')!='SCORED' or obj.get('truth_last') is not True or obj.get('target_feedback_forbidden') is not True:
                    raise ValueError('Only independently frozen truth-last scores are allowed')
                cache[path]=obj
            obj=cache[path]
            if (obj['run_id'],obj['row_id'],obj['model_seed'])!=(r['run_id'],r['row_id'],seed):raise ValueError('Wrong source F/seed')
            raw={key(x):x for x in obj['results'] if x['readout']=='raw'}
            selected={key(x):x for x in obj['results'] if x['readout']==r['readout']}
            if len(raw)!=sum(x['readout']=='raw' for x in obj['results']) or len(selected)!=sum(x['readout']==r['readout'] for x in obj['results']):
                raise ValueError('Duplicate score strata')
            if not raw or set(raw)!=set(selected):raise ValueError('Missing raw/G strata')
            tasks={'main','source_unseen_date'} if pid=='chrono' else {'main'}
            if {(k[0],k[1]) for k in selected}!={(t,v) for t in tasks for v in VIEWS}:raise ValueError('Seven-view/task coverage differs')
            if baseline_keys is None:baseline_keys=set(selected)
            elif set(selected)!=baseline_keys:raise ValueError('Factorial target strata differ')
            family=name.split('+')[0];cm={k:x['confusion'] for k,x in raw.items()}
            if family in raw_by_F and raw_by_F[family]!=cm:raise ValueError('Same-F raw differs across restoration roles')
            raw_by_F[family]=cm
            pp={key(x):x for x in obj['paired_raw'] if x['readout']==r['readout']}
            for k,x in selected.items():
                m=verified_metrics(x,len(cell['classes']));verified_metrics(raw[k],len(cell['classes']));checked+=1
                if x['count']!=raw[k]['count']:raise ValueError('Raw/G counts differ')
                values[(pid,seed,name,*k)]=m
                p=pp[k];net=sum(x['confusion'][i][i]-raw[k]['confusion'][i][i] for i in range(len(cell['classes'])))
                if p['count']!=x['count'] or p['net']!=net or p['repairs']-p['damage']!=net:raise ValueError('Invalid G net corrections')
                pairs.append(dict(protocol_id=pid,seed=seed,cell=name,test=k[0],view=k[1],dimension=k[2],stratum=k[3],
                    count=p['count'],repairs=p['repairs'],damage=p['damage'],net=p['net']))
        counts={k:sum(map(sum,cm)) for k,cm in raw_by_F['Sup'].items()}
        if counts!={k:sum(map(sum,cm)) for k,cm in raw_by_F['PL'].items()}:raise ValueError('Sup/PL sample counts differ')
        f=read(cell['source_completion'])
        costs.append(dict(protocol_id=pid,model_seed=seed,PL_F_shared_worker_seconds=f.get('seconds'),
            new_F_training_seconds=0,new_G_training_seconds=0,source_completion=cell['source_completion']))
        if pid not in G_cache_cost:
            cc=read(cell['cache_config']);done=read(Path(cc['output_root'])/'complete.json')
            if cc['readout']!='DURA-L' or done['status']!='DURA_IQ_CACHE_COMPLETE':raise ValueError('Wrong shared G-L cache')
            G_cache_cost[pid]=dict(shared_cache_seconds=done['seconds'],checkpoint=cc['G']['checkpoint'])
    for pid,seed in expected:
        for name in NAMES:
            strata={k[3:] for k in values if k[:3]==(pid,seed,name) and k[4]=='clean'}
            for test,_,dimension,stratum in strata:
                rows=[values[pid,seed,name,test,v,dimension,stratum] for v in VIEWS[1:]]
                values[pid,seed,name,test,'six_practical_mean',dimension,stratum]={m:statistics.mean(x[m] for x in rows) for m in ('accuracy','macro_f1')}
    per=[];grouped=defaultdict(list);contrasts=defaultdict(list)
    def moments(v):
        if len(v)!=3:raise ValueError('Three actual paired F seeds required')
        return dict(mean=statistics.mean(v),sd=statistics.stdev(v),positive_seeds=sum(x>0 for x in v))
    for k,m in sorted(values.items()):
        pid,seed,name,test,view,dim,stratum=k
        for metric,v in m.items():
            per.append(dict(protocol_id=pid,seed=seed,cell=name,test=test,view=view,dimension=dim,stratum=stratum,metric=metric,percent=100*v))
            grouped[pid,name,test,view,dim,stratum,metric].append(100*v)
    conditions={(k[0],*k[3:]) for k in values}
    for pid,test,view,dim,stratum in sorted(conditions):
        for metric in ('accuracy','macro_f1'):
            for seed in SEEDS:
                v={n:100*values[pid,seed,n,test,view,dim,stratum][metric] for n in NAMES}
                effects={'PL_minus_Sup_at_G_L':v['PL+G-L']-v['Sup+G-L'],
                    'PL_minus_Sup_at_G_LU':v['PL+G-LU']-v['Sup+G-LU'],
                    'G_LU_minus_L_at_Sup':v['Sup+G-LU']-v['Sup+G-L'],
                    'G_LU_minus_L_at_PL':v['PL+G-LU']-v['PL+G-L'],
                    'F_PL_x_G_U_interaction':(v['PL+G-LU']-v['Sup+G-LU'])-(v['PL+G-L']-v['Sup+G-L'])}
                for label,x in effects.items():contrasts[pid,label,test,view,dim,stratum,metric].append(x)
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    def dump(name,rows):
        with (out/name).open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump('gfactor_per_seed.csv',per)
    dump('gfactor_summary.csv',[dict(zip(('protocol_id','cell','test','view','dimension','stratum','metric'),k),**moments(v)) for k,v in sorted(grouped.items())])
    dump('gfactor_paired_effects.csv',[dict(zip(('protocol_id','effect','test','view','dimension','stratum','metric'),k),**moments(v)) for k,v in sorted(contrasts.items())])
    dump('gfactor_net_corrections.csv',pairs)
    write(out/'reused_costs.json',dict(PL_F=costs,G_L_shared_cache=G_cache_cost,new_F=0,new_G=0,
        note='Historical F/G ancestors are reused, not free or trained again. Cache cost is one-time shared with external models, not multiplied by51. Sup-F and G-LU original source costs remain in main comparison metadata. Cached classification is not online end-to-end latency.'))
    write(out/'verification.json',dict(status='VERIFIED',paired_source_models=51,readout_cells=204,
        source_score_files=len(cache),serialized_CM_recomputed=checked,three_actual_F_seeds=True,
        conditional_on_fixed_G=True,raw_reuse_verified=True,target_truth_reads=0,score_feedback_used=False))
    write(out/'complete.json',dict(status='PAMF_GFACTOR_ANALYZED',run_id=c['run_id'],models=51,
        cells=204,truth_last=True,target_feedback_forbidden=True,new_F=0,new_G=0))
