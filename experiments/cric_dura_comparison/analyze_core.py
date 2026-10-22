"""Independent score-side aggregation of the predeclared 2x2x2 F experiment."""
from collections import defaultdict
import csv
import math
from pathlib import Path
import statistics
from .io import read, write
from .core_groups import SEEDS, switches

VIEWS=('clean','practical_high','practical_mid','practical_low_suburban',
       'practical_high_urban','practical_mid_urban','practical_low_urban')


def verified_metrics(row,classes):
    cm=row['confusion']
    if (len(cm)!=classes or any(len(line)!=classes for line in cm)
            or any(type(n) is not int or n<0 for line in cm for n in line)):
        raise ValueError('Invalid serialized confusion matrix')
    count=sum(map(sum,cm));correct=sum(cm[i][i] for i in range(classes))
    if count<=0 or row['count']!=count:raise ValueError('CM sample count differs')
    f1=[]
    for i in range(classes):
        denominator=sum(cm[i])+sum(line[i] for line in cm)
        f1.append(2*cm[i][i]/denominator if denominator else 0.)
    actual=dict(accuracy=correct/count,macro_f1=sum(f1)/classes)
    if any(not math.isfinite(row[k]) or abs(row[k]-v)>1e-10 for k,v in actual.items()):
        raise ValueError('Saved score and CM reconstruction disagree')
    return actual


def key(row):
    return (row['test'],row['view'],row['dimension'],row['stratum'])


def moments(values):
    if len(values)!=3:raise ValueError('Three actual model seeds required')
    return dict(mean=statistics.mean(values),sd=statistics.stdev(values),
        minimum=min(values),maximum=max(values),positive_seeds=sum(v>0 for v in values))


def csvfile(path,rows):
    if not rows:raise ValueError('Empty predeclared table')
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(c):
    cells=c['cells'];expected={(bits,seed) for bits in ('000','001','010','011','100','101','110','111') for seed in SEEDS}
    if len(cells)!=24 or {(v['bits'],v['model_seed']) for v in cells}!=expected:
        raise ValueError('Incomplete/duplicate core factorial')
    cache={};values={};paired=[];checked=0;cost=[]
    for cell in cells:
        bits=cell['bits'];seed=cell['model_seed'];baseline_raw=None
        if set(cell['readouts'])!={'raw','DURA-L','DURA-LU'}:raise ValueError('Incomplete raw/L/LU readouts')
        for readout,record in cell['readouts'].items():
            path=record['scores']
            if path not in cache:
                done=read(record['completion'])
                if done['status']!='SCORED_COMPLETE':raise ValueError('Score not independently complete')
                obj=read(path)
                if obj['status']!='SCORED' or obj.get('truth_last') is not True or obj.get('target_feedback_forbidden') is not True:
                    raise ValueError('Frozen truth-last score required')
                cache[path]=obj
            obj=cache[path]
            if (obj['run_id'],obj['row_id'],obj['model_seed'])!=(record['run_id'],record['row_id'],seed):
                raise ValueError('Score belongs to another source row/seed')
            raw={key(r):r for r in obj['results'] if r['readout']=='raw'}
            if not raw or len(raw)!=sum(r['readout']=='raw' for r in obj['results']):
                raise ValueError('Raw diagnostic matrix absent or duplicated')
            raw_cm={k:r['confusion'] for k,r in raw.items()}
            if baseline_raw is None:baseline_raw=raw_cm
            elif baseline_raw!=raw_cm:raise ValueError('Same-F raw CM differs across reused readouts')
            selected=[r for r in obj['results'] if r['readout']==record['readout']]
            if len({key(r) for r in selected})!=len(selected) or {key(r) for r in selected}!=set(raw):
                raise ValueError('Incomplete readout strata/views')
            if {r['view'] for r in selected}!=set(VIEWS):raise ValueError('Seven fixed views required')
            pairs={key(r):r for r in obj['paired_raw'] if r['readout']==record['readout']}
            for row in selected:
                metric=verified_metrics(row,c['num_classes']);checked+=1
                verified_metrics(raw[key(row)],c['num_classes'])
                values[(bits,seed,readout,*key(row))]=metric
                if readout!='raw':
                    p=pairs[key(row)];r=raw[key(row)]
                    net=sum(row['confusion'][i][i]-r['confusion'][i][i] for i in range(c['num_classes']))
                    if p['net']!=net or p['repairs']-p['damage']!=net or p['count']!=row['count']:
                        raise ValueError('Paired corrections and CM difference disagree')
                    paired.append(dict(bits=bits,seed=seed,readout=readout,test=row['test'],view=row['view'],
                        dimension=row['dimension'],stratum=row['stratum'],count=p['count'],repairs=p['repairs'],damage=p['damage'],net=p['net']))
        source=cell['source'];completion=Path(source['completion'])
        evidence=read(completion) if completion.is_file() else {}
        if source['new_F']:
            cfg=evidence['config']
            if cfg.get('core_mechanisms')!=switches(bits) or evidence.get('updates')!=11100:
                raise ValueError('New F mechanism or update budget differs')
        cost.append(dict(bits=bits,seed=seed,new_F=source['new_F'],
            cumulative_F_epochs=evidence.get('epoch',50),cumulative_F_updates=evidence.get('updates',11100),
            shared_worker_seconds=evidence.get('seconds'),
            scope='F elapsed shared-worker cost; fixed G ancestry separate; original T1 cost may be referenced externally'))
    # The practical endpoint averages the six per-view metrics within seed first.
    strata={k[3:] for k in values if k[4]=='clean'}
    for bits,seed in expected:
        for readout in ('raw','DURA-L','DURA-LU'):
            for test,_,dimension,stratum in strata:
                rows=[values[(bits,seed,readout,test,v,dimension,stratum)] for v in VIEWS[1:]]
                values[(bits,seed,readout,test,'six_practical_mean',dimension,stratum)]={
                    metric:statistics.mean(r[metric] for r in rows) for metric in ('accuracy','macro_f1')}
    per_seed=[];grouped=defaultdict(list)
    for k,metric in sorted(values.items()):
        bits,seed,reader,test,view,dim,stratum=k
        for name,v in metric.items():
            per_seed.append(dict(bits=bits,seed=seed,readout=reader,test=test,view=view,dimension=dim,
                stratum=stratum,metric=name,percent=100*v))
            grouped[(bits,reader,test,view,dim,stratum,name)].append(100*v)
    summary=[dict(zip(('bits','readout','test','view','dimension','stratum','metric'),k),**moments(v)) for k,v in sorted(grouped.items())]
    effects=[]
    conditions={k[2:] for k in values}
    for reader,test,view,dim,stratum in sorted(conditions):
        for metric in ('accuracy','macro_f1'):
            per_effect=defaultdict(list)
            for seed in SEEDS:
                f={bits:100*values[(bits,seed,reader,test,view,dim,stratum)][metric] for bits,_ in expected if _==seed}
                for index,label in enumerate(('A','B','C')):
                    without=list('111');without[index]='0'
                    per_effect[label+'_at_other_groups_on'].append(f['111']-f[''.join(without)])
                    difference=[f[b]-f[b[:index]+'0'+b[index+1:]] for b in f if b[index]=='1']
                    per_effect[label+'_factorial_average'].append(statistics.mean(difference))
                for i,j,label in ((0,1,'AB'),(0,2,'AC'),(1,2,'BC')):
                    terms=[]
                    for bit in ('0','1'):
                        base=['0','0','0'];base[3-i-j]=bit
                        a=base.copy();b=base.copy();both=base.copy();a[i]='1';b[j]='1';both[i]=both[j]='1'
                        terms.append(f[''.join(both)]-f[''.join(a)]-f[''.join(b)]+f[''.join(base)])
                    per_effect[label+'_interaction_average'].append(statistics.mean(terms))
                per_effect['ABC_interaction'].append(sum((1 if b.count('1')%2 else -1)*v for b,v in f.items()))
            for label,v in sorted(per_effect.items()):
                effects.append(dict(effect=label,readout=reader,test=test,view=view,dimension=dim,stratum=stratum,
                    metric=metric,**moments(v)))
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    csvfile(out/'core_per_seed.csv',per_seed);csvfile(out/'core_summary.csv',summary)
    csvfile(out/'core_paired_effects.csv',effects);csvfile(out/'core_net_corrections.csv',paired);csvfile(out/'core_F_cost.csv',cost)
    report=['# PAMF 三核心机制：完整2×2×2\n',
        'A=跨RX身份对比；B=成对预测一致性；C=完整选择性半监督。共同真实标签CE骨架及部署网络不变。\n',
        '主表为111/011/101/110；其余格用于交互，不按收益选择组合。三Fseed条件于固定G-L/G-LU；C-off+G-LU不是全监督系统。\n',
        '|ABC|读出|clean Acc%|六景均值 Acc%|\n|---|---|---:|---:|\n']
    for bits in c['main_table']:
        for reader in ('raw','DURA-L','DURA-LU'):
            found={r['view']:r for r in summary if r['bits']==bits and r['readout']==reader and r['test']=='main'
                and r['dimension']=='overall' and r['stratum']=='ALL' and r['metric']=='accuracy'}
            fmt=lambda k:'%.4f ± %.4f'%(found[k]['mean'],found[k]['sd'])
            report.append('|%s|%s|%s|%s|\n'%(bits,reader,fmt('clean'),fmt('six_practical_mean')))
    (out/'core_results.md').write_text('\n'.join(report),encoding='utf-8')
    write(out/'verification.json',dict(status='VERIFIED',logical_cells=24,model_seeds=list(SEEDS),
        readouts=3,views=7,serialized_CM_recomputed=checked,raw_CM_reuse_verified=True,
        target_truth_reads=0,score_feedback_used=False,conditional_on_fixed_G=True))
    write(out/'complete.json',dict(status='CORE_ANALYZED',cells=24,source_score_files=len(cache),
        CM_checked=checked,target_truth_reads=0,three_model_seeds=list(SEEDS)))
