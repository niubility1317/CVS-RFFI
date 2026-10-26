"""Stdlib replay of pooled confusion matrices and same-seed branch effects."""
import argparse,csv,gzip,json,math,statistics
from collections import defaultdict
from pathlib import Path
from verify_pamf_branch_ieee_focus import verify as verify_primary
DEFAULT=Path('E:/type10-7/exports/reports/20261026-PAMF-branch-IEEE-r23')
ARMS=('full','NO_TIME','NO_SIGNATURE','NO_PA','NO_REFERENCE','WRONG_REFERENCE')
SEEDS=(2026092701,2026092702,2026092703)
TASKS=('cross_rx','cross_both');COUNTS=dict(zip(TASKS,(126000,42000)))
SCENES=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')

def rows(p):
    with (gzip.open if p.suffix=='.gz' else open)(p,'rt',encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)

def records(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)

def scores(cm):
    total=sum(map(sum,cm));correct=sum(cm[k][k] for k in range(6));f=[]
    for k in range(6):
        tp=cm[k][k];fp=sum(cm[i][k] for i in range(6) if i!=k);fn=sum(cm[k][j] for j in range(6) if j!=k)
        f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return {'accuracy':100*correct/total,'macro_f1':100*sum(f)/6}

def verify(root):
    result=verify_primary(root,expected_figures=4,verbose=False);maximum=result['maximum_absolute_numeric_difference']
    def close(a,b):
        nonlocal maximum
        d=abs(float(a)-float(b));maximum=max(maximum,d)
        if not math.isfinite(d) or d>1e-10:raise ValueError(('Pooled arithmetic mismatch',a,b))
    source={};population_metrics={}
    for r in records(root/'inputs/population_confusions.jsonl.gz'):
        a,s,t,v=r['structure'],int(r['seed']),r['population'],r['view'];key=(a,s,t,v)
        assert a in ARMS and s in SEEDS and t in TASKS and v in SCENES and r['reader']=='PAMF_direct' and key not in source
        cm=r['confusion'];assert len(cm)==6 and all(len(line)==6 for line in cm)
        assert all(type(x) is int and x>=0 for line in cm for x in line)
        assert r['count']==sum(map(sum,cm))==COUNTS[t]
        mm=scores(cm);close(mm['accuracy'],r['frozen_accuracy']);close(mm['macro_f1'],r['frozen_macro_f1'])
        source[key]=cm;population_metrics[key]=mm
    assert len(source)==252
    pooled_cm={};pooled_metrics={};weighted_F1_gap=0.
    for a in ARMS:
        for s in SEEDS:
            for v in SCENES:
                cm=[[sum(source[a,s,t,v][i][j] for t in TASKS) for j in range(6)] for i in range(6)]
                assert sum(map(sum,cm))==168000;pooled_cm[a,s,v]=cm;mm=scores(cm)
                for metric,value in mm.items():pooled_metrics[a,s,v,metric]=value
                close(mm['accuracy'],sum(COUNTS[t]*population_metrics[a,s,t,v]['accuracy'] for t in TASKS)/168000)
                weighted_F1_gap=max(weighted_F1_gap,abs(mm['macro_f1']-sum(COUNTS[t]*population_metrics[a,s,t,v]['macro_f1'] for t in TASKS)/168000))
            for m in ('accuracy','macro_f1'):pooled_metrics[a,s,'six_practical_mean',m]=statistics.mean(pooled_metrics[a,s,v,m] for v in SCENES[1:])
    cm_rows=list(records(root/'inputs/pooled_confusions.jsonl.gz'));assert len(cm_rows)==126
    for r in cm_rows:assert r['count']==168000 and r['confusion']==pooled_cm[r['structure'],int(r['seed']),r['view']]
    population=list(rows(root/'tables/population_per_seed.csv'));assert len(population)==576
    for r in population:
        a,s,t,v,m=r['structure'],int(r['seed']),r['population'],r['view'],r['metric']
        value=statistics.mean(population_metrics[a,s,t,x][m] for x in SCENES[1:]) if v=='six_practical_mean' else population_metrics[a,s,t,v][m]
        close(value,r['percent']);assert int(r['packets_per_scene'])==COUNTS[t]
    pooled=list(rows(root/'tables/pooled_per_seed.csv'));assert len(pooled)==288
    for r in pooled:close(pooled_metrics[r['structure'],int(r['seed']),r['view'],r['metric']],r['percent'])
    def moments(r,values):
        close(statistics.mean(values),r['mean']);close(statistics.stdev(values),r['sd'])
        for a,b in zip(values,map(float,r['seed_values'].split(';'))):close(a,b)
        assert int(r['negative_seeds'])==sum(x<0 for x in values) and int(r['positive_seeds'])==sum(x>0 for x in values)
    summary=list(rows(root/'tables/pooled_summary.csv'));paired=list(rows(root/'tables/pooled_variant_minus_full.csv'));assert len(summary)==96 and len(paired)==80
    for r in summary:moments(r,[pooled_metrics[r['structure'],s,r['view'],r['metric']] for s in SEEDS])
    for r in paired:
        assert r['definition']=='variant_minus_full' and r['structure'] in ARMS[1:]
        moments(r,[pooled_metrics[r['structure'],s,r['view'],r['metric']]-pooled_metrics['full',s,r['view'],r['metric']] for s in SEEDS])
    figure=list(rows(root/'tables/figure_composite.csv'));assert len(figure)==12
    for r in figure:
        a,v=r['structure'],r['view'];values=[pooled_metrics[a,s,v,'accuracy'] for s in SEEDS];delta=[pooled_metrics[a,s,v,'accuracy']-pooled_metrics['full',s,v,'accuracy'] for s in SEEDS]
        close(statistics.mean(values),r['mean']);close(statistics.stdev(values),r['sd']);close(statistics.mean(delta),r['delta_mean']);close(statistics.stdev(delta),r['delta_sd'])
        for x,y in zip(delta,map(float,r['delta_seed_values'].split(';'))):close(x,y)
    weights=list(rows(root/'tables/population_weights.csv'));assert len(weights)==2
    for r in weights:assert int(r['packets_per_scene'])==COUNTS[r['population']];close(float(r['weight']),COUNTS[r['population']]/168000)
    body=(root/'report.md').read_text(encoding='utf-8');tex=(root/'ieee/main.tex').read_text(encoding='utf-8')
    assert '两类测试综合统计' in body and '168000' in body and '不是两个准确率的简单平均' in body
    assert '30000' not in body and 'cross_date' not in body and '三类' not in body
    assert '\\label{fig:composite_testing}' in tex and '\\label{tab:pooled}' in tex
    result.update(test_population='joint_generalization_primary_plus_pooled_two_disjoint_populations',primary_packets_per_scene=42000,pooled_packets_per_scene=168000,population_counts=COUNTS,population_CMs_replayed=252,pooled_CMs_replayed=126,pooled_seed_rows=288,pooled_summary_rows=96,pooled_paired_rows=80,pooled_macro_f1_recomputed=True,weighted_F1_average_difference_max_pp=weighted_F1_gap,aggregation_within_seed_before_three_seed_statistics=True,maximum_absolute_numeric_difference=maximum)
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);a=p.parse_args();verify(a.root)
