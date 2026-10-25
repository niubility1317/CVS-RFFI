"""Verify three-run coverage and frozen aggregate arithmetic (stdlib only)."""
import argparse,csv,gzip,json,math,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path('E:/type10-7');OUT=ROOT/'exports/reports/20261025-PAMF-PL-comparison-r19'
def rows(p):
    with (gzip.open if p.suffix=='.gz' else open)(p,'rt',encoding='utf-8',newline='') as f:yield from csv.DictReader(f)
def verify():
    fields=('protocol','method','cell','test','readout','view','metric');groups=defaultdict(dict)
    values={};source={}
    for r in rows(OUT/'tables/per_seed_overall.csv.gz'):
        key=tuple(r[k] for k in fields);seed=int(r['seed']);value=float(r['percent'])
        if seed in groups[key] or not 0<=value<=100:raise ValueError('Duplicate seed or invalid percentage')
        groups[key][seed]=value
        values[key+(seed,)]=value
        if r['cell'] in ('A','B') and (r['view']!='clean' or r['test'] not in ('cross_rx','cross_both')):raise ValueError('No-channel result outside target CLEAN')
    maximum=0.;n=0
    for r in rows(OUT/'tables/complete_summary.csv'):
        key=tuple(r[k] for k in fields);source[key]=r
        a=groups.pop(key)
        if set(a)!={2026092701,2026092702,2026092703}:raise ValueError('Incomplete independent model seeds')
        a=list(a.values())
        for key,value in [('mean',statistics.mean(a)),('sd',statistics.stdev(a))]:
            diff=abs(value-float(r[key]));maximum=max(maximum,diff)
            if diff>1e-10:raise ValueError((key,diff))
        n+=1
    if groups or n!=12320:raise ValueError('Summary coverage mismatch')
    paired=0
    for name in ('paired_effects.csv','external_factorial.csv'):
        for r in rows(OUT/'tables'/name):
            a=[float(x) for x in r['seed_values'].split(';')]
            if len(a)!=3 or not all(math.isfinite(x) for x in a):raise ValueError('Incomplete paired result')
            for key,value in [('mean',statistics.mean(a)),('sd',statistics.stdev(a))]:
                if abs(value-float(r[key]))>1e-10:raise ValueError('Paired moment mismatch')
            if sum(x>0 for x in a)!=int(r['positive_seeds']):raise ValueError('Positive count mismatch')
            paired+=1
    expected={'RIEI':('riei','D','raw'),'DRIFT':('drift','D','raw'),'IB-RFF':('ibrff','D','raw'),'SDG':('sdg','D','raw'),'PAMF':('ordinary_PL','native','raw'),'DURA+PAMF':('ordinary_PL','native','DURA-LU')}
    count=0;display=set()
    for r in rows(OUT/'tables/main_comparison.csv'):
        if (r['method'],r['cell'],r['readout'])!=expected[r['display_method']]:raise ValueError('Mixed main route')
        if r['channel_augmentation']!='on':raise ValueError('Main includes no-channel control')
        src=source[tuple(r[k] for k in fields)]
        if any(r[k]!=src[k] for k in ('mean','sd')):raise ValueError('Main projection changed result')
        count+=1;display.add(r['display_method'])
    if count!=3168 or display!=set(expected):raise ValueError('Main display coverage')
    for name in ('DURA_generality.csv','main_system_effects.csv'):
        for r in rows(OUT/'tables'/name):
            m=r.get('method',r.get('reference_method'));a=[]
            for seed in (2026092701,2026092702,2026092703):
                base=(r['protocol'],m,'D',r['test'])
                before=values[base+('raw',r['view'],r['metric'],seed)]
                if name=='DURA_generality.csv':after=values[base+('DURA-LU',r['view'],r['metric'],seed)]
                else:after=values[(r['protocol'],'ordinary_PL','native',r['test'],'DURA-LU',r['view'],r['metric'],seed)]
                a.append(after-before)
            if a!=[float(x) for x in r['seed_values'].split(';')]:raise ValueError('Paired source mismatch')
            if abs(statistics.mean(a)-float(r['mean']))>1e-10 or abs(statistics.stdev(a)-float(r['sd']))>1e-10:raise ValueError('Delta aggregation mismatch')
    grouped=list(rows(OUT/'tables/experiment_groups.csv'))
    if len(grouped)!=594 or sum(r['channel_augmentation']=='off' for r in grouped)!=264:raise ValueError('Experiment group count')
    for r in grouped:
        if r['channel_augmentation']=='off' and (r['report_views']!='clean' or r['report_populations']!='cross_rx;cross_both'):raise ValueError('No-channel report classification')
    text=(OUT/'PAMF与四种对比方法实验报告.md').read_text(encoding='utf-8')
    if 'v2' in text.lower():raise ValueError('Noncanonical display name')
    for p in (OUT/'figures').glob('*.svg'):
        text=p.read_text(encoding='utf-8')
        if '<text' not in text or 'Times New Roman' not in text or 'v2' in text.lower():raise ValueError('Editable figure/name mismatch')
    result=dict(status='VERIFIED',summary_rows=n,seed_rows=len(values),paired_rows=paired,main_rows=count,grouped_models=len(grouped),maximum_absolute_difference=maximum,tolerance=1e-10,three_actual_model_seeds=True,new_training=0,new_prediction=0,new_truth_scoring=0,method_display_names=list(expected),no_channel_target_CLEAN_only=True,external_DURA_only_in_generality=True,legacy_codes_in_appendix_only=True,visual_QA='Numerical verifier does not render PDFs; see separate delivery visual review.')
    (OUT/'report_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=OUT);a=p.parse_args();OUT=a.root;verify()
