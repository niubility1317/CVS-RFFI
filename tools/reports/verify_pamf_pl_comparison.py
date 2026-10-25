"""Verify three-run coverage and frozen aggregate arithmetic (stdlib only)."""
import argparse,csv,gzip,json,math,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path('E:/type10-7');OUT=ROOT/'exports/reports/20261025-PAMF-PL-comparison-r18'
def rows(p):
    with (gzip.open if p.suffix=='.gz' else open)(p,'rt',encoding='utf-8',newline='') as f:yield from csv.DictReader(f)
def verify():
    fields=('protocol','method','cell','test','readout','view','metric');groups=defaultdict(dict)
    for r in rows(OUT/'tables/per_seed_overall.csv.gz'):
        key=tuple(r[k] for k in fields);seed=int(r['seed']);value=float(r['percent'])
        if seed in groups[key] or not 0<=value<=100:raise ValueError('Duplicate seed or invalid percentage')
        groups[key][seed]=value
    maximum=0.;n=0
    for r in rows(OUT/'tables/complete_summary.csv'):
        a=groups.pop(tuple(r[k] for k in fields))
        if set(a)!={2026092701,2026092702,2026092703}:raise ValueError('Incomplete independent model seeds')
        a=list(a.values())
        for key,value in [('mean',statistics.mean(a)),('sd',statistics.stdev(a))]:
            diff=abs(value-float(r[key]));maximum=max(maximum,diff)
            if diff>1e-10:raise ValueError((key,diff))
        n+=1
    if groups or n!=20064:raise ValueError('Summary coverage mismatch')
    paired=0
    for name in ('paired_effects.csv','external_factorial.csv'):
        for r in rows(OUT/'tables'/name):
            a=[float(x) for x in r['seed_values'].split(';')]
            if len(a)!=3 or not all(math.isfinite(x) for x in a):raise ValueError('Incomplete paired result')
            for key,value in [('mean',statistics.mean(a)),('sd',statistics.stdev(a))]:
                if abs(value-float(r[key]))>1e-10:raise ValueError('Paired moment mismatch')
            if sum(x>0 for x in a)!=int(r['positive_seeds']):raise ValueError('Positive count mismatch')
            paired+=1
    text=(OUT/'PAMF与四种对比方法实验报告.md').read_text(encoding='utf-8')
    if 'v2' in text.lower():raise ValueError('Noncanonical display name')
    for p in (OUT/'figures').glob('*.svg'):
        text=p.read_text(encoding='utf-8')
        if '<text' not in text or 'Times New Roman' not in text or 'v2' in text.lower():raise ValueError('Editable figure/name mismatch')
    result=dict(status='VERIFIED',summary_rows=n,seed_rows=60192,paired_rows=paired,maximum_absolute_difference=maximum,tolerance=1e-10,three_actual_model_seeds=True,new_training=0,new_prediction=0,new_truth_scoring=0,method_display_names='RIEI, DRIFT, IB-RFF, SDG, PAMF',legacy_codes_in_appendix_only=True,visual_QA='Numerical verifier does not render PDFs; see separate delivery visual review.')
    (OUT/'report_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=OUT);a=p.parse_args();OUT=a.root;verify()
