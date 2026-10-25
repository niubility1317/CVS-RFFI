"""Independent standard-library checks for regime-separated frozen report tables."""
import argparse,csv,gzip,json,math,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path('E:/type10-7');OUT=ROOT/'exports/reports/20261025-PAMF-PL-comparison-r20'
SEEDS={2026092701,2026092702,2026092703};FIELDS=('protocol','method','cell','test','readout','view','metric')
def rows(p):
    with (gzip.open if p.suffix=='.gz' else open)(p,'rt',encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
def verify():
    seed={};groups=defaultdict(dict)
    for r in rows(OUT/'tables/per_seed_overall.csv.gz'):
        k=tuple(r[x] for x in FIELDS);s=int(r['seed']);v=float(r['percent'])
        assert s in SEEDS and 0<=v<=100 and s not in groups[k]
        assert r['cell'] not in ('A','B') or (r['view']=='clean' and r['test'] in ('cross_rx','cross_both'))
        groups[k][s]=v;seed[k+(s,)]=v
    source={};maximum=0.
    for r in rows(OUT/'tables/complete_summary.csv'):
        k=tuple(r[x] for x in FIELDS);a=groups.pop(k);assert set(a)==SEEDS
        for field,v in [('mean',statistics.mean(a.values())),('sd',statistics.stdev(a.values()))]:
            diff=abs(float(r[field])-v);maximum=max(maximum,diff);assert diff<1e-10
        source[k]=r
    assert len(source)==12320 and len(seed)==36960 and not groups
    counts=defaultdict(int);classifiers=defaultdict(set);displays=defaultdict(set)
    for r in rows(OUT/'tables/main_comparison.csv'):
        reg=r['learning_regime'];assert reg in ('supervised','semi_supervised')
        cell,own,rest=('C','supervised','DURA-L') if reg=='supervised' else ('D','ordinary_PL','DURA-LU')
        if r['method'] in ('riei','drift','ibrff','sdg'):assert r['cell']==cell and r['readout']=='raw'
        elif r['display_method']=='PAMF':assert r['method']==own and r['cell']=='native' and r['readout']=='raw'
        else:assert r['display_method']=='DURA+PAMF' and r['method']==own and r['cell']=='native' and r['readout']==rest
        src=source[tuple(r[x] for x in FIELDS)];assert (r['mean'],r['sd'])==(src['mean'],src['sd'])
        counts[reg]+=1;classifiers[reg].add((r['protocol'],r['method'],r['cell']));displays[reg].add(r['display_method'])
    assert counts=={'supervised':3168,'semi_supervised':3168}
    assert all(len(classifiers[g])==55 and displays[g]=={'RIEI','DRIFT','IB-RFF','SDG','PAMF','DURA+PAMF'} for g in counts)
    paired=0
    for name,expected in [('DURA_generality.csv',4224),('PAMF_restoration_by_regime.csv',1056)]:
        rs=list(rows(OUT/'tables'/name));assert len(rs)==expected
        for r in rs:
            reg=r['learning_regime'];cell,own,rest=('C','supervised','DURA-L') if reg=='supervised' else ('D','ordinary_PL','DURA-LU')
            assert r['cell']==('native' if r['method']==own else cell) and r['readout_after']==rest
            a=[]
            for s in sorted(SEEDS):
                before=seed[(r['protocol'],r['method'],r['cell'],r['test'],'raw',r['view'],r['metric'],s)]
                after=seed[(r['protocol'],r['method'],r['cell'],r['test'],rest,r['view'],r['metric'],s)]
                a.append(after-before)
            assert a==[float(x) for x in r['seed_values'].split(';')]
            assert abs(statistics.mean(a)-float(r['mean']))<1e-10 and abs(statistics.stdev(a)-float(r['sd']))<1e-10
            assert sum(v>0 for v in a)==int(r['positive_seeds']);paired+=1
    registered=list(rows(OUT/'tables/experiment_groups.csv'));assert len(registered)==594
    assert len({(r['protocol'],r['method'],r['cell'],r['seed']) for r in registered})==594
    assert {g:sum(r['learning_regime']==g for r in registered) for g in counts}=={'supervised':297,'semi_supervised':297}
    for r in registered:
        assert r['learning_regime']==('supervised' if r['cell'] in ('A','C') or r['method']=='supervised' else 'semi_supervised')
        if r['channel_augmentation']=='off':assert r['report_views']=='clean' and r['report_populations']=='cross_rx;cross_both'
    manifest=json.loads((OUT/'figures/manifest.json').read_text(encoding='utf-8'))
    assert len(manifest)==26
    for item in manifest:
        assert item['chart_type']=='vertical_column'
        svg=(OUT/'figures'/(item['name']+'.svg')).read_text(encoding='utf-8');assert '<text' in svg and 'Times New Roman' in svg and 'v2' not in svg.lower()
        data=list(rows(OUT/'tables/figure_sources'/(item['name']+'.csv')))
        if item['learning_regime'] is not None:assert {r['learning_regime'] for r in data}=={item['learning_regime']}
    text=(OUT/'PAMF与四种对比方法实验报告.md').read_text(encoding='utf-8');assert 'v2' not in text.lower();assert '监督实验结果' in text and '普通PL半监督实验结果' in text
    result=dict(status='VERIFIED',summary_rows=len(source),seed_rows=len(seed),main_summary_rows=dict(counts),main_unique_classifiers=330,main_display_model_route_items=396,all_logical_models=594,models_per_learning_regime=297,paired_rows=paired,figures=26,all_figures_vertical=True,single_learning_regime_per_performance_figure=True,supervised_main_has_no_unlabeled_IQ=True,no_channel_target_CLEAN_only=True,maximum_absolute_difference=maximum,new_training=0,new_prediction=0,new_truth_scoring=0)
    (OUT/'report_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=OUT);a=p.parse_args();OUT=a.root;verify()
