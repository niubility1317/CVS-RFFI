"""Independent stdlib replay of the PAMF-only joint-domain branch report."""
import argparse,csv,gzip,json,math,statistics
from collections import defaultdict
from pathlib import Path
DEFAULT=Path('E:/type10-7/exports/reports/20261026-PAMF-branch-IEEE-r21')
ARMS=('full','NO_TIME','NO_SIGNATURE','NO_PA','NO_REFERENCE','WRONG_REFERENCE')
SEEDS=(2026092701,2026092702,2026092703)
SCENES=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')

def rows(p):
    with (gzip.open if p.suffix=='.gz' else open)(p,'rt',encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)

def verify(root,expected_figures=3,verbose=True):
    maximum=0.
    def close(a,b):
        nonlocal maximum
        d=abs(float(a)-float(b));maximum=max(maximum,d)
        if not math.isfinite(d) or d>1e-10:raise ValueError(('Numeric mismatch',a,b))
    source=list(rows(root/'inputs/frozen_per_seed.csv'));ix={}
    for r in source:
        assert r['test']=='cross_both' and r['readout']=='raw' and r['group']=='strict_structure' and r['protocol']=='F0'
        assert r['method'] in {'CE_'+a for a in ARMS} and r['dimension']=='overall' and r['stratum']=='ALL'
        assert int(r['packets_per_view'])==42000 and int(r['seed']) in SEEDS
        key=(r['method'][3:],r['view'],r['metric'],int(r['seed']));assert key not in ix
        ix[key]=float(r['percent'])
    assert len(ix)==len(source)==288
    for a in ARMS:
        for m in ('accuracy','macro_f1'):
            for s in SEEDS:close(statistics.mean(ix[a,v,m,s] for v in SCENES[1:]),ix[a,'six_practical_mean',m,s])
    def moments(r,a):
        close(statistics.mean(a),r['mean']);close(statistics.stdev(a),r['sd'])
        assert len(a)==3 and len(r['seed_values'].split(';'))==3
        for x,y in zip(a,map(float,r['seed_values'].split(';'))):close(x,y)
        assert int(r['negative_seeds'])==sum(x<0 for x in a) and int(r['positive_seeds'])==sum(x>0 for x in a)
    summary=list(rows(root/'tables/summary.csv'));paired=list(rows(root/'tables/variant_minus_full.csv'))
    assert len(summary)==96 and len(paired)==80
    si={};pi={}
    for r in summary:
        key=r['structure'],r['view'],r['metric'];assert key not in si;si[key]=r
        moments(r,[ix[*key,s] for s in SEEDS])
    for r in paired:
        key=r['structure'],r['view'],r['metric'];assert key not in pi;pi[key]=r
        assert r['definition']=='variant_minus_full' and r['structure'] in ARMS[1:]
        moments(r,[ix[*key,s]-ix['full',r['view'],r['metric'],s] for s in SEEDS])
    assert float(pi['NO_PA','six_practical_mean','accuracy']['mean'])<0
    assert float(pi['NO_SIGNATURE','six_practical_mean','accuracy']['mean'])>0
    expected_counts={'figure_accuracy.csv':42,'figure_paired.csv':20,'figure_scene_changes.csv':35}
    for name,count in expected_counts.items():
        rr=list(rows(root/'tables'/name));assert len(rr)==count
        for r in rr:
            reference=si[r['structure'],r['view'],r['metric']] if name=='figure_accuracy.csv' else pi[r['structure'],r['view'],r['metric']]
            close(reference['mean'],r['mean']);close(reference['sd'],r['sd'])
    groups=defaultdict(dict)
    fields=('method','view','dimension','stratum','metric')
    strata=list(rows(root/'tables/per_stratum_per_seed.csv.gz'))
    for r in strata:
        assert r['test']=='cross_both' and r['readout']=='raw' and r['method'] in {'CE_'+a for a in ARMS}
        key=tuple(r[k] for k in fields);seed=int(r['seed']);assert seed not in groups[key];groups[key][seed]=float(r['percent'])
    assert len(strata)==6624 and len(groups)==2208 and all(set(v)==set(SEEDS) for v in groups.values())
    for filename,count,is_pair in [('per_stratum_summary.csv.gz',2208,False),('per_stratum_variant_minus_full.csv.gz',1840,True)]:
        rr=list(rows(root/'tables'/filename));assert len(rr)==count
        for r in rr:
            key=tuple(r[k] for k in fields);a=[groups[key][s] for s in SEEDS]
            if is_pair:
                assert r['definition']=='variant_minus_full';a=[groups[key][s]-groups['CE_full',*key[1:]][s] for s in SEEDS]
            moments(r,a)
    manifest=json.loads((root/'figures/manifest.json').read_text(encoding='utf-8'))
    assert len(manifest)==expected_figures
    for item in manifest:
        svg=(root/'figures'/(item['name']+'.svg')).read_text(encoding='utf-8')
        assert '<text' in svg and 'Times New Roman' in svg and (root/item['source_csv']).exists()
    tex=(root/'ieee/main.tex').read_text(encoding='utf-8');body=(root/'report.md').read_text(encoding='utf-8')
    assert '\\documentclass[10pt,journal]{IEEEtran}' in tex
    assert 'DURA' not in body and '接入' not in body and '时延' not in body and '参数量' not in body
    assert 'cross_rx' not in body and 'cross_date' not in body and 'v2' not in body.lower()
    result=dict(status='VERIFIED',models=18,structures=6,selected_variants=5,model_seeds=list(SEEDS),test_population='unseen_receiver_and_earlier_unseen_date',reader='PAMF_direct_only',packets_per_scene=42000,overall_seed_rows=288,summary_rows=96,paired_rows=80,stratum_seed_rows=6624,stratum_summary_rows=2208,stratum_paired_rows=1840,delta_definition='variant_minus_full',negative_is_degradation=True,maximum_absolute_numeric_difference=maximum,tolerance=1e-10,figures=expected_figures,IEEEtran=True,new_training=0,new_prediction=0,new_truth_access=0)
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if verbose:print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);a=p.parse_args();verify(a.root)
