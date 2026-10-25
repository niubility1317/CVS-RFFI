"""Portable stdlib reproduction of the supervised architecture report arithmetic."""
import argparse,csv,gzip,json,math,statistics
from collections import defaultdict
from pathlib import Path

DEFAULT=Path('E:/type10-7/exports/reports/20261025-PAMF-branch-ablation-r19')
SEEDS=(2026092701,2026092702,2026092703)
PARAMS={'full':232993,'SIG_TIME':196001,'NO_TIME':232993,'SHUFFLE_TIME':232993,'NO_SIGNATURE':177025,'NO_PA':221953,'NO_REFERENCE':220193,'WRONG_REFERENCE':232993,'COV_X3':232993}

def rows(p):
    with (gzip.open if p.suffix=='.gz' else open)(p,'rt',encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)

def quantile(a,p):
    a=sorted(a);x=(len(a)-1)*p;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)

def verify(root):
    maximum=0.
    def close(a,b):
        nonlocal maximum
        d=abs(float(a)-float(b));maximum=max(maximum,d)
        if not math.isfinite(d) or d>1e-9:raise ValueError(('Numeric mismatch',a,b))
    groups=defaultdict(dict);n=0
    fields=('method','test','readout','view','dimension','stratum','metric')
    for r in rows(root/'tables/all_strata_per_seed.csv.gz'):
        assert r['method'] in {'CE_'+a for a in PARAMS} and r['readout'] in ('raw','DURA-L')
        key=tuple(r[k] for k in fields);seed=int(r['seed']);value=float(r['percent'])
        assert seed not in groups[key] and 0<=value<=100
        groups[key][seed]=value;n+=1
    assert n==69984 and len(groups)==23328
    for values in groups.values():assert set(values)==set(SEEDS)
    for filename,expected in [('per_stratum_summary.csv.gz',23328),('summary_overall.csv',864)]:
        count=0
        for r in rows(root/'tables'/filename):
            a=list(groups[tuple(r[k] for k in fields)].values())
            close(statistics.mean(a),r['mean']);close(statistics.stdev(a),r['sd']);count+=1
        assert count==expected
    for filename,expected in [('per_stratum_full_minus_variant.csv.gz',20736),('full_minus_variant.csv',768)]:
        count=0
        for r in rows(root/'tables'/filename):
            key=(r['method'],*[r[k] for k in fields[1:]]) if 'method' in r else ('CE_'+r['arm'],r['test'],r['readout'],r['view'],'overall','ALL',r['metric'])
            base=groups[('CE_full',*key[1:])];variant=groups[key]
            a=[base[s]-variant[s] for s in SEEDS]
            close(statistics.mean(a),r['mean']);close(statistics.stdev(a),r['sd'])
            for x,y in zip(a,map(float,r['seed_values'].split(';'))):close(x,y)
            assert sum(x>0 for x in a)==int(r['positive_seeds']);count+=1
        assert count==expected
    def value(arm,t,rd,metric,s):return groups['CE_'+arm,t,rd,'six_practical_mean','overall','ALL',metric][s]
    for r in rows(root/'tables/restoration_by_branch_interaction.csv'):
        a=[(value('full',r['test'],'DURA-L',r['metric'],s)-value(r['arm'],r['test'],'DURA-L',r['metric'],s))-(value('full',r['test'],'raw',r['metric'],s)-value(r['arm'],r['test'],'raw',r['metric'],s)) for s in SEEDS]
        close(statistics.mean(a),r['mean']);close(statistics.stdev(a),r['sd'])
    total_models=0
    for r in rows(root/'tables/training_exposures.csv'):
        assert r['U_enabled']=='False' and r['U_truth_read']=='False'
        assert int(r['U_seen'])==int(r['U_accepted'])==int(r['source_actual_EMA_updates'])==0
        assert int(r['L_packet_forward_evaluations'])==11100*384;total_models+=1
    assert total_models==27
    epochs=defaultdict(set)
    for r in rows(root/'tables/training_epochs.csv'):
        assert r['U_enabled']==r['EMA_enabled']=='False' and float(r['label_smoothing'])==0
        assert float(r['classification_only'])==1 and float(r['L_forward_evaluations_per_update'])==384
        key=(r['method'],r['seed']);e=int(r['logical_epoch']);assert e not in epochs[key];epochs[key].add(e)
    assert len(epochs)==27 and all(es==set(range(1,51)) for es in epochs.values())
    stages=defaultdict(list);folders=set()
    for r in rows(root/'tables/source_stage_cost.csv'):
        assert r['folder'] not in folders;folders.add(r['folder']);stages[r['arm'],r['seed']].append(r)
    assert len(folders)==81
    for r in rows(root/'tables/source_cost.csv'):
        ss=stages[r['arm'],r['seed']];assert len(ss)==3
        assert sum(int(s['epochs']) for s in ss)==50 and sum(int(s['updates']) for s in ss)==11100
        close(sum(float(s['seconds']) for s in ss)/3600,r['cumulative_shared_worker_hours'])
    for r in rows(root/'tables/parameters.csv'):
        assert int(r['loaded_parameters'])==PARAMS[r['arm']]
        close(100*(PARAMS['full']-PARAMS[r['arm']])/PARAMS['full'],r['reduction_percent'])
    timing=defaultdict(list);case_rounds=0
    with gzip.open(root/'inputs/timing_samples.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);assert r['exclusive_GPU'] is True
            timing[r['case'],r['batch']].append(r);case_rounds+=1
    assert case_rounds==450 and len(timing)==90
    for r in rows(root/'tables/timing.csv'):
        case='CE_'+r['arm']+('_F' if r['route']=='direct' else '_GF');ss=timing[case,int(r['batch'])]
        assert {s['round'] for s in ss}==set(range(5))
        a=[x for s in ss for x in s['wall_ms']]
        close(statistics.median(a),r['wall_median_ms']);close(quantile(a,.95),r['wall_P95_ms'])
        close(statistics.stdev(statistics.median(s['wall_ms']) for s in ss),r['round_median_sd'])
        close(int(r['batch'])*1000/statistics.median(a),r['throughput_IQ_s'])
    net=0
    for r in rows(root/'tables/net_corrections.csv.gz'):
        assert int(r['repairs'])-int(r['damage'])==int(r['net'])
        if r['dimension']=='overall':
            a=groups[r['method'],r['test'],'DURA-L',r['view'],'overall','ALL','accuracy'][int(r['seed'])]
            b=groups[r['method'],r['test'],'raw',r['view'],'overall','ALL','accuracy'][int(r['seed'])]
            close(a-b,100*int(r['net'])/int(r['count']));net+=1
    assert net==567
    for p in (root/'figures').glob('*.svg'):
        text=p.read_text(encoding='utf-8');assert '<text' in text and 'Times New Roman' in text
    result=dict(status='VERIFIED',scope='Classification-only supervised network branches',models=27,structures=9,model_seeds=list(SEEDS),overall_summary_rows=864,all_strata_seed_rows=n,all_strata_summary_rows=23328,all_strata_paired_rows=20736,training_epochs=1350,unique_training_segments=81,timing_case_rounds=450,restoration_net_count_checks=net,maximum_absolute_numeric_difference=maximum,tolerance=1e-9,new_training=0,new_predictions=0,new_truth_scoring=0)
    (root/'report_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);a=p.parse_args();verify(a.root)
