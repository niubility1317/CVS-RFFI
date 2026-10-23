"""Seed-first six-view external summaries, paired effects and reused costs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from .analyze_main import SEEDS,VIEWS,metrics,write_csv,moments
from .reproduce_final import PortableInput


def run(folder,output):
    src=PortableInput(folder);m=src.read('manifest.json');c=src.read(m['analysis_configs']['external'])
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    values={};nets=[];costs=[];cm_count=0;source_stages=0
    for item in c['rows']:
        pid,fam,var,reg,seed=(item[k] for k in ('protocol_id','family','variant','training_regime','model_seed'))
        score=src.read(item['scores']);local={};counts={};correct={}
        if not score['target_feedback_forbidden']:raise ValueError('Target feedback forbidden flag absent')
        for r in score['results']:
            checked=metrics(r,10 if pid=='ManyRx' else 6);cm_count+=1
            key=(r['test'],r['readout'],r['view'],r['dimension'],str(r['stratum']))
            if key in local:raise ValueError('Duplicate external stratum')
            local[key]=checked;counts[key]=r['count'];correct[key]=sum(r['confusion'][i][i] for i in range(len(r['confusion'])))
        tests={'main','source_unseen_date'} if pid=='chrono' else {'main'}
        for test in tests:
            base={(k[3],k[4]):n for k,n in counts.items() if k[:3]==(test,'raw','clean')}
            for reader in ('raw','DURA-L','DURA-LU'):
                for view in VIEWS:
                    if {(k[3],k[4]):n for k,n in counts.items() if k[:3]==(test,reader,view)}!=base:
                        raise ValueError('External physical strata/counts differ across views/readouts')
            expected_n=30000 if test=='source_unseen_date' else (144000 if pid.startswith('m') else 123021 if pid=='ManyRx' else 168000)
            if base['overall','ALL']!=expected_n:raise ValueError('External target set differs')
        for key in list(local):
            if key[2]!='clean':continue
            six=[local[(*key[:2],v,*key[3:])] for v in VIEWS[1:]]
            local[(*key[:2],'six_practical_mean',*key[3:])]={k:statistics.mean(x[k] for x in six) for k in ('accuracy','macro_f1')}
        for key,v in local.items():
            if key[2]=='six_practical_mean' or key[3]=='overall':
                for metric,x in v.items():
                    full=(pid,fam,var,reg,seed,*key,metric)
                    if full in values:raise ValueError('Duplicate external F seed')
                    values[full]=100*x
        for p in score['paired_raw']:
            key=(p['test'],'raw',p['view'],p['dimension'],str(p['stratum']))
            if p['repairs']<0 or p['damage']<0 or p['repairs']>p['count']-correct[key] or p['damage']>correct[key]:
                raise ValueError('Invalid correction count')
            if p['dimension']=='overall':
                nets.append(dict(protocol=pid,family=fam,variant=var,regime=reg,seed=seed,**p))
        f=src.read(Path(item['source_output'])/'freeze.json');stages=f['stages'];source_stages+=len(stages)
        if f['target_access'] or f['target_score_used'] or not stages:raise ValueError('Invalid source freeze')
        costs.append(dict(protocol=pid,family=fam,variant=var,regime=reg,seed=seed,
            source_shared_worker_seconds=f['seconds'],stage_seconds=sum(x['seconds'] for x in stages),stages=len(stages),
            epochs=sum(x['epochs'] for x in stages),optimizer_steps=sum(x['optimizer_steps'] for x in stages),
            labeled_exposure=sum(x['labeled_exposure'] for x in stages),satellite_examples=f['satellite_examples'],
            satellite_applied_examples=f['satellite_applied_examples'],PL_seen=f['pl_seen'],PL_accepted=f['pl_accepted'],
            new_F_training_seconds=0,new_G_training_seconds=0))
    keys=('protocol','family','variant','regime','seed','task','readout','view','dimension','stratum','metric')
    per=[dict(zip(keys,k),percent=v) for k,v in sorted(values.items())]
    grouped=defaultdict(dict)
    for k,v in values.items():grouped[k[:4]+k[5:]][k[4]]=v
    summary=[]
    for k,v in sorted(grouped.items()):
        if set(v)!=set(SEEDS):raise ValueError('Missing three F seeds')
        summary.append(dict(zip(keys[:4]+keys[5:],k),**moments([v[s] for s in SEEDS])))
    contrasts=defaultdict(dict)
    for k,v in values.items():
        pid,fam,var,reg,seed,task,reader,view,dim,z,metric=k
        tail=(task,reader,view,dim,z,metric)
        if var=='full':
            other=(pid,fam,'ce',reg,seed,*tail)
            if other not in values:raise ValueError('Full has no matched CE')
            contrasts[('Full_minus_CE',pid,fam,'paired',reg,*tail)][seed]=v-values[other]
        if reg=='pl':
            other=(pid,fam,var,'supervised',seed,*tail)
            if other not in values:raise ValueError('PL has no matched Sup')
            contrasts[('PL_minus_Sup',pid,fam,var,'paired',*tail)][seed]=v-values[other]
        if reader=='DURA-LU':
            other=(pid,fam,var,reg,seed,task,'DURA-L',view,dim,z,metric)
            contrasts[('LU_minus_L',pid,fam,var,reg,task,'paired',view,dim,z,metric)][seed]=v-values[other]
        if reader!='raw':
            other=(pid,fam,var,reg,seed,task,'raw',view,dim,z,metric)
            contrasts[('G_minus_raw',pid,fam,var,reg,*tail)][seed]=v-values[other]
    paired=[]
    effectkeys=('contrast','protocol','family','variant','regime','task','readout','view','dimension','stratum','metric')
    for k,v in sorted(contrasts.items()):
        if set(v)!=set(SEEDS):raise ValueError('Missing paired F seed')
        paired.append(dict(zip(effectkeys,k),**moments([v[s] for s in SEEDS])))
    caches=[]
    for row in m['cache_rows']:
        cfg=src.read(row['config']);d=src.read(row['completion'])
        if (d['status']!='DURA_IQ_CACHE_COMPLETE' or d['truth_read'] or d['F_reads'] or d['model_updates']
                or d['dtype']!='float32' or d['extra_normalization']):
            raise ValueError('Shared IQ cache contract differs')
        caches.append(dict(protocol=cfg['protocol_id'],readout=cfg['readout'],seconds=d['seconds']))
    for name,rows in (('per_seed',per),('summary',summary),('paired',paired),('net_overall',nets),('source_cost',costs),('cache_cost',caches)):
        write_csv(out/('external_final_'+name+'.csv'),rows)
    for name,rows in (('summary',summary),('paired',paired)):
        write_csv(out/('external_final_'+name+'_overall.csv'),[r for r in rows if r['dimension']=='overall'])
    proof=dict(status='VERIFIED_EXTERNAL516_SEED_FIRST',models=516,serialized_CM_recomputed=cm_count,source_stages=source_stages,
        frozen_IQ_caches=len(caches),cache_seconds=sum(r['seconds'] for r in caches),
        reused_F_shared_worker_seconds=sum(r['source_shared_worker_seconds'] for r in costs),new_F=0,new_G=0,
        six_view_mean_within_seed_first=True,SD_not_significance=True,ManyRx_classes=10,other_contract_classes=6,
        target_feedback_used=False,negative_results_retained=True)
    (out/'external_final_verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(proof));return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.input,a.output)
