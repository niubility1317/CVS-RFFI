"""Strict CE architecture controls:24 new F and3 matched reused X3 baselines."""
import argparse
from collections import defaultdict
import json,math,statistics
from pathlib import Path
from .analyze_main import SEEDS,VIEWS,metrics,moments,read,write_csv

ARMS=('X3_CE','SIG_TIME','NO_TIME','NO_SIGNATURE','NO_PA','NO_REFERENCE','COV_X3','SHUFFLE_TIME','WRONG_REFERENCE')
READERS=('raw','DURA-L')


def aggregate(folder):
    folder=Path(folder);manifest=read(folder/'manifest.json')
    items=manifest['structure_rows']+manifest['structure_baselines']
    if len(items)!=27:raise ValueError('Expected24 new and3 matched reused architecture rows')
    expected={(a,s) for a in ARMS for s in SEEDS};seen=set();values={};counts={};nets=[];cost=[]
    geometry=defaultdict(dict);sensitivity=defaultdict(dict);ids=None;g=None;cm_count=0
    for item in items:
        rec=read(folder/item['member']);arm,seed=item['architecture'],item['seed'];identity=arm,seed
        if identity not in expected or identity in seen:raise ValueError('Architecture or seed differs')
        seen.add(identity);c=rec['source_config'];pc=rec['prediction_config'];s=rec['score'];done=rec['complete']
        if (c['ablation']!='F15' or c['protocol_id']!='F0' or c['model_seed']!=seed or c['num_classes']!=6
                or c.get('source_IQ_binding_fix')!='physical_ID_index_v1' or c['posthoc_head_fit'] or not c['default_head_joint']
                or pc['strict_supervised_system'] is not True or pc['readouts']['DURA-L']['roles']!='L'):
            raise ValueError('Strict same-contract CE/default-head ancestry differs')
        if arm!='X3_CE' and c.get('supervised_architecture')!=arm:raise ValueError('Wrong architecture config')
        if g is None:g=pc['readouts']['DURA-L']
        elif pc['readouts']['DURA-L']!=g:raise ValueError('G-L changed between architectures')
        if (s['model_seed']!=seed or s['status']!='SCORED' or not s['truth_last'] or not s['target_feedback_forbidden']
                or done['status']!='SCORED_COMPLETE' or done['views']!=list(VIEWS) or set(done['readouts'])!=set(READERS)
                or rec['prediction_complete']['truth_read'] is not False or rec['prediction_complete']['query_fit'] is not False):
            raise ValueError('Incomplete frozen seven-view independent score')
        local={}
        for r in s['results']:
            k=r['readout'],r['test'],r['view'],r['dimension'],str(r['stratum'])
            if k in local or k[0] not in READERS or k[1]!='main' or k[2] not in VIEWS:raise ValueError('Duplicate/undeclared score')
            local[k]=r;values[(*identity,*k)]=metrics(r,6);counts[(*identity,*k)]=r['count'];cm_count+=1
        raw={k[1:]:r for k,r in local.items() if k[0]=='raw'}
        if ({k[1:] for k in local if k[0]=='DURA-L'}!=set(raw)
                or set(raw)!={(t,v,d,z) for t,_,d,z in raw for v in VIEWS}
                or raw['main','clean','overall','ALL']['count']!=168000):raise ValueError('F0 view/strata coverage differs')
        pp={(r['readout'],r['test'],r['view'],r['dimension'],str(r['stratum'])):r for r in s['paired_raw']}
        if len(pp)!=len(s['paired_raw']) or set(pp)!={k for k in local if k[0]=='DURA-L'}:raise ValueError('Correction coverage differs')
        for k,p in pp.items():
            rr,gr=raw[k[1:]],local[k];n=rr['count'];nc=sum(rr['confusion'][i][i] for i in range(6))
            net=sum(gr['confusion'][i][i]-rr['confusion'][i][i] for i in range(6))
            if (gr['count']!=n or p['count']!=n or p['net']!=net or p['repairs']-p['damage']!=net
                    or min(p['repairs'],p['damage'])<0 or p['repairs']>n-nc or p['damage']>nc):raise ValueError('CM net/joint corrections differ')
            nets.append(dict(architecture=arm,seed=seed,**dict(zip(('readout','test','view','dimension','stratum'),k)),
                             count=n,repairs=p['repairs'],damage=p['damage'],net=net))
        meta=rec['source_metadata'];source=meta['complete.json'];stages=source['segments']
        if (source['epoch']!=50 or source['updates']!=11100 or source['target_access'] or source['head_fit_updates']
                or len(stages)!=3 or [p['epoch'] for p in stages]!=[20,10,20]
                or [p['updates'] for p in stages]!=[4440,2220,4440]):raise ValueError('Source budget differs')
        prog=[meta['segment%d/progress.json'%n] for n in (20,30,50)]
        if any(p['U_actual'] or p['PL_seen'] or p['PL_accepted'] for p in prog):raise ValueError('Strict CE consumed U')
        exposure=sum(p['L_actual']+p['extra_L'] for p in prog)
        seconds=sum(p['seconds'] for p in stages)
        if exposure!=11100*384 or abs(seconds-source['seconds'])>1e-8:raise ValueError('Actual CE exposure or elapsed differs')
        cost.append(dict(architecture=arm,seed=seed,new_F=arm!='X3_CE',model_epochs=50,updates=11100,
                         shared_worker_seconds=seconds,L_packet_forward_evaluations=exposure,U_IQ_exposures=0))
        d=rec['source_physics']
        if (d['samples']!=720 or d['groups']!=90 or d['source_U_reads'] or d['target_reads'] or d['fitted_probe_updates']
                or len(d['physical_ids'])!=720 or len(set(d['physical_ids']))!=720 or len(d['interventions'])!=11):
            raise ValueError('Source-L physics permissions/coverage differ')
        if ids is None:ids=d['physical_ids']
        elif d['physical_ids']!=ids:raise ValueError('Physics source samples are not paired')
        for name in ('same_TX_cross_RX_distance','different_TX_same_RX_distance','separation_margin'):
            geometry[arm,name][seed]=d['geometry'][name]
        for q in d['interventions']:
            for name in ('feature_cosine_distance','default_head_flip_fraction','reference_descriptor_relative_L2'):
                if name in q:
                    if not math.isfinite(q[name]):raise ValueError('Non-finite physics sensitivity')
                    sensitivity[arm,q['intervention'],name][seed]=q[name]
    if seen!=expected:raise ValueError('Missing architecture seed')
    base={k[3:]:n for k,n in counts.items() if k[:3]==('X3_CE',SEEDS[0],'raw')}
    for arm,seed in expected:
        for reader in READERS:
            if {k[3:]:n for k,n in counts.items() if k[:3]==(arm,seed,reader)}!=base:raise ValueError('Cross-architecture physical strata differ')
    for k in list(values):
        if k[4]=='clean':
            six=[values[(*k[:4],v,*k[5:])] for v in VIEWS[1:]]
            values[(*k[:4],'six_practical_mean',*k[5:])]={m:statistics.mean(r[m] for r in six) for m in ('accuracy','macro_f1')}
    per=[];group=defaultdict(dict)
    for k,r in sorted(values.items()):
        arm,seed,reader,test,view,dim,z=k
        for metric,value in r.items():
            per.append(dict(architecture=arm,seed=seed,readout=reader,test=test,view=view,dimension=dim,stratum=z,metric=metric,percent=100*value))
            group[arm,reader,test,view,dim,z,metric][seed]=100*value
    fields=('architecture','readout','test','view','dimension','stratum','metric')
    summary=[dict(zip(fields,k),**moments([v[s] for s in SEEDS])) for k,v in sorted(group.items())]
    conditions=sorted({k[3:] for k in values if k[:3]==('X3_CE',SEEDS[0],'raw')});paired=[]
    for condition in conditions:
        c=dict(zip(('test','view','dimension','stratum'),condition))
        for metric in ('accuracy','macro_f1'):
            for arm in ARMS[1:]:
                for reader in READERS:
                    v=[100*(values[('X3_CE',s,reader,*condition)][metric]-values[(arm,s,reader,*condition)][metric]) for s in SEEDS]
                    paired.append(dict(contrast='X3_CE_minus_variant',architecture=arm,readout=reader,**c,metric=metric,**moments(v)))
            for arm in ARMS:
                v=[100*(values[(arm,s,'DURA-L',*condition)][metric]-values[(arm,s,'raw',*condition)][metric]) for s in SEEDS]
                paired.append(dict(contrast='G_L_minus_raw',architecture=arm,readout='paired',**c,metric=metric,**moments(v)))
    ph=[dict(architecture=a,source_metric=m,**moments([v[s] for s in SEEDS])) for (a,m),v in sorted(geometry.items())]
    sens=[dict(architecture=a,intervention=i,source_metric=m,**moments([v[s] for s in SEEDS])) for (a,i,m),v in sorted(sensitivity.items())]
    data=dict(per_seed=per,summary=summary,paired=paired,net_corrections=nets,F_cost=cost,source_geometry=ph,source_sensitivity=sens)
    proof=dict(status='ANALYZED_STRICT_STRUCTURE24',new_F=24,reused_X3_CE_F=3,actual_F_seeds=list(SEEDS),
        serialized_CM_recomputed=cm_count,net_correction_strata=len(nets),new_source_stages=72,new_model_epochs=1200,new_updates=266400,
        all_L_only=True,U_IQ_exposures=0,source_physics_panels=27,source_L_samples_per_panel=720,source_groups=90,source_interventions=11,
        source_fitted_probe_updates=0,conditional_on_one_G_L=True,target_truth_reads_by_analysis=0,target_feedback_used=False,
        new_F_shared_worker_seconds=sum(r['shared_worker_seconds'] for r in cost if r['new_F']),
        reused_X3_CE_shared_worker_seconds=sum(r['shared_worker_seconds'] for r in cost if not r['new_F']))
    proof['WRONG_REFERENCE_seed01_technical_recovery_used']=any(read(folder/p['member'])['source_config']['run_id']=='20261023-phase1-PAMF-physics-device-recovery-r01' for p in manifest['structure_rows'])
    return data,proof


def run(folder,output):
    data,proof=aggregate(folder);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for name,rows in data.items():write_csv(out/('structure_'+name+'.csv'),rows)
    for name in ('summary','paired'):write_csv(out/('structure_'+name+'_overall.csv'),[r for r in data[name] if r['dimension']=='overall'])
    (out/'structure_verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');print(json.dumps(proof))
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.input,a.output)
