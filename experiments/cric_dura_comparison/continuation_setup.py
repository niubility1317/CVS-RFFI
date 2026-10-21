"""Prepare source-only G selectors and fixed, result-independent continuation jobs."""
from pathlib import Path
import os
import subprocess
from .io import read,write
from .g_cache import prepare_index

G_RUN='20261021-phase1-CRIC-DURA-G-lineage-m35-r01'
STUDENT_RUN='20261021-phase1-CRIC-DURA-G-ablation-m21-r01'
TIMING_RUN='20261021-phase1-CRIC-DURA-latency-r01'


def prepare(rt):
    from .g_ablation import ARMS
    plan=read(rt['matrix']);source=read(rt['source_runtime'])
    handles=read(source['reused_source_handles'])
    frows=read(Path(source['run_root'])/'F_source_rows.json')
    cache=prepare_index(rt,handles,plan['protocols'])
    configs=Path(rt['run_root'])/'configs';configs.mkdir(parents=True,exist_ok=False)
    common=dict(code_commit=rt['code_commit'],frozen_sources=rt['frozen_sources'])
    jobs=[]
    def add(key,kind,c,completion,statuses,dependencies=(),cwd=None,exclusive=False):
        name=key.replace('/','__');path=configs/(name+'.json')
        if kind!='F':
            c=dict(c,stage=kind,**common);write(path,c)
        else:path=Path(c)
        jobs.append(dict(id=key,kind=kind,config=str(path),completion=str(completion),successful_statuses=list(statuses),
            dependencies=list(map(str,dependencies)),log_name=name,exclusive=exclusive))
    for c in frows:
        cfg=Path(source['run_root'])/'F_source_configs'/(c['run_id']+'__'+c['row_id']+'.json')
        add(c['run_id']+'/'+c['row_id'],'F',str(cfg),Path(c['output_root'])/'complete.json',['SOURCE_FROZEN'])
    g_specs={};g_done={}
    original=dict(kind='original_F0_LU',roles='LU',source_contract=handles['F0']['source_contract'],checkpoint=rt['original_G_checkpoint'])
    for row in plan['G_training_pipelines']:
        pid=row['protocol_id'];role='L' if row.get('source_roles')=='L_s' else 'LU';seed=row['model_seed']
        key=(pid,role,seed)
        if row['status']=='EXACT_CONTRACT_FROZEN_REUSE':
            g_specs[key]=original;g_done[key]=[];continue
        root=Path(rt['project'])/'runs'/G_RUN/row['job_id']/'source'
        c=dict(run_id=G_RUN,row_id=row['job_id'],model_seed=seed,output_root=str(root),cache=cache[pid+'-'+role])
        dependencies=[] if not cache[pid+'-'+role]['new_observations'] else [Path(cache[pid+'-'+role]['new_cache'])/'complete.json']
        add(G_RUN+'/'+row['job_id'],'G',c,root/'complete.json',['SOURCE_FROZEN'],dependencies)
        g_specs[key]=dict(kind='contract_matched',roles=role,checkpoint=str(root/'r69/final.pt'))
        g_done[key]=[root/'complete.json']
    def evaluation(key,run,row,pid,seed,source_output,readouts,dependencies,original_arm=None,source_row_id=None):
        base=Path(rt['project'])/'runs'/run/row
        pred=base/'predictions-control-r01';scores=base/'scores-control-r01'
        c=dict(run_id=run,row_id=row,protocol_id=pid,model_seed=seed,source_output=source_output,
            source_contract=handles[pid]['source_contract'],source_cache=handles[pid]['source_cache'],readouts=readouts,output_root=str(pred),
            tests={name:dict(views_root=h['views_root']) for name,h in handles[pid]['tests'].items()})
        if original_arm:c['original_arm']=original_arm
        if source_row_id:c['source_row_id']=source_row_id
        add(key+'/predict','predict',c,pred/'complete.json',['ALL_PREDICTIONS_COMPLETE'],dependencies)
        sc=dict(run_id=run,row_id=row,protocol_id=pid,model_seed=seed,classes=handles[pid]['classes'],
            source_days=plan['protocols'][pid]['source_days'],prediction_output=str(pred),output_root=str(scores),
            readouts=list(readouts),tests=handles[pid]['tests'])
        add(key+'/score','score',sc,scores/'complete.json',['SCORED_COMPLETE'],[pred/'complete.json'])
    for c in frows:
        pid=c['protocol_id'];seed=c['model_seed'];key=c['run_id']+'/'+c['row_id']
        roles=('LU',) if c['regime']!='matched_sat_supervised' else ('L','LU')
        readouts={'raw':None};dependencies=[Path(c['output_root'])/'complete.json']
        for role in roles:
            readouts['DURA-'+role]=g_specs[(pid,role,2026092701)];dependencies+=g_done[(pid,role,2026092701)]
        evaluation(key,c['run_id'],c['row_id'],pid,seed,c['output_root'],readouts,dependencies)
    for seed in (2026092701,2026092702,2026092703):
        for arm in ARMS:
            row=arm+'-s%d'%seed;root=Path(rt['project'])/'runs'/STUDENT_RUN/row/'source'
            c=dict(run_id=STUDENT_RUN,row_id=row,arm=arm,model_seed=seed,output_root=str(root),cache=cache['F0-LU'],
                teacher_checkpoint=rt['original_R65_checkpoint'],teacher_source_contract=handles['F0']['source_contract'])
            add(STUDENT_RUN+'/'+row,'G_student',c,root/'complete.json',['SOURCE_FROZEN'])
            variant=dict(kind='G_ablation',roles='LU',checkpoint=str(root/'final.pt'),arm=arm)
            evaluation(STUDENT_RUN+'/'+row,STUDENT_RUN,row,'F0',seed,None,{'raw':None,arm:variant},[root/'complete.json'],original_arm='CROSS_RX')
    # Existing BASE/ALL_TX models are frozen references, not extra F training.
    for arm in ('BASE','ALL_TX'):
        for seed in (2026092701,2026092702,2026092703):
            row='factorial-'+arm+'-s%d'%seed
            evaluation('factorial/'+row,rt['main_run'],row,'F0',seed,None,{'raw':None,'DURA-LU':original},[],original_arm=arm)
    # Additional m3 G seeds form the declared 3x3 F/G crossed robustness panel.
    for gseed in (2026092702,2026092703):
        for c in frows:
            if c['protocol_id']=='m3-natural' and c['regime']=='native_cric':
                row=c['row_id']+'-Gseed%d'%gseed
                evaluation('crossed/'+row,rt['main_run'],row,'m3-natural',c['model_seed'],c['output_root'],
                    {'raw':None,'DURA-Gseed%d'%gseed:g_specs[('m3-natural','LU',gseed)]},
                    [Path(c['output_root'])/'complete.json',*g_done[('m3-natural','LU',gseed)]],source_row_id=c['row_id'])
    timing_root=Path(rt['project'])/'runs'/TIMING_RUN
    external=[]
    for family in ('riei','drift','ibrff','sdg'):
        for variant in ('ce','full'):
            row='F0-'+family+'-'+variant+'-s2026092701'
            external.append(dict(name=family+'_'+variant,source_output=rt['external_run_root']+'/'+row+'/source'))
    timing=dict(run_id=TIMING_RUN,output_root=str(timing_root),source_contract=handles['F0']['source_contract'],
        source_cache=handles['F0']['source_cache'],G=original,external_release=rt['external_release'],external_models=external)
    add(TIMING_RUN,'timing',timing,timing_root/'complete.json',['TIMING_COMPLETE'],exclusive=True)
    write(rt['jobs'],jobs)
    write(Path(rt['run_root'])/'prepared.json',dict(status='CONTINUATION_PREPARED',jobs=len(jobs),
        counts={kind:sum(j['kind']==kind for j in jobs) for kind in ('F','G','G_student','predict','score','timing')},target_data_feedback=False))
    # Observation generation is CPU/source-only. It may run beside GPU jobs.
    for dataset in ('ManySig','ManyRx'):
        cfg=dict(cache_root=str(Path(rt['run_root'])/'g_data'/dataset),frozen_sources=rt['frozen_sources'])
        path=configs/('observations-'+dataset+'.json');write(path,cfg)
        cmd=[rt['python'],'-u','-m','experiments.cric_dura_comparison.continuation_setup','observations','--config',str(path)]
        with (Path(rt['run_root'])/('observations-'+dataset+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=rt['release_root'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(Path(rt['run_root'])/('observations-'+dataset+'-submit.json'),dict(pid=child.pid,argv=cmd,source_only=True))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','observations']);p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config)
    if a.mode=='prepare':prepare(c)
    else:
        from .g_cache import prepare_observations
        prepare_observations(c['cache_root'],c['frozen_sources'],workers=2)
