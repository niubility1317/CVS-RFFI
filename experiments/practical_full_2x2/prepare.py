"""Resolve the one registered matrix to source-only jobs and blind readouts."""
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from .plan import RUN,NEW_RX,TASKS

def prepare(rt,p):
    from .data import prepare as data,old_G_selector
    handles=data(rt,p);root=Path(rt['run_root']);cfgroot=root/'configs';cfgroot.mkdir()
    oldjobs=read(rt['old_jobs']);old_predict={}
    for job in oldjobs:
        if job['kind']=='predict':
            c=read(job['config'])
            if c.get('run_id')=='20261021-phase1-T1-DURA-comparison-m153-r01':old_predict[c['row_id']]=c
    jobs=[];sources={};G={}
    def add(kind,row,c,completion,statuses,deps=()):
        c['stage']=kind
        c['external_releases']=rt['external_releases']
        path=cfgroot/(kind+'__'+row+'.json');write(path,c)
        jobs.append(dict(id=kind+'__'+row,kind=kind,config=str(path),completion=str(completion),
            successful_statuses=list(statuses),dependencies=list(deps),log_name=kind+'__'+row,
            python=rt['external_python'] if c.get('kind')=='external' and kind=='F' else rt['python']))
    for row in p['G_rows']:
        pid,role=row['protocol_id'],row['roles'];h=handles[pid];out=root/'G'/row['row_id']/'source'
        selector=old_G_selector(rt,h,pid,role,root/'g_selectors')
        c=dict(row,stage='G',run_id=RUN,output_root=str(out),cache=selector,
            code_commit=rt['code_commit'],frozen_sources=rt['frozen_sources'],
            training_acceleration=dict(teacher_cache='epoch',storage='auto',device_limit_mib=64,max_mib=512,
                frozen_prefix=True,skip_zero_spectral=True,reduce_syncs=True))
        add('G',row['row_id'],c,out/'complete.json',['SOURCE_FROZEN'])
        G[pid,role]=dict(kind='contract_matched',roles=role,source_contract=h['source_contract'],checkpoint=str(out/'r69/final.pt'))
    for pid in p['protocols']:
        if pid in NEW_RX:continue
        for role in ('L','LU'):
            spec=deepcopy(rt['G_templates'][pid]['readouts']['DURA-'+role])
            actual=read(spec.get('source_contract',rt['G_templates'][pid]['source_contract']));expected=read(handles[pid]['source_contract'])
            if any(actual[k]!=expected[k] for k in ('role_ids','counts','source_rxs','source_days','classes')):raise ValueError('Historical G contract mismatch')
            spec['source_contract']=handles[pid]['source_contract'];G[pid,role]=spec
    for row in p['rows']:
        pid=row['protocol_id'];h=handles[pid];out=root/'F'/row['row_id']/'source';source=deepcopy(row)
        source.update(run_id=RUN,output_root=str(out),code_commit=rt['code_commit'],
            frozen_sources=rt['frozen_sources'],source_rxs=p['protocols'][pid]['source_rxs'],source_days=[1,2,3],
            source_cache=h['source_cache'],source_contract=h['source_contract'],num_classes=len(h['classes']),
            augmentation_cache=rt['augmentation_cache'])
        if row['kind']=='external':
            e=h['external'][row['training_regime']]
            source.update(source_cache=e['source_cache'],source_contract=e['source_contract'],
                source_contract_path=e['source_contract'],num_domains=len(p['protocols'][pid]['source_rxs']),classes=h['classes'],
                ibrff_model_root=rt['ibrff_model_root'],sdg_model_root=rt['sdg_model_root'])
            expected='SOURCE_TRAINED_FROZEN';completion=out/'freeze.json'
        else:
            source.setdefault('ablation',None)
            expected='SOURCE_FROZEN';completion=out/'complete.json'
        add('F',row['row_id'],source,completion,[expected]);sources[row['row_id']]=dict(source_output=str(out),source_config=source,source_completion=str(completion))
    for row in p['reused_rows']:
        pid=row['protocol_id'];oldkey=f"{pid}-native_cric-s{row['model_seed']}";c=old_predict[oldkey]
        if c.get('original_arm'):
            sources[row['row_id']]=dict(source_output=None,original_arm='T1',source_completion=None)
        else:
            out=c['source_output'];done=read(Path(out)/'complete.json')
            if done['status']!='SOURCE_FROZEN' or read(done['config']['source_contract'])!=read(handles[pid]['source_contract']):raise ValueError('Historical T1 contract mismatch')
            sources[row['row_id']]=dict(source_output=out,source_completion=str(Path(out)/'complete.json'))
    caches={}
    for pid,h in handles.items():
        for role in ('L','LU'):
            row=f'{pid}-DURA-{role}';out=root/'restored_IQ'/row;spec=G[pid,role]
            c=dict(run_id=RUN,row_id=row,protocol_id=pid,stage='cache',G=spec,readout='DURA-'+role,
                source_contract=h['source_contract'],source_cache=h['source_cache'],output_root=str(out),
                tests={t:dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
                frozen_sources=rt['frozen_sources'],code_commit=rt['code_commit'])
            deps=[str(Path(spec['checkpoint']).parent.parent/'complete.json')] if pid in NEW_RX else []
            add('cache',row,c,out/'complete.json',['DURA_IQ_CACHE_COMPLETE'],deps);caches[pid,role]=str(out)
    for row in p['rows']+p['reused_rows']:
        pid=row['protocol_id'];h=handles[pid];src=sources[row['row_id']];out=root/'readouts'/row['row_id'];role=row['G_roles']
        c=dict(row,stage='predict',run_id=RUN,source_output=src['source_output'],source_contract=h['source_contract'],
            source_cache=h['source_cache'],classes=h['classes'],source_days=[1,2,3],
            readouts=['raw','DURA-'+role],restored_cache=caches[pid,role],output_root=str(out/'predictions'),
            tests={t:dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
            frozen_sources=rt['frozen_sources'],code_commit=rt['code_commit'])
        if row['kind']=='external':c['external_source_contract']=src['source_config']['source_contract']
        if src.get('original_arm'):c['original_arm']='T1'
        deps=[str(Path(caches[pid,role])/'complete.json')]
        if src['source_completion']:deps.append(src['source_completion'])
        add('predict',row['row_id'],c,out/'predictions/complete.json',['ALL_PREDICTIONS_COMPLETE'],deps)
        score=dict(c,stage='score',prediction_output=str(out/'predictions'),output_root=str(out/'scores'),tests=h['tests'])
        # CPU scoring never receives models, datasets, or training cache handles.
        for k in ('source_output','restored_cache','source_cache','external_source_contract'):score.pop(k,None)
        add('score',row['row_id'],score,out/'scores/complete.json',['SCORED_COMPLETE'],[str(out/'predictions/complete.json')])
    write(root/'jobs.json',jobs);write(root/'resolved_sources.json',dict(F=sources,G={pid+'-'+role:s for (pid,role),s in G.items()}))
    write(root/'preparation_complete.json',dict(status='PREPARED',new_F=600,new_G=4,reused_F=27,
        predictions=627,scores=627,restored_IQ_caches=22,jobs=len(jobs),target_score_reads=0))
    return jobs
