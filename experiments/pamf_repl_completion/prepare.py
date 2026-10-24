"""Bind the approved matrix to existing data/G and immutable new F outputs."""
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from .plan import RUN,configuration,validate

def prepare(rt):
    context=read(rt['context']);matrix=read(rt['matrix'])['matrix'];validate(matrix)
    root=Path(rt['run_root']);cfgroot=root/'configs';cfgroot.mkdir();jobs=[];sources={};prefixes={}
    def add(kind,rid,c,completion,status,deps=(),**extra):
        path=cfgroot/(kind+'__'+rid+'.json');write(path,c)
        jobs.append(dict(id=kind+'__'+rid,kind=kind,config=str(path),completion=str(completion),
            successful_statuses=[status],dependencies=list(deps),log_name=kind+'__'+rid,**extra))
    for row in matrix['rows']:
        seed=str(row['model_seed']);rid=row['row_id'];c=configuration(context,row,rt['project'],rt['code_commit'])
        c['frozen_sources']=rt['frozen_sources']
        if row['reuse']:
            key={'supervised':'F20','ordinary_PL':'F21','REPL':'REPL','without_extrapolation':'NO_RXE'}[row['method']]
            source=context['sources'][seed][key]
            done=read(Path(source)/'complete.json')
            if read(done['config']['source_contract'])!=read(context['handles'][row['protocol_id']]['source_contract']):raise ValueError('Reused F contract differs')
            sources[rid]=dict(source_output=source,config=done['config'],reused=True)
            continue
        if row['U']:
            if row['group']=='mechanism':
                c['shared_prefix']=context['sources'][seed]['F21']+'/segment20'
            else:
                key=f"{row['protocol_id']}-A{int(row['L_A'])}B{int(row['L_B'])}-{row['structure']}-{c['ablation']}-s{seed}"
                if key not in prefixes:
                    folder=root/'prefixes'/key/'segment20';p=deepcopy(c);p['output_root']=str(folder)
                    add('prefix',key,p,folder/'complete.json','SOURCE_FROZEN')
                    prefixes[key]=str(folder)
                c['shared_prefix']=prefixes[key]
        deps=[str(Path(c['shared_prefix'])/'complete.json')] if c.get('shared_prefix') else []
        add('F',rid,c,Path(c['output_root'])/'complete.json','SOURCE_FROZEN',deps)
        sources[rid]=dict(source_output=c['output_root'],config=c,reused=False)
    if len(prefixes)!=84 or sum(j['kind']=='F' for j in jobs)!=189:raise ValueError('Unique-prefix/model counts differ')
    for row in matrix['rows']:
        rid=row['row_id'];pid=row['protocol_id'];h=context['handles'][pid];src=sources[rid]
        out=root/'readouts'/rid
        c=dict(row,run_id=RUN,source_output=src['source_output'],source_contract=h['source_contract'],source_cache=h['source_cache'],source_days=[1,2,3],
            classes=h['classes'],tests={t:dict(views_root=v['views_root']) for t,v in h['tests'].items()},
            restored_caches={k:v['path'] for k,v in context['caches'][pid].items()},readouts=['raw','DURA-L','DURA-LU'],
            output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'],source_physics_diagnostics=pid=='F0',code_commit=rt['code_commit'])
        add('predict',rid,c,out/'predictions/complete.json','ALL_PREDICTIONS_COMPLETE',[str(Path(src['source_output'])/'complete.json')])
        s=deepcopy(c);s.update(prediction_output=c['output_root'],output_root=str(out/'scores'),tests=h['tests'])
        for k in ('source_output','source_cache','restored_caches','frozen_sources'):s.pop(k,None)
        add('score',rid,s,out/'scores/complete.json','SCORED_COMPLETE',[str(out/'predictions/complete.json')])
    for item in context['G_crossed']:
        for seed in matrix['model_seeds']:
            h=context['handles']['m3-natural'];source=sources[f'm3-natural-REPL-s{seed}']['source_output']
            rid=f'm3-natural-Fs{seed}-Gs{item["seed"]}';out=root/'crossed_readouts'/rid
            c=dict(row_id=rid,run_id=RUN,protocol_id='m3-natural',method='REPL_Gseed'+str(item['seed']),group='G_crossed',model_seed=seed,G_seed=item['seed'],G_component=True,
                source_output=source,source_run_id=RUN,source_row_id=f'm3-natural-REPL-s{seed}',source_days=[1,2,3],
                source_contract=h['source_contract'],source_cache=h['source_cache'],classes=h['classes'],readouts={'raw':None,'DURA-LU':item['spec']},
                tests={t:dict(views_root=v['views_root']) for t,v in h['tests'].items()},output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'],source_physics_diagnostics=False)
            add('predict',rid,c,out/'predictions/complete.json','ALL_PREDICTIONS_COMPLETE',[str(Path(source)/'complete.json')],heavy_G=True)
            s=deepcopy(c);s.update(prediction_output=c['output_root'],output_root=str(out/'scores'),tests=h['tests'],readouts=list(c['readouts']))
            for k in ('source_output','source_cache','G_component','frozen_sources'):s.pop(k,None)
            add('score',rid,s,out/'scores/complete.json','SCORED_COMPLETE',[str(out/'predictions/complete.json')])
    for item in context['G_components']:
        seed=str(item['seed']);h=context['handles']['F0'];source=context['sources'][seed]['REPL'];done=read(Path(source)/'complete.json')
        rid='G-'+item['arm']+'-s'+seed;out=root/'G_readouts'/rid
        c=dict(row_id=rid,run_id=RUN,protocol_id='F0',method='G_'+item['arm'],group='G_component',model_seed=int(seed),G_component=True,
            source_output=source,source_run_id=done['config']['run_id'],source_row_id=done['config']['row_id'],
            source_contract=h['source_contract'],source_cache=h['source_cache'],source_days=[1,2,3],classes=h['classes'],
            readouts={'raw':None,item['arm']:item['spec']},tests={t:dict(views_root=v['views_root']) for t,v in h['tests'].items()},
            output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'],source_physics_diagnostics=False)
        add('predict',rid,c,out/'predictions/complete.json','ALL_PREDICTIONS_COMPLETE',[],heavy_G=True)
        s=deepcopy(c);s.update(prediction_output=c['output_root'],output_root=str(out/'scores'),tests=h['tests'],readouts=list(c['readouts']))
        for k in ('source_output','source_cache','G_component','frozen_sources'):s.pop(k,None)
        add('score',rid,s,out/'scores/complete.json','SCORED_COMPLETE',[str(out/'predictions/complete.json')])
    # Dedicated timing is admitted only to a naturally empty GPU, after all input F freeze.
    for panel,rounds in matrix['dedicated_timing_case_rounds'].items():
        c=dict(run_id=RUN,panel=panel,output_root=str(root/'timing'/panel),context=rt['context'],sources=str(root/'sources.json'),
            source_contract=context['handles']['F0']['source_contract'],source_cache=context['handles']['F0']['source_cache'],
            frozen_sources=rt['frozen_sources'],case_rounds=rounds)
        methods=([('REPL' if a=='full' else 'REPL_'+a) for a in matrix['structure_arms']]
            if panel=='REPL_structure' else ['CE_'+a for a in matrix['structure_arms']]
            if panel=='strict_structure' else ['supervised','ordinary_PL'] if panel=='main_controls' else ['REPL'])
        deps=[str(Path(sources['F0-'+m+'-s2026092701']['source_output'])/'complete.json') for m in methods]
        add('timing',panel,c,root/'timing'/panel/'complete.json','TIMING_COMPLETE',deps,exclusive=True)
    write(root/'sources.json',sources);write(root/'jobs.json',jobs)
    write(root/'preparation_complete.json',dict(status='PREPARED',models=201,new_F=189,reused_F=12,prefix_jobs=84,
        F_jobs=189,prediction_jobs=228,score_jobs=228,timing_jobs=4,new_G=0,new_IQ_cache=0,jobs=len(jobs)))
    return jobs
