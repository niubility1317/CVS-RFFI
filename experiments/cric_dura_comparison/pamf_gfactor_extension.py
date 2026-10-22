"""Append51 missing frozen PL/G-L readers, preserving every existing job."""
from pathlib import Path
from .io import read,write
from .pamf_gfactor import GFACTOR,ANALYSIS,config,validate_cells


def prepare(rt,jobs,previous,active):
    cells=validate_cells(rt['pamf_gfactor_manifest']['cells']);new=[];records=[]
    keys={j['id'] for j in jobs};folder=Path(rt['control_root'])/'gfactor_configs'
    def add(key,kind,c,completion,deps,status):
        if key in keys:raise ValueError('Duplicate PL Gfactor job')
        path=folder/(key.replace('/','__')+'.json')
        write(path,dict(c,stage=kind,frozen_sources=rt['frozen_sources'],code_commit=rt['code_commit']))
        j=dict(id=key,kind=kind,config=str(path),completion=str(completion),dependencies=list(deps),
            successful_statuses=[status],log_name=key.replace('/','__'),worker_release=rt['release_root'],execution_commit=rt['code_commit'])
        new.append(j);previous['jobs'][key]=dict(status='QUEUED');keys.add(key);return j
    for cell in cells:
        root=Path(rt['project'])/'runs'/GFACTOR/cell['row_id'];pc=config(cell);pc['output_root']=str(root/'predictions')
        pj=add(GFACTOR+'/'+cell['row_id']+'/predict','predict',pc,root/'predictions/complete.json',
            [cell['source_completion'],cell['raw_completion'],cell['cache_completion']],'ALL_PREDICTIONS_COMPLETE')
        sc=dict(run_id=GFACTOR,row_id=cell['row_id'],protocol_id=cell['protocol_id'],model_seed=cell['model_seed'],
            source_days=cell['source_days'],classes=cell['classes'],readouts=['raw','DURA-L'],tests=cell['tests'],
            prediction_output=pc['output_root'],output_root=str(root/'scores'))
        sj=add(GFACTOR+'/'+cell['row_id']+'/score','score',sc,root/'scores/complete.json',[pj['completion']],'SCORED_COMPLETE')
        sources=dict(cell['existing_scores']);pl_new=dict(run_id=GFACTOR,row_id=cell['row_id'],scores=str(root/'scores/scores.json'),completion=sj['completion'])
        records.append(dict(protocol_id=cell['protocol_id'],model_seed=cell['model_seed'],classes=cell['classes'],
            source_completion=cell['source_completion'],cache_config=cell['cache_config'],records={
            'Sup+G-L':dict(sources['matched_sat_supervised'],readout='DURA-L'),
            'Sup+G-LU':dict(sources['matched_sat_supervised'],readout='DURA-LU'),
            'PL+G-L':dict(pl_new,readout='DURA-L'),
            'PL+G-LU':dict(sources['matched_sat_plain_pl'],readout='DURA-LU')}))
    out=Path(rt['project'])/'runs'/ANALYSIS/'results'
    ac=dict(run_id=ANALYSIS,row_id='all51pairedcells',analysis_kind='pamf_gfactor',output_root=str(out),cells=records)
    dependencies=sorted({v['completion'] for r in records for v in r['records'].values()})
    add(ANALYSIS,'score',ac,out/'complete.json',dependencies,'PAMF_GFACTOR_ANALYZED')
    if len(new)!=103:raise ValueError('Expected51 predictions,51 scores and1 CPU analysis')
    previous['pamf_gfactor_extension']=dict(new_jobs=103,new_F=0,new_G=0,logical_PL_F_models=51,
        source_contracts=17,F_seeds=3,raw_reused=True,shared_existing_G_L_IQ=True,new_cache_jobs=0,
        healthy_PIDs_preserved=list(active),no_existing_job_or_output_changes=True)
    write(Path(rt['control_root'])/'pamf_gfactor_extension.json',previous['pamf_gfactor_extension'])
    write(Path(rt['project'])/'runs'/ANALYSIS/'manifest-at-launch.json',ac)
    return jobs+new
