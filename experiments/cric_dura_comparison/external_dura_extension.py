"""Append frozen external readouts and the bounded source-diagnostic recovery."""
from copy import deepcopy
from pathlib import Path
from .io import read, write
from .external_dura import (CACHE,SUP,OTHER,ANALYSIS,RECOVERY,CORE_RECOVERY,READOUTS,
    model_config,validate_rows,compatible_sources)

BAD_CORE='20261022-phase1-PAMF-core-G-L-readouts-m9-r01'
OLD_ANALYSIS='20261022-phase1-PAMF-core-factorial-analysis-r01'
ERROR='RuntimeError: Could not infer dtype of numpy.int64'


def recovery_needed(state):
    failed={k:v.get('failure') for k,v in state['jobs'].items() if v['status']=='FAILED' and k not in state.get('excluded_job_ids',[])}
    bad={BAD_CORE+'/'+bits+'-s2026092701/predict' for bits in ('ABC101','ABC111')}
    if {k for k,v in failed.items() if v==ERROR} != bad:
        raise ValueError('Reconcile source-diagnostic failures before handover')
    if any(v not in (ERROR,"ModuleNotFoundError: No module named 'baselines.riei_fd'") for v in failed.values()):
        raise ValueError('Unresolved unrelated technical failure; no new owner')
    return bad


def prepare(rt, jobs, previous, active):
    manifest=rt['external_manifest'];validate_rows(manifest['rows'])
    by={j['id']:j for j in jobs};new=[];configs=Path(rt['control_root'])/'external_configs'
    old_configs={};records=[]
    def add(key,kind,c,completion,deps,status):
        path=configs/(key.replace('/','__')+'.json')
        write(path,dict(c,stage=kind,frozen_sources=rt['frozen_sources'],code_commit=rt['code_commit']))
        j=dict(id=key,kind=kind,config=str(path),completion=str(completion),dependencies=list(deps),
            successful_statuses=[status],log_name=key.replace('/','__'),worker_release=rt['release_root'],execution_commit=rt['code_commit'])
        if key in by:raise ValueError('Duplicate external/readout job')
        new.append(j);previous['jobs'][key]=dict(status='QUEUED');return j

    # Recover only the two identified failures. No source training is repeated.
    bad=recovery_needed(previous);replacements={};excluded=set(previous.get('excluded_job_ids',[]))
    for key in sorted(bad):
        row=key.split('/')[1];old=by[key];pc=read(old['config']);old_configs[key]=pc
        if key in active:raise ValueError('Never replace a healthy source diagnostic')
        folder=Path(rt['project'])/'runs'/RECOVERY/row
        pc.update(run_id=RECOVERY,output_root=str(folder/'predictions'))
        pj=add(RECOVERY+'/'+row+'/predict','predict',pc,folder/'predictions/complete.json',old['dependencies'],'ALL_PREDICTIONS_COMPLETE')
        sjold=by[key[:-7]+'score'];sc=read(sjold['config']);oldpath=str(Path(sc['output_root'])/'scores.json')
        sc.update(run_id=RECOVERY,prediction_output=pc['output_root'],output_root=str(folder/'scores'))
        sj=add(RECOVERY+'/'+row+'/score','score',sc,folder/'scores/complete.json',[pj['completion']],'SCORED_COMPLETE')
        replacements[oldpath]=str(folder/'scores/scores.json');replacements[sjold['completion']]=sj['completion']
        replacements[sjold['id']]=sj['id']
        excluded.update((key,sjold['id']))
    old_analysis=by[OLD_ANALYSIS];ac=read(old_analysis['config'])
    def replace(v):
        if isinstance(v,dict):
            result={k:replace(x) for k,x in v.items()}
            if result.get('scores') in replacements.values() and 'row_id' in result:result['run_id']=RECOVERY
            return result
        if isinstance(v,list):return [replace(x) for x in v]
        return replacements.get(v,v) if isinstance(v,str) else v
    ac=replace(ac);ac.update(run_id=CORE_RECOVERY,output_root=str(Path(rt['project'])/'runs'/CORE_RECOVERY/'results'))
    add(CORE_RECOVERY,'score',ac,Path(ac['output_root'])/'complete.json',replace(old_analysis['dependencies']),'CORE_ANALYZED')
    excluded.add(OLD_ANALYSIS)
    previous['excluded_job_ids']=sorted(excluded)
    previous.setdefault('resolved_technical_failures',[]).append(dict(fingerprint=ERROR,replaced_jobs=sorted(bad),
        replacement_run=RECOVERY,analysis_run=CORE_RECOVERY,old_outputs_preserved=True,no_F_G_retraining=True))
    previous['failures']=dict(previous.get('failures',{}))
    previous['failures'].pop(ERROR,None)
    # Change execution only for not-yet-started consumers of the exact fixed
    # NumPy/Torch selection operation; retain their science configs/output roots.
    migrated=[]
    for j in jobs:
        if j['id'] in excluded or j['id'] in active or previous['jobs'][j['id']]['status']!='QUEUED' or j['kind']!='predict':continue
        c=read(j['config'])
        if c.get('source_physics_diagnostics'):
            j.update(worker_release=rt['release_root'],execution_commit=rt['code_commit']);migrated.append(j['id'])

    cache_jobs={};cache_cfgs={}
    for pid,t in manifest['G_templates'].items():
        for name in READOUTS[1:]:
            row=pid+'-'+name;root=Path(rt['project'])/'runs'/CACHE/row
            c=dict(run_id=CACHE,row_id=row,protocol_id=pid,external_action='cache',readout=name,G=t['readouts'][name],
                source_contract=t['source_contract'],source_cache=t['source_cache'],
                tests={k:{'views_root':v['views_root']} for k,v in t['tests'].items()},output_root=str(root))
            spec=c['G'];cp=Path(spec['checkpoint']);dependency=cp.parents[1]/'complete.json' if spec['kind']=='contract_matched' else cp.parent/'complete.json'
            # Original fixed R69 has a different historical completion layout;
            # its immutable checkpoint already exists and is fully validated.
            deps=[str(dependency)] if spec['kind']=='contract_matched' else [str(cp)]
            j=add(CACHE+'/'+row,'G',c,root/'complete.json',deps,'DURA_IQ_CACHE_COMPLETE')
            cache_jobs[pid,name]=j;cache_cfgs[pid,name]=j['config']
    for item in manifest['rows']:
        pid=item['protocol_id'];t=manifest['G_templates'][pid]
        run=SUP if pid=='F0' and item['training_regime']=='supervised' else OTHER
        row=item['row_id'];root=Path(rt['project'])/'runs'/run/row
        compatible_sources(read(item['source_contract']),read(t['source_contract']),item['classes'])
        pc=model_config(item,t,rt)
        pc.update(run_id=run,row_id=row,external_action='predict',output_root=str(root/'predictions'),
            cache_configs={n:cache_cfgs[pid,n] for n in READOUTS[1:]},
            tests={k:{'views_root':v['views_root']} for k,v in item['tests'].items()})
        deps=[cache_jobs[pid,n]['completion'] for n in READOUTS[1:]]+[str(Path(item['prediction_output'])/'complete.json')]
        pj=add(run+'/'+row+'/predict','predict',pc,root/'predictions/complete.json',deps,'ALL_PREDICTIONS_COMPLETE')
        sc=dict(run_id=run,row_id=row,protocol_id=pid,model_seed=item['model_seed'],classes=item['classes'],source_days=pc['source_days'],
            prediction_output=pc['output_root'],output_root=str(root/'scores'),readouts=list(READOUTS),tests=item['tests'])
        sj=add(run+'/'+row+'/score','score',sc,root/'scores/complete.json',[pj['completion']],'SCORED_COMPLETE')
        records.append(dict(run_id=run,row_id=row,protocol_id=pid,family=item['family'],variant=item['variant'],
            training_regime=item['training_regime'],model_seed=item['model_seed'],scores=str(root/'scores/scores.json'),
            source_output=item['source_output'],completion=sj['completion']))
    out=Path(rt['project'])/'runs'/ANALYSIS/'results'
    ac=dict(run_id=ANALYSIS,row_id='all516models',analysis_kind='external_dura',output_root=str(out),rows=records)
    add(ANALYSIS,'score',ac,out/'complete.json',[r['completion'] for r in records],'EXTERNAL_DURA_ANALYZED')
    if len(new)!=1072 or len(cache_jobs)!=34:raise ValueError('Expected1067 frozen stages and5 technical recovery stages')
    previous['external_dura_extension']=dict(new_jobs=1067,recovery_jobs=5,F_models=516,new_F=0,new_G=0,
        new_cache_jobs=34,readouts=list(READOUTS),reused_raw=True,readout_rows=516,
        healthy_PIDs_preserved=list(active),migrated_pending_source_diagnostics=migrated,
        replacement_paths=replacements,new_analysis_run=ANALYSIS,all_seven_views=True)
    write(Path(rt['control_root'])/'external_dura_extension.json',previous['external_dura_extension'])
    write(Path(rt['project'])/'runs'/ANALYSIS/'manifest-at-launch.json',ac)
    return jobs+new
