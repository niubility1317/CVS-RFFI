"""Replace exactly the F0 IQ/ID-misaligned F rows; retain all original artifacts."""
from copy import deepcopy
from pathlib import Path
from .io import read,write,tensor

CONTROL='20261022-phase1-T1-DURA-F-binding-control-r01'
ABLATION='20261022-phase1-T1-DURA-F-ablation-binding-m63-r01'
CONTROLS='20261022-phase1-T1-DURA-F0-controls-binding-m6-r01'
TIMING='20261022-phase1-T1-DURA-F-binding-mechanism-timing-r01'


def smoke(rt,output):
    from .bootstrap import activate,numerical_policy
    activate(rt);numerical_policy()
    from .f_data import SourceDataset
    from .frozen_inference import load_original_T1
    import numpy as np
    import torch
    job=next(j for j in read(rt['source_jobs']) if j['kind']=='F' and 'F01-s2026092701' in j['id'])
    c=read(job['config']);contract=read(c['source_contract'])
    dataset=SourceDataset(c['source_cache'],'L_s',contract)
    original=load_original_T1(2026092701,contract,'cpu')
    y=torch.tensor(dataset.metadata['y'][:256].tolist())
    raw=tensor(np.load(Path(c['source_cache'])/'L_s_x.npy',mmap_mode='r')[:256])
    def predict(x):
        with torch.no_grad():return original.classify_features(original.features(x)).argmax(1)
    wrong=float((predict(raw)==y).float().mean())
    corrected=float((predict(dataset.x[:256])==y).float().mean())
    passed=corrected>=.80 and corrected-wrong>=.50
    write(output,dict(status='PASS' if passed else 'FAIL',source_role='L_s',samples=256,old_binding_accuracy=wrong,correct_binding_accuracy=corrected,
        source_IQ_binding=dataset.iq_binding,checkpoint='Original fixed final T1 seed2026092701; full source ancestry checked',
        checkpoint_used_for_training_initialization=False,query_reads=0,U_TX_truth_reads=0,head_fit_updates=0))
    if not passed:raise ValueError('Fixed source checkpoint does not confirm restored IQ/ID binding')


def prepare(rt,jobs,previous,active):
    """Called after retirement/adoption, so no old submission can race this scope."""
    by_id={j['id']:j for j in jobs};new=[];replacements=[];excluded=set(previous.get('excluded_job_ids',[]))
    original=[j for j in jobs if j['kind']=='F' and read(j['config']).get('protocol_id')=='F0']
    if len(original)!=69:raise ValueError('Exactly63 F ablations plus6 F0 matched controls required')
    config_dir=Path(rt['control_root'])/'repair_configs'
    source_map={}
    def add(old,c,key,completion,dependencies):
        path=config_dir/(key.replace('/','__')+'.json')
        c=dict(c,code_commit=rt['code_commit'],frozen_sources=rt['frozen_sources'])
        write(path,c)
        j=dict(old,id=key,config=str(path),completion=str(completion),dependencies=dependencies,
            log_name=key.replace('/','__'),worker_release=rt['release_root'],execution_commit=rt['code_commit'])
        new.append(j);previous['jobs'][key]=dict(status='QUEUED')
        excluded.add(old['id'])
        if old['id'] not in active and previous['jobs'][old['id']]['status'] not in ('COMPLETE','FAILED'):
            previous['jobs'][old['id']]=dict(previous['jobs'][old['id']],status='SKIPPED_INPUT_BINDING_REPLACED',replacement=key)
        return j
    for old in original:
        c=read(old['config']);run=ABLATION if c.get('ablation') else CONTROLS
        root=Path(rt['project'])/'runs'/run/c['row_id'];source=root/'source'
        if source.exists():raise FileExistsError('New repair source output exists')
        c=dict(c,run_id=run,output_root=str(source),source_IQ_binding_fix='physical_ID_index_v1',replaces_source_row=old['id'])
        key=run+'/'+c['row_id'];train=add(old,c,key,source/'complete.json',list(old.get('dependencies',[])))
        source_map[str(Path(read(old['config'])['output_root']))]=str(source)
        pred_old=by_id[old['id']+'/predict'];pc=read(pred_old['config']);pred=root/'predictions-control-r01'
        pc=dict(pc,run_id=run,source_output=str(source),output_root=str(pred))
        pred_deps=[train['completion'] if p==old['completion'] else p for p in pred_old['dependencies']]
        pred_job=add(pred_old,pc,key+'/predict',pred/'complete.json',pred_deps)
        score_old=by_id[old['id']+'/score'];sc=read(score_old['config']);scores=root/'scores-control-r01'
        sc=dict(sc,run_id=run,prediction_output=str(pred),output_root=str(scores))
        add(score_old,sc,key+'/score',scores/'complete.json',[pred_job['completion']])
        replacements.append(dict(original=old['id'],replacement=key,old_status=previous['jobs'][old['id']]['status'],
            training_budget='same20+10+20/11100, scratch and strict own segments',G='same frozen exact-contract DURA'))
    mechanism=next(j for j in jobs if j['kind']=='timing' and j['id'].endswith('/mechanisms'))
    mc=read(mechanism['config']);root=Path(rt['project'])/'runs'/TIMING
    mc=dict(mc,run_id=TIMING,output_root=str(root),F_mechanisms=[dict(item,source_output=source_map[item['source_output']]) for item in mc['F_mechanisms']])
    dependencies=[str(Path(source_map[p[:-14]])/'complete.json') if p.endswith('/complete.json') and p[:-14] in source_map else p for p in mechanism['dependencies']]
    add(mechanism,mc,TIMING,root/'complete.json',dependencies)
    if len(new)!=208:raise ValueError('Expected69 train +69 predict +69 score +1 replacement mechanism timing')
    jobs.extend(new)
    previous['excluded_job_ids']=sorted(excluded)
    previous['F_binding_recovery']=dict(status='TECHNICAL_INPUT_BINDING_REPLACEMENT',new_scientific_rows=69,
        repeated_hypotheses=0,new_jobs=len(new),replacements=replacements,healthy_owned_PIDs_preserved=list(active),
        original_scope_preserved=True,unaffected_G_retrained=False)
    write(Path(rt['control_root'])/'F_binding_recovery.json',previous['F_binding_recovery'])
    return jobs
