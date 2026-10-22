"""Add only missing strict-supervised architecture rows and L-only readouts."""
from copy import deepcopy
from pathlib import Path
from .io import read, write
from .supervised_physics import ARMS, validate_config

CONTROL='20261022-phase1-PAMF-supervised-control-r01'
RUN='20261022-phase1-PAMF-supervised-physics-m24-r01'
READOUT='20261022-phase1-PAMF-supervised-Lonly-readouts-m6-r01'
TIMING='20261022-phase1-PAMF-supervised-physics-timing-r01'
SEEDS=(2026092701,2026092702,2026092703)


def smoke(rt,output):
    from .f_binding_recovery import smoke as actual_checkpoint_smoke
    actual_checkpoint_smoke(rt,str(Path(output).with_name('strict_source_checkpoint_check.json')))
    from .f_data import SourceDataset
    from .f_train import arguments,class_dimension,original_config
    from .f_model import install
    from experiments.cvs_phase1_stack.runtime import installed
    import torch
    candidate=next(j for j in read(rt['source_jobs']) if j['kind']=='F'
        and read(j['config']).get('ablation')=='F15' and 'binding' in j['id'])
    template=read(candidate['config']);contract=read(template['source_contract'])
    source=SourceDataset(template['source_cache'],'L_s',contract)
    x=source.x[:4];y=torch.tensor(source.metadata['y'][:4].tolist())
    rows=[]
    for arm in ARMS:
        c=architecture_config(template,arm,SEEDS[0],rt['project'],rt['code_commit'],rt['frozen_sources'])
        torch.manual_seed(SEEDS[0])
        with class_dimension(c),installed(original_config(SEEDS[0])) as native:
            a=arguments(c,'cpu');a.input_len=256
            if a.use_unlabeled or a.use_ema_teacher:raise ValueError('Supervised smoke activated U/EMA')
            model=install(native.build_baseline_model(a,'cpu'),c)
        logits=model(x);loss=torch.nn.functional.cross_entropy(logits,y);loss.backward()
        head=model.id_backbone.encoder.core.id_backbone.cls_head.weight
        if logits.shape!=(4,6) or not torch.isfinite(logits).all() or head.grad is None or not torch.isfinite(head.grad).all():
            raise ValueError('Supervised actual-source architecture smoke failed')
        rows.append(dict(architecture=arm,parameters=sum(p.numel() for p in model.parameters()),source_U_reads=0))
    write(output,dict(status='PASS',architectures=rows,actual_checkpoint_no_query=True,
        source_role='L_s',target_reads=0,U_IQ_reads=0,training_initialization='scratch only'))


def validate_supervised_source(source, readouts):
    legal=(source.get('regime')=='matched_sat_supervised' or source.get('ablation') in ('F15','F20'))
    if not legal:raise ValueError('F source is not strictly supervised')
    for name,spec in readouts.items():
        if name=='raw':continue
        if spec.get('kind')!='contract_matched' or spec.get('roles')!='L':
            raise ValueError('Strict supervised system must not inherit a G-LU ancestor')


def architecture_config(template, arm, seed, project, commit, frozen_sources):
    c=deepcopy(template)
    c.update(run_id=RUN,row_id=arm+'-s'+str(seed),model_seed=seed,
        display_method='PAMF-'+arm+'-CE-Sup+DURA-L',method_variant='strict_supervised_architecture',
        regime='matched_sat_supervised',ablation='F15',supervised_architecture=arm,
        U_use=False,unlabeled_batch=0,U_start_epoch=None,own_EMA=False,native_U_gate=False,
        plain_PL_threshold=None,plain_PL_weight=None,G_source_roles='L_s',
        satellite_concat=True,satellite_CE_weight=.68,supervised_only_epochs=50,
        output_root=str(Path(project)/'runs'/RUN/(arm+'-s'+str(seed))/'source'),
        code_commit=commit,frozen_sources=frozen_sources)
    c.pop('replaces_source_row',None)
    validate_config(c)
    return c


def prepare(rt,jobs,previous,active):
    """Run once after exact-owner handover, leaving every old job unchanged."""
    old_ids=[j['id'] for j in jobs]
    configs=Path(rt['control_root'])/'supplement_configs'
    f15={};f20={};supervised={}
    for j in jobs:
        if j['kind']!='F' or j['id'] in previous.get('excluded_job_ids',[]):continue
        c=read(j['config'])
        if c.get('protocol_id')=='F0' and c.get('ablation')=='F15':f15[c['model_seed']]=(j,c)
        if c.get('protocol_id')=='F0' and c.get('ablation')=='F20':f20[c['model_seed']]=(j,c)
        if c.get('regime')=='matched_sat_supervised':supervised[(c['protocol_id'],c['model_seed'])]=j
    if set(f15)!=set(SEEDS) or set(f20)!=set(SEEDS) or len(supervised)!=51:
        raise ValueError('Expected69-repair scope and51 existing strict supervised main rows')
    by_id={j['id']:j for j in jobs}
    matched=next(j for j in jobs if j['kind']=='predict' and j['id'].startswith(supervised[('F0',SEEDS[0])]['id']+'/'))
    ptemplate=read(matched['config']);gspec=ptemplate['readouts']['DURA-L']
    validate_supervised_source(dict(regime='matched_sat_supervised'),{'raw':None,'DURA-L':gspec})
    gcompletion=str(Path(gspec['checkpoint']).parents[1]/'complete.json')
    new=[];training=[];reuse=[]
    def add(key,kind,c,completion,depends,exclusive=False):
        path=configs/(key.replace('/','__')+'.json')
        write(path,dict(c,stage=kind,code_commit=rt['code_commit'],frozen_sources=rt['frozen_sources']))
        j=dict(id=key,kind=kind,config=str(path),completion=str(completion),
            successful_statuses={'F':['SOURCE_FROZEN'],'predict':['ALL_PREDICTIONS_COMPLETE'],
                'score':['SCORED_COMPLETE'],'timing':['TIMING_COMPLETE']}[kind],
            dependencies=list(depends),log_name=key.replace('/','__'),exclusive=exclusive,
            worker_release=rt['release_root'],execution_commit=rt['code_commit'])
        new.append(j);previous['jobs'][key]=dict(status='QUEUED')
        return j
    def evaluation(source_job,source,run,row,reuse_raw=False):
        base=Path(rt['project'])/'runs'/run/row;pred=base/'predictions';score=base/'scores'
        c=dict(ptemplate,run_id=run,row_id=row,model_seed=source['model_seed'],
            source_run_id=source['run_id'],source_row_id=source['row_id'],source_output=source['output_root'],
            readouts={'raw':None,'DURA-L':gspec},strict_supervised_system=True,
            source_physics_diagnostics=True,output_root=str(pred))
        dependencies=[source_job['completion'],gcompletion]
        if reuse_raw:
            original=by_id[source_job['id']+'/predict']
            c['raw_predictions_from']=read(original['config'])['output_root']
            dependencies.append(original['completion'])
        pj=add(run+'/'+row+'/predict','predict',c,pred/'complete.json',dependencies)
        sc=read(by_id[matched['id'].removesuffix('/predict')+'/score']['config'])
        sc=dict(sc,run_id=run,row_id=row,model_seed=source['model_seed'],prediction_output=str(pred),
            output_root=str(score),readouts=['raw','DURA-L'])
        add(run+'/'+row+'/score','score',sc,score/'complete.json',[pj['completion']])
    for seed in SEEDS:
        template=f15[seed][1]
        for arm in ARMS:
            c=architecture_config(template,arm,seed,rt['project'],rt['code_commit'],rt['frozen_sources'])
            key=RUN+'/'+c['row_id'];j=add(key,'F',c,Path(c['output_root'])/'complete.json',[])
            training.append(dict(id=key,architecture=arm,seed=seed));evaluation(j,c,RUN,c['row_id'])
        for name,found in (('X3_CE',f15),('T1_SUP',f20)):
            j,c=found[seed];row=name+'-s'+str(seed)
            evaluation(j,c,READOUT,row,reuse_raw=True)
            reuse.append(dict(source=j['id'],readout=READOUT+'/'+row,training_repeated=False))
    timing_old=next(j for j in jobs if j['kind']=='timing' and 'F-binding-mechanism' in j['id'])
    tc=read(timing_old['config']);models=[dict(name='X3_CE',source_output=f15[SEEDS[0]][1]['output_root'])]
    models += [dict(name=arm,source_output=str(Path(rt['project'])/'runs'/RUN/(arm+'-s'+str(SEEDS[0]))/'source')) for arm in ARMS]
    tr=Path(rt['project'])/'runs'/TIMING
    tc=dict(tc,run_id=TIMING,output_root=str(tr),G=gspec,F_mechanisms=models,G_mechanisms=[],external_models=[],
        mechanisms_only=True,strict_supervised_system=True)
    add(TIMING,'timing',tc,tr/'complete.json',[str(Path(m['source_output'])/'complete.json') for m in models]+[gcompletion],exclusive=True)
    if len(new)!=85:raise ValueError('Expected24 F +30 predictions +30 scores +1 timing')
    jobs.extend(new)
    if [j['id'] for j in jobs[:len(old_ids)]]!=old_ids:raise ValueError('Old queue modified')
    previous['supervised_extension']=dict(new_F=24,new_readout_rows=30,new_jobs=85,
        trained=training,reused=reuse,existing_strict_supervised_F=51,new_G_training=0,
        source_U_IQ_reads=0,healthy_PIDs_preserved=list(active),source_contract='F0 existing validated roles')
    write(Path(rt['control_root'])/'supervised_extension.json',previous['supervised_extension'])
    return jobs
