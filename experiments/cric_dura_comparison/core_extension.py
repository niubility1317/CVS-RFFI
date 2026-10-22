"""Append the missing grouped-objective cells; keep all healthy workers intact."""
from copy import deepcopy
from pathlib import Path
from .io import read, write
from .core_groups import RUN, READOUT, ANALYSIS, REPAIR, MAIN, L_READOUT, SEEDS, REUSE, switches, validate_config, plan


def training_config(template, bits, seed, rt):
    c = deepcopy(template)
    use_u = bits[2] == '1'
    row = 'ABC' + bits + '-s' + str(seed)
    c.update(run_id=RUN, row_id=row, model_seed=seed, core_mechanisms=switches(bits),
        display_method='PAMF core ABC' + bits, method_variant='three_core_factorial',
        regime='native_cric', method_id='pamf_t1_e50', ablation=None if use_u else 'F20',
        U_use=use_u, own_EMA=use_u, native_U_gate=use_u,
        U_start_epoch=14 if use_u else None, unlabeled_batch=256 if use_u else 0,
        supervised_only_epochs=13 if use_u else 50,
        satellite_concat=False, satellite_CE_weight=0.,
        output_root=str(Path(rt['project'])/'runs'/RUN/row/'source'),
        code_commit=rt['code_commit'], frozen_sources=rt['frozen_sources'])
    for key in ('replaces_source_row','supervised_architecture','parent_source','step_offset','epoch_offset','segment_epochs'):
        c.pop(key, None)
    validate_config(c)
    return c


def smoke(rt, output):
    """One real fixed-checkpoint source check, then four missing objective cells."""
    from .f_binding_recovery import smoke as actual_checkpoint_smoke
    actual_checkpoint_smoke(rt, str(Path(output).with_name('core_actual_checkpoint_smoke.json')))
    import torch
    from torch.nn import functional as F
    from .f_data import SourceDataset
    from .f_train import arguments, original_config, class_dimension
    from .t1_model import install
    from experiments.cvs_phase1_stack.runtime import installed
    jobs=read(rt['source_jobs'])
    candidate=next(j for j in jobs if j['id']==REPAIR+'/F16-s'+str(SEEDS[0]))
    template=read(candidate['config']);contract=read(template['source_contract'])
    data=SourceDataset(template['source_cache'],'L_s',contract)
    x=data.x[:128];y=torch.tensor(data.metadata['y'][:128].tolist())
    domains=torch.tensor(data.metadata['domain'][:128].tolist())
    metadata={k:torch.tensor(data.metadata[k][:128].tolist()) for k in ('rx_i','day_i','eq_i','sig_i','base_index')}
    observations=[]
    for bits in ('000','001','010','100'):
        c=training_config(template,bits,SEEDS[0],rt)
        # Exercise full CRIC weight without shortening or initializing a real run.
        c['step_offset']=4440
        torch.manual_seed(SEEDS[0])
        with class_dimension(c),installed(original_config(SEEDS[0])) as native:
            args=arguments(c,'cpu');args.input_len=256
            if args.use_unlabeled != (bits[2]=='1') or args.use_ema_teacher != (bits[2]=='1'):
                raise ValueError('Grouped SSL-off did not disable native U/EMA')
            model=install(native.build_baseline_model(args,'cpu'),c).train()
        with model._phase1_labeled_feature_context():
            out=model(x,return_aux=True,domain_labels=domains)
        ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
        correction,metrics=model._source_labeled_feature_hook(model=model,labeled_output=out,labels=y,
            extra=(None,metadata),domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
        loss=ce+correction;loss.backward()
        head=model.id_backbone.encoder.core.id_backbone.cls_head.weight
        if (not torch.isfinite(loss) or head.grad is None or not torch.isfinite(head.grad).all()
                or metrics['T1_L_packets']!=768 or model._cd_calls!=1):
            raise ValueError('Actual L/default-head grouped objective smoke failed')
        observations.append(dict(bits=bits,core={k:metrics[k] for k in ('core_A','core_B','core_C')},
            native_U_enabled=args.use_unlabeled,native_EMA_enabled=args.use_ema_teacher,
            L_forward_packets=metrics['T1_L_packets'],optimizer_updates=0))
    write(output,dict(status='PASS',real_checkpoint_no_query=True,missing_cells=observations,
        source_role='L_s',U_IQ_reads_in_smoke=0,target_reads=0,head_fit_updates=0,
        real_training_budgets_unchanged=True))


def prepare(rt, jobs, previous, active):
    by_id={j['id']:j for j in jobs}
    excluded=set(previous.get('excluded_job_ids',[]))
    required=[REPAIR+'/'+arm+'-s'+str(seed) for arm in ('F04','F16','F20') for seed in SEEDS]
    if any(k not in by_id or k in excluded for k in required):
        raise ValueError('Corrected F04/F16/F20 sources missing; never use old input-invalid F')
    template=read(by_id[REPAIR+'/F16-s'+str(SEEDS[0])]['config'])
    # Obtain already-registered matching source/test handles, never reconstruct data.
    original_predictions={seed:by_id[MAIN+'/F0-native_cric-s'+str(seed)+'/predict'] for seed in SEEDS}
    ptemplate=read(original_predictions[SEEDS[0]]['config'])
    sup_key='20261022-phase1-T1-DURA-F0-controls-binding-m6-r01/F0-matched_sat_supervised-s'+str(SEEDS[0])
    sup=read(by_id[sup_key+'/predict']['config'])
    gs={'DURA-L':sup['readouts']['DURA-L'],'DURA-LU':ptemplate['readouts']['DURA-LU']}
    if gs['DURA-L'].get('roles')!='L' or gs['DURA-L'].get('kind')!='contract_matched':
        raise ValueError('Same-contract L-only G required')
    g_l_done=str(Path(gs['DURA-L']['checkpoint']).parents[1]/'complete.json')
    sctemplate=read(by_id[MAIN+'/F0-native_cric-s'+str(SEEDS[0])+'/score']['config'])
    configs=Path(rt['control_root'])/'core_configs';new=[];cells=[]

    def add(key,kind,c,completion,dependencies,status):
        path=configs/(key.replace('/','__')+'.json')
        write(path,dict(c,stage=kind,code_commit=rt['code_commit'],frozen_sources=rt['frozen_sources']))
        job=dict(id=key,kind=kind,config=str(path),completion=str(completion),dependencies=list(dependencies),
            successful_statuses=[status],log_name=key.replace('/','__'),
            worker_release=rt['release_root'],execution_commit=rt['code_commit'])
        new.append(job);previous['jobs'][key]=dict(status='QUEUED')
        return job

    def evaluate(run,row,bits,seed,source_job,source_cfg,original=None,reuse=None,readouts=None):
        folder=Path(rt['project'])/'runs'/run/row;pred=folder/'predictions';scores=folder/'scores'
        pc=dict(ptemplate,run_id=run,row_id=row,model_seed=seed,source_output=source_cfg.get('output_root'),
            source_run_id=source_cfg['run_id'],source_row_id=source_cfg['row_id'],
            output_root=str(pred),readouts=readouts,source_physics_diagnostics=True)
        pc.pop('original_arm',None)
        deps=[g_l_done]
        if source_job is not None:deps.append(source_job['completion'])
        if original is not None:pc['original_arm']=original
        if reuse is not None:
            rc=read(reuse['config']);pc['raw_predictions_from']=rc['output_root'];deps.append(reuse['completion'])
        pj=add(run+'/'+row+'/predict','predict',pc,pred/'complete.json',deps,'ALL_PREDICTIONS_COMPLETE')
        sc=dict(sctemplate,run_id=run,row_id=row,model_seed=seed,prediction_output=str(pred),
            output_root=str(scores),readouts=list(readouts))
        sj=add(run+'/'+row+'/score','score',sc,scores/'complete.json',[pj['completion']],'SCORED_COMPLETE')
        return dict(run_id=run,row_id=row,model_seed=seed,scores=str(scores/'scores.json'),completion=sj['completion'])

    for item in plan()['rows']:
        bits=item['bits'];seed=item['model_seed'];row=item['row_id']
        if bits not in REUSE:
            c=training_config(template,bits,seed,rt)
            fj=add(RUN+'/'+row,'F',c,Path(c['output_root'])/'complete.json',[],'SOURCE_FROZEN')
            record=evaluate(RUN,row,bits,seed,fj,c,readouts={'raw':None,**gs})
            records={name:dict(record,readout=name) for name in ('raw','DURA-L','DURA-LU')}
            source=dict(run_id=RUN,row_id=row,completion=fj['completion'],new_F=True)
        else:
            arm=REUSE[bits]
            if arm=='T1':
                source_job=None;oldpred=original_predictions[seed]
                oldpc=read(oldpred['config']);oldscore=by_id[MAIN+'/F0-native_cric-s'+str(seed)+'/score']
                c=dict(run_id=oldpc['run_id'],row_id=oldpc['row_id'])
            else:
                key=REPAIR+'/'+arm+'-s'+str(seed);source_job=by_id[key];c=read(source_job['config'])
                oldpred=by_id[key+'/predict'];oldscore=by_id[key+'/score']
            sc=read(oldscore['config'])
            oldrecord=dict(run_id=sc['run_id'],row_id=sc['row_id'],model_seed=seed,
                scores=str(Path(sc['output_root'])/'scores.json'),completion=oldscore['completion'])
            records={name:dict(oldrecord,readout=name) for name in ('raw','DURA-LU')}
            if arm=='F20':
                key=L_READOUT+'/T1_SUP-s'+str(seed)+'/score';j=by_id[key];cscore=read(j['config'])
                records['DURA-L']=dict(run_id=cscore['run_id'],row_id=cscore['row_id'],model_seed=seed,
                    scores=str(Path(cscore['output_root'])/'scores.json'),completion=j['completion'],readout='DURA-L')
            else:
                record=evaluate(READOUT,row,bits,seed,source_job,c,
                    original='T1' if arm=='T1' else None,reuse=oldpred,readouts={'raw':None,'DURA-L':gs['DURA-L']})
                records['DURA-L']=dict(record,readout='DURA-L')
            source=dict(run_id=c['run_id'],row_id=c['row_id'],new_F=False,
                completion=source_job['completion'] if source_job is not None else 'Existing original T1 source chain')
        cells.append(dict(bits=bits,model_seed=seed,source=source,readouts=records))
    if len(new)!=54 or sum(j['kind']=='F' for j in new)!=12:
        raise ValueError('Expected12 new F +21prediction +21score before analysis')
    analysis_root=Path(rt['project'])/'runs'/ANALYSIS
    ac=dict(run_id=ANALYSIS,row_id='all24cells',analysis_kind='core_factorial',
        output_root=str(analysis_root/'results'),cells=cells,main_table=['111','011','101','110'],num_classes=6)
    add(ANALYSIS,'score',ac,analysis_root/'results/complete.json',
        sorted({r['completion'] for cell in cells for r in cell['readouts'].values()}),'CORE_ANALYZED')
    alljobs=jobs+new
    # Scientific priority is fixed by the user's new main-table scope, never scores.
    priority=set(required)|{j['id'] for j in new if j['kind']=='F'}
    alljobs=[j for j in alljobs if j['id'] in priority]+[j for j in alljobs if j['id'] not in priority]
    previous['core_extension']=dict(status='THREE_FUNCTIONAL_GROUPS',main_table=['111','011','101','110'],
        new_F=12,reused_F=12,new_G=0,logical_models=24,new_jobs=55,new_prediction_rows=21,
        new_score_rows=21,healthy_PIDs_preserved=list(active),old_jobs_not_cancelled=True,
        new_scientific_run=RUN,manifest=str(analysis_root/'manifest-at-launch.json'))
    write(analysis_root/'manifest-at-launch.json',ac)
    write(Path(rt['control_root'])/'core_extension.json',previous['core_extension'])
    return alljobs
