"""P69 CPU-only optimizer-continuity factorial; no IQ or checkpoint reads."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'
import argparse
import importlib.util
import json
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np

RUN = '20261018-phase1-FcrossRX-optimizerKeep-e30-s2026092701-m2-r01'
REF = '20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
PARENT_COHORT = '20261018-diagnostic-FcrossRX-parent20-cohort-s2026092701-m1-r01'
SEED = 2026092701
ARMS = ('NOAS30', 'AS30', 'NOAS_KEEP', 'AS_KEEP')
NEW_ARMS = ARMS[2:]
PAIRS = (('AS30','NOAS30'), ('NOAS_KEEP','NOAS30'), ('AS_KEEP','AS30'),
         ('AS_KEEP','NOAS_KEEP'), ('AS_KEEP','NOAS30'))


def load_common():
    path = Path(__file__).with_name('cvs_temporal_bank67_analysis_20261018.py')
    spec = importlib.util.spec_from_file_location('_p69_cpu_common67', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.RUN, module.ARMS, module.NEW_ARMS, module.PAIRS = RUN, ARMS, NEW_ARMS, PAIRS
    module.c.RUN, module.c.NEW_ARMS, module.c.PAIRS = RUN, NEW_ARMS, PAIRS
    module.c.row_specs, module.c.training_complete = row_specs, training_complete
    return module


def row_specs(project):
    p = Path(project)/'runs'
    return [dict(key=key, run=REF if i<2 else RUN, arm='B00' if i%2==0 else 'B01',
        row_id=('B00' if i%2==0 else 'B01')+'-s'+str(SEED), reused=i<2,
        root=p/(REF if i<2 else RUN)/(('B00' if i%2==0 else 'B01')+'-s'+str(SEED)),
        parent=p/REF/(('B00' if i%2==0 else 'B01')+'-s'+str(SEED))/'stage20')
        for i,key in enumerate(ARMS)]


def training_complete(spec,project):
    roles=c.read(Path(project)/'runs'/c.CONTRACT_RUN/'source_contract.json')['role_ids']
    for folder,parent,e,l,u,updates in ((spec['parent'],True,20,13,7,4440),
                                      (spec['root']/'source',False,10,0,10,2220)):
        done=c.read(folder/'completion.json');cfg=done['config'];frozen=c.read(folder/'source_frozen.json')
        init=c.read(folder/'initialization.json');result=c.read(folder/'result.json')
        args=c.read(folder/'actual_native_args.json');args=args.get('args',args)
        expected=[] if parent else [str(spec['parent']/'final_ssdg.pth')]
        c.require((done.get('epoch'),done.get('optimizer_steps'),cfg.get('epochs'),cfg.get('label_epochs'),
                   cfg.get('pseudo_epochs'),cfg.get('updates'))==(e,updates,e,l,u,updates), 'Exact E20+10 calendar required')
        c.require(cfg.get('run_id')==(REF if parent else spec['run']) and cfg.get('row_id')==spec['row_id'] and
                  cfg.get('model_seed')==SEED and cfg.get('As') is (spec['arm']=='B01'), 'Exact actual row/run/seed/As required')
        c.require(cfg.get('checkpoint_sources')==init.get('checkpoint_sources')==expected and
                  init.get('scratch_only') is parent and init.get('source_roles')=='EXACT_MATCH' and
                  init.get('target_contact') is False, 'Exact source-fixed own student/EMA provenance required')
        c.require(frozen.get('status')=='SOURCE_FROZEN' and frozen.get('config')==cfg and
                  frozen.get('selection')=='final_only' and frozen.get('target_access') is False,
                  'Fixed final-only default head required')
        c.require(done.get('target_access') is False and done.get('head_fit_updates')==0 and
                  done.get('native_fitted_probes_disabled') is True and result.get('config')==cfg and
                  result.get('U_truth_read') is False and result.get('target_access') is False and
                  result.get('head_fit_updates')==0, 'Joint head/hidden U/source boundaries differ')
        c.require(c.read(folder/'source_contract.json')['role_ids']==roles and result.get('source_RX')==list(c.RX) and
                  result.get('source_RX_holdout') is None and done['final_source_metrics'].get('source_val_count')==27000,
                  'Unchanged complete fullsource/all5RX roles required')
        c.require(cfg.get('Ag_mode')=='NONE' and cfg.get('new_Ag') is False and cfg.get('L_NN_per_step')==768 and
                  cfg.get('total_L_loss_scale')==1.25, 'Original sixL/1.25/no-Ag recipe required')
        c.require(args.get('pseudo_temporal_mode')=='batch_neighbor' and args.get('pseudo_temporal_gate') is True and
                  args.get('pseudo_temporal_bank_min_streak')==2, 'Original native U masks required')
        if not parent:
            c.require(cfg.get('parent_source')==str(spec['parent']) and cfg.get('epoch_offset')==20 and
                      cfg.get('step_offset')==4440, 'Strict own20 and reset native tail clocks required')
    return c.read(spec['root']/'source/completion.json')


q=load_common();c=q.c
_write=q._write


def effects(rows,new,control):
    result={}
    for view in c.VIEWS:
        result[view]={key+'_delta_pp':100*(rows[new][view][key]-rows[control][view][key])
                      for key in ('accuracy','macro_f1','worst_RX','satellite_retention')}
    result['urban_accuracy_mean_delta_pp']=float(np.mean([result[v]['accuracy_delta_pp'] for v in c.VIEWS[-3:]]))
    return result


def enrich_formal(obj):
    rows=obj['rows']
    obj['interactions']={view:q.interaction({arm:rows[arm][view] for arm in ARMS},
        ('accuracy','macro_f1','worst_RX','satellite_retention')) for view in c.VIEWS}
    obj['mechanism']={new:dict(control=control,effects=effects(rows,new,control))
        for new,control in (('NOAS_KEEP','NOAS30'),('AS_KEEP','AS30'))}
    obj['stage']={}
    for new,control in (('AS30','NOAS30'),('AS_KEEP','NOAS_KEEP')):
        delta=effects(rows,new,control);clean=delta['clean']['accuracy_delta_pp']
        obj['stage'][new]=dict(matched_noAs=control,clean_delta_pp=clean,
            low_urban_delta_pp=delta[c.VIEWS[-1]]['accuracy_delta_pp'],
            mid_urban_delta_pp=delta[c.VIEWS[-2]]['accuracy_delta_pp'],
            urban_mean_delta_pp=delta['urban_accuracy_mean_delta_pp'],
            clean_within_0p1pp=bool(clean>=-.1-1e-10),single_seed_observed_dual_gain=bool(clean>0 and delta['urban_accuracy_mean_delta_pp']>0),
            reproducible_dual_gain=False)
    obj['factorial_interaction_is_descriptive']=True
    obj['optimizer_state_is_not_identity_gain']=True


def processed_write(path,obj):
    if Path(path).name=='formal_summary.json':enrich_formal(obj)
    _write(path,obj)


c.write=processed_write


def source_criteria(as_clean,noas_clean,as_midlow,old_as_midlow):
    clean=100*(as_clean-noas_clean);urban=100*(as_midlow-old_as_midlow)
    checks=dict(matched_clean_ge_minus_0p1pp=clean>=-.1-1e-10,midlow_vs_As30_ge_0p5pp=urban>=.5-1e-10)
    return dict(checks=checks,source_qualified=all(checks.values()),clean_delta_pp=clean,midlow_delta_pp=urban)


def read_source_decision(project):
    value=c.read(Path(project)/'runs'/RUN/'source_matrix_decision.json')
    c.require(value.get('status') in c.DECISION_COMPLETE|c.DECISION_PARTIAL and value.get('formal_scores_read') is False and
              value.get('target_feedback') is False and value.get('NN')==0 and value.get('model_seed_count')==1,
              'Saved source-only decision must precede any formal read')
    if value['status'] in c.DECISION_COMPLETE:
        checks=value.get('checks');c.require(isinstance(checks,dict) and len(checks)==2 and
            all(type(v) is bool for v in checks.values()) and type(value.get('source_qualified')) is bool and
            value['source_qualified']==all(checks.values()), 'Exactly two fixed source criteria and verdict required')
    return value


def optimizer_readout(folder):
    """Read saved summaries only; concrete core telemetry is validated below."""
    folder=Path(folder)
    initialization=c.read(folder/'optimizer_initialization.json')
    epochs=list(c.lines(folder/'optimizer_state_epochs.jsonl.gz'))
    final=c.read(folder/'optimizer_state_final.json')
    value=validate_optimizer_telemetry(initialization,epochs,final)
    saved=c.read(folder/'result.json')['optimizer_keep_summary']
    c.require(saved.get('initialization')==initialization and saved.get('final')==final and saved.get('epochs')==10 and
        saved.get('inherited_per_parameter_steps') is True and saved.get('fresh_cosine10') is True and
        saved.get('fresh_Sinc_segment_counters') is True and saved.get('old_LR_restored') is False,
        'Final result must report actual parent state and fresh schedule/Sinc counters')
    return value


def state_statistics(value,max_step):
    c.require(isinstance(value,dict),'Actual AdamW state statistics required')
    entries=value.get('state_entries');parameters=value.get('parameter_count');missing=value.get('state_missing_parameters')
    c.require(type(entries) is int and type(parameters) is int and type(missing) is int and
              0<entries<=parameters and missing==parameters-entries,'Used/unused optimizer parameters must reconcile')
    hist=value.get('state_step_histogram');c.require(isinstance(hist,dict) and hist and
        all(str(int(k))==k and 0<=int(k)<=max_step and type(n) is int and n>0 for k,n in hist.items()) and
        sum(hist.values())==entries,'Finite complete per-parameter AdamW step histogram required')
    c.require(value.get('state_step_min')==min(map(int,hist)) and value.get('state_step_max')==max(map(int,hist)),
              'Optimizer step extrema differ from saved histogram')
    for key in ('exp_avg_numel','exp_avg_sq_numel'):c.require(type(value.get(key)) is int and value[key]>0,'Actual nonempty moment tensor sizes required')
    c.require(value['exp_avg_numel']==value['exp_avg_sq_numel'],'First and second moment tensor sizes differ')
    for key in ('exp_avg_l2','exp_avg_sq_l2'):c.require(c.finite(value.get(key),key)>=0,'Nonnegative moment norm required')
    groups=value.get('groups');c.require(isinstance(groups,list) and groups and sum(g['state_entries'] for g in groups)==entries,
        'All saved optimizer groups must reconcile')
    return value


def validate_optimizer_telemetry(initialization,epochs,final):
    c.require(initialization.get('reset') is False and initialization.get('parent_state_loaded') is True,
              'Parent full-model AdamW state must actually be inherited')
    c.require(initialization.get('schedule')=='cosine10' and initialization.get('lr')==.0002 and
        initialization.get('weight_decay')==.0001 and initialization.get('parent_optimizer_steps')==4440 and
        initialization.get('fresh_sinc_unit_steps')==initialization.get('fresh_sinc_physical_clip_calls')==0 and
        initialization.get('extra_NN')==initialization.get('optimizer_updates_during_load')==0,
        'Parent moment load must retain the fresh LR/Sinc/update schedule')
    restored=state_statistics(initialization.get('restored'),4440)
    c.require(initialization.get('state_entries')==restored['state_entries'] and
        all(restored.get(key) is True for key in ('all_state_tensors_equal','per_parameter_steps_equal',
        'fresh_hyperparameters_equal','current_parameter_pointers_retained','custom_attributes_retained',
        'named_layout_checked','parent_named_layout_checked')),'Exact loaded moments/steps/fresh options/layout required')
    for group in restored['groups']:
        c.require(group.get('fresh_lr')==group.get('restored_lr')==group.get('lr'),
                  'Fresh group LR must survive optimizer state load')
        c.finite(group.get('parent_lr'),'parent group lr')
    c.require(len(epochs)==10, 'Every tail epoch optimizer-state summary required')
    flat=[]
    for epoch,record in enumerate(epochs,1):
        updates=epoch*222;stats=state_statistics(record.get('optimizer_state'),4440+updates)
        c.require(record.get('epoch')==epoch and record.get('cumulative_epoch')==epoch+20 and
            record.get('successful_updates')==updates and record.get('parent_successful_updates')==4440 and
            record.get('sinc_unit_steps')==record.get('sinc_physical_clip_calls')==updates and
            record.get('no_extra_optimizer_updates') is True and record.get('extra_NN')==0,
            'One actual summary per original native epoch and fresh Sinc counter required')
        inherited=record.get('inherited_state_entries');new=record.get('new_state_entries');hist=record.get('inherited_step_delta_histogram')
        c.require(inherited==restored['state_entries'] and type(new) is int and new>=0 and inherited+new==stats['state_entries'] and
            isinstance(hist,dict) and sum(hist.values())==inherited and
            all(str(int(k))==k and 0<=int(k)<=updates and type(v) is int and v>0 for k,v in hist.items()),
            'Inherited per-parameter step deltas must be bounded by actual updates; unused states may be absent')
        last=record.get('last_unit_step');c.require(isinstance(last,dict) and last.get('step')==updates and
            last.get('physical_clip_calls')==updates,'Unchanged final original Sinc unit/clip step required')
        flat.append(dict(epoch=epoch,cumulative_epoch=epoch+20,successful_updates=updates,
            state_entries=stats['state_entries'],state_missing_parameters=stats['state_missing_parameters'],
            step_min=stats['state_step_min'],step_max=stats['state_step_max'],
            exp_avg_l2=stats['exp_avg_l2'],exp_avg_sq_l2=stats['exp_avg_sq_l2'],
            inherited_state_entries=inherited,new_state_entries=new,inherited_step_delta_histogram=hist))
    c.require(final==epochs[-1],'Final optimizer summary must equal completed epoch10 state')
    return dict(initialization=initialization,epochs=flat,raw_epoch_records=epochs,final=final,
                optimizer_readout_is_not_accuracy=True,unused_parameters_not_forced_to_step4440=True)


def training_analysis(project,specs,out):
    tails={};curves=[];coverage=[];optimizer_rows=[];by_key={s['key']:s for s in specs}
    for spec in specs:
        key=spec['key'];tail,ordinary,_=q.parse_tail(spec['root']/'source',spec['arm']=='B01',False)
        if not spec['reused']:
            ref=by_key['AS30' if spec['arm']=='B01' else 'NOAS30']
            tail['physical_sequence']=q.same_sequence(spec['root']/'source/training_sequence.jsonl.gz',ref['root']/'source/training_sequence.jsonl.gz')
            c.require(ordinary==tails[ref['key']]['ordinary'],'Original ordinary MSE stream must be preserved')
            tail['optimizer_readout']=optimizer_readout(spec['root']/'source')
            optimizer_rows.extend(dict(row_id=key,**record) for record in tail['optimizer_readout']['epochs'])
        tail['ordinary']=ordinary
        result=c.read(spec['root']/'source/result.json')
        if spec['arm']=='B01':
            channel=result['L_channel'];cells=channel['strata']
            c.require(channel.get('packets')==142080 and channel.get('session_count')==15 and
                sum(r['count'] for r in cells)==142080 and {r['RX'] for r in cells}==set(c.RX) and
                {r['day'] for r in cells}==set(c.DAYS) and {r['scene'] for r in cells}==set(c.SCENES),
                'Original As physical source coverage required')
            coverage.extend(dict(row_id=key,**r) for r in cells)
            if not spec['reused']:
                joint=result['TX_RX_scene_state_lock']
                c.require(sum(r['count'] for r in joint)==142080 and
                    len({(int(r['TX']),int(r['RX']),r['scene']) for r in joint})==90,
                    'All true TX/source RX/scene labeled satellite support required')
                c.csv_write(out/(key+'_TX_RX_scene_state_lock.csv'),joint)
        curves.extend(dict(row_id=key,reused=spec['reused'],model_seed=SEED,**r)
                      for r in c.epoch_curve(spec['parent'],0)+tail['curve'])
        tails[key]=tail
    for tail in tails.values():tail.pop('ordinary')
    c.csv_write(out/'training_curves.csv',curves);c.csv_write(out/'optimizer_epochs.csv',optimizer_rows)
    c.csv_write(out/'training_As_coverage.csv',coverage)
    result=dict(status='FULL_TRAINING_LOGS_VERIFIED',rows=tails,new_model_epochs=20,new_updates=4440,
        shared_parent_model_epochs=40,unique_total_model_epochs=80,U_truth_read=False,extra_NN=0,
        native_U_temporal_mode='batch_neighbor',optimizer_state_is_not_identity_gain=True)
    c.write(out/'training_summary.json',result)
    return result,curves


def parent_cohort_metadata(meta):
    expected=dict(status='SOURCE_READOUT_COMPLETE',run_id=PARENT_COHORT,model_seed=SEED,
        checkpoint_run=REF,checkpoint_row='B01-s'+str(SEED),checkpoint_stage='stage20',checkpoint_key='model',
        checkpoint_epoch=20,checkpoint_optimizer_steps=4440,checkpoint_selection='final_only',source_role='V',
        source_unique_V=1800,channel_draws=2,scenes=list(c.SCENES),condition='full',condition_index=0,
        NN=12602,training_updates=0,head_fit_updates=0,U_truth_read=False,target_access=False,
        formal_scores_read=False,new_channel_calls=0,model_state_unchanged=True,all_parameters_frozen=True,
        model_parameters=900722,model_state_keys=206,default_head_shape=[6,160],default_head_scale=30.)
    c.require(isinstance(meta,dict) and all(meta.get(k)==v for k,v in expected.items()),
              'Exact P43 As20 student cohort required; never AS_CE20/EMA/As30')
    return meta


def parent_cohort_arrays(parent,current,physics=None):
    """All arrays are saved source default-head logits; no sorting or fitting."""
    for name in ('ids','TX','RX','day'):c.require(np.array_equal(parent[name],current[name]),'Parent/source physical order differs')
    n=len(parent['ids']);c.require(parent['clean'].shape==current['clean'].shape==(n,6) and
        parent['satellite'].shape==current['satellite'].shape==(2,3,n,6) and
        all(np.isfinite(x[k]).all() for x in (parent,current) for k in ('clean','satellite')),
        'Finite same-shape parent/current default-head source logits required')
    good=parent['clean'].argmax(1)==parent['TX'];records=[]
    for scene in ('clean',*c.SCENES):
        si=None if scene=='clean' else c.SCENES.index(scene)
        for draw in ((-1,) if si is None else (-1,0,1)):
            ds=(0,1) if draw==-1 else (draw,);copies=1 if si is None else len(ds)
            y=np.tile(parent['TX'],copies);cohort=np.tile(good,copies)
            rx=np.tile(parent['RX'],copies);day=np.tile(parent['day'],copies)
            zp=parent['clean'] if si is None else np.concatenate([parent['satellite'][d,si] for d in ds])
            zc=current['clean'] if si is None else np.concatenate([current['satellite'][d,si] for d in ds])
            masks=[('overall','ALL',np.ones(len(y),bool))]
            for dim,values,allowed in (('RX',rx,c.RX),('day',day,c.DAYS),('TX',y,range(6))):masks.extend((dim,str(v),values==v) for v in allowed)
            if physics is not None and si is not None:
                for field,labels,dim in (('state',c.STATES,'state'),('lock',c.LOCKS,'lock')):
                    z=np.concatenate([physics[field][d,si,0] for d in ds]);masks.extend((dim,name,z==i) for i,name in enumerate(labels))
                noise=np.concatenate([physics['signal_to_added_noise_db'][d,si,0] for d in ds])
                masks.extend(('signal_to_added_noise_db',name,(noise>=lo)&(noise<hi)) for name,lo,hi in
                    (('<3',-np.inf,3),('3-8',3,8),('8-15',8,15),('>=15',15,np.inf)))
                masks.append(('signal_to_added_noise_db','MISSING',np.isnan(noise)))
            for dim,label,mask in masks:
                cc=cohort&mask;den=int(cc.sum());old=zp.argmax(1)==y;new=zc.argmax(1)==y
                z=zc.copy();z[np.arange(len(y)),y]=-np.inf;margin=(zc[np.arange(len(y)),y]-z.max(1))/30.
                records.append(dict(scene=scene,draw=draw,dimension=dim,stratum=label,count=int(mask.sum()),
                    parent_clean_correct_count=den,parent_retained=int((old&cc).sum()),current_retained=int((new&cc).sum()),
                    parent_retention=float(old[cc].mean()) if den else None,current_retention=float(new[cc].mean()) if den else None,
                    retention_delta_pp=100*int(((new.astype(int)-old.astype(int))*cc).sum())/den if den else None,
                    repairs_within_parent_cohort=int((~old&new&cc).sum()),new_errors_within_parent_cohort=int((old&~new&cc).sum()),
                    mean_margin_over30_within_parent_cohort=float(margin[cc].mean()) if den else None,empty_cohort=den==0))
    return dict(parent_clean_correct_count=int(good.sum()),records=records)


def parent_cohort_analysis(project,specs,out):
    root=Path(project)/'runs'/PARENT_COHORT
    if not (root/'complete.json').is_file() or not (root/'predictions.npz').is_file():
        raise FileNotFoundError('Exact cached P43 As20 student source cohort unavailable; no replacement or new inference')
    parent_cohort_metadata(c.read(root/'complete.json'))
    with np.load(root/'predictions.npz',allow_pickle=False) as f:parent={k:f[k].copy() for k in ('ids','TX','RX','day','clean','satellite')}
    c.require(len(parent['ids'])==1800 and len(set(parent['ids'].tolist()))==1800,'Original parent panel count/IDs required')
    panel=Path(project)/'runs'/c.PANEL_RUN/'diagnostic'
    with np.load(panel/'observations/physics.npz',allow_pickle=False) as f:physics={k:f[k].copy() for k in ('state','lock','signal_to_added_noise_db')}
    records=[];count=None
    for spec in specs:
        with np.load(spec['root']/'source_qualification/predictions.npz',allow_pickle=False) as f:current={k:f[k].copy() for k in parent}
        result=parent_cohort_arrays(parent,current,physics);count=result['parent_clean_correct_count']
        records.extend(dict(row_id=spec['key'],own_As20_parent=spec['arm']=='B01',**r) for r in result['records'])
    c.csv_write(out/'P43_As20_clean_cohort.csv',records)
    result=dict(status='CACHED_P43_AS20_COHORT_ANALYZED',checkpoint_run=REF,checkpoint_row='B01-s'+str(SEED),
        checkpoint_key='model',checkpoint_epoch=20,parent_clean_correct_count=count,source_unique_V=1800,
        channel_draws=2,model_seed_count=1,NN=0,IQ_reads=0,checkpoint_reads=0,
        NoAs20_own_parent_cohort_available=False,formal_parent20_cohort_available=False,
        NoAs_rows_are_comparators_on_As20_cohort_not_their_own_parent=True,
        fixed_NoAs30_cohort_is_reported_separately=True)
    c.write(out/'parent_cohort_summary.json',result)
    return result


def costs(project,specs,local_nn=0,smoke_nn=8):
    c.require(local_nn==0 and type(smoke_nn) is int and smoke_nn>=0,'Actual nonnegative shared smoke/local0 required')
    rows={};new={}
    for spec in specs:
        tail=c.native_cost(spec['root']/'source');source=c.read(spec['root']/'source_qualification/complete.json');prediction=c.read(spec['root']/'prediction/complete.json')
        c.require(source.get('NN')==12600 and prediction.get('actual_NN')==1176002,'Actual source/formal forward counts differ')
        record=dict(parent=c.native_cost(spec['parent']),tail=tail,source_NN=12600,formal_NN=1176002,
            source_wall_seconds=source.get('wall_seconds'),formal_wall_seconds=prediction.get('wall_seconds'))
        rows[spec['key']]=record
        if not spec['reused']:new[spec['key']]=record
    smoke=c.read(Path(project)/'runs'/RUN/'real_checkpoint_smoke.json')
    c.require(smoke.get('status')=='PASS' and smoke.get('NN')==smoke_nn,'Actual shared no-query smoke differs')
    subtotal=sum(r['tail']['training_total_NN']+r['source_NN']+r['formal_NN'] for r in new.values())
    c.require(len(new)==2 and subtotal==8703128 and all(r['tail']['native_NN']==3162962 and
        r['tail']['extra_student_U_NN']==r['tail']['extra_frozen_teacher_U_NN']==0 for r in new.values()),'Two original-native-only tails required')
    return dict(rows=rows,new_native_NN=6325924,new_source_NN=25200,new_formal_NN=2352004,
        shared_remote_smoke_NN=smoke_nn,new_total_remote_NN=subtotal+smoke_nn,local_verification_NN=0,
        new_model_epochs=20,new_updates=4440,unique_shared_parent_model_epochs=40,unique_shared_parent_updates=8880,
        unique_total_model_epochs_including_reused=80,unique_total_updates_including_reused=17760,
        student_cumulative_epochs_each=30,student_cumulative_updates_each=6660,parent_training_reused_not_repeated=True,
        new_training_wall_seconds_sum=sum(r['tail']['wall_seconds'] for r in new.values()),
        extra_student_NN=0,extra_teacher_NN=0,reference_reprediction=0,reference_rescoring=0,analysis_NN=0,
        historical_parent_source_cohort_NN=12602,parent_cohort_new_NN=0,budget_exhaustion_is_not_convergence=True)


def plots(out,curves,source_rows,formal,training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(ARMS,('#64748b','#d97706','#8b5cf6','#2563eb')));files=[]
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):
            path=out/(name+'.'+ext);fig.savefig(path,dpi=180);files.append(str(path))
        plt.close(fig)
    if curves:
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for arm in ARMS:
            rr=[r for r in curves if r['row_id']==arm]
            for ax,key,title in zip(axes,('native_loss','source_val_accuracy_pct','grad_backbone'),('Native loss','Source V accuracy (%)','Backbone gradient norm')):
                ax.plot([r['epoch'] for r in rr],[r[key] for r in rr],label=arm,color=colors[arm]);ax.set(title=title,xlabel='Cumulative epoch');ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);save(fig,'full_training')
    if source_rows:
        fig,axes=plt.subplots(2,4,figsize=(20,8))
        for si,scene in enumerate(c.SCENES[1:]):
            for di,dim in enumerate(('state','lock','receiver_quality_proxy_db','signal_to_added_noise_db')):
                ax=axes[si,di];rr=[r for r in source_rows if r['scene']==scene and r['draw']==-1 and r['dimension']==dim];labels=list(dict.fromkeys(r['stratum'] for r in rr));xx=np.arange(len(labels))
                for i,arm in enumerate(ARMS):
                    z=[next(r for r in rr if r['row_id']==arm and r['stratum']==label) for label in labels]
                    ax.bar(xx+(i-1.5)*.2,[100*r['accuracy'] if r['accuracy'] is not None else np.nan for r in z],.2,label=arm,color=colors[arm])
                ax.set(xticks=xx,xticklabels=labels,title=scene+' / '+dim,ylabel='Source accuracy (%)');ax.tick_params(axis='x',labelrotation=25);ax.grid(axis='y',alpha=.2)
        axes[0,0].legend(fontsize=8);save(fig,'source_state_quality_lock_noise')
    fig,axes=plt.subplots(1,2,figsize=(15,5));xx=np.arange(7)
    for new,old,label in (('AS_KEEP','NOAS_KEEP','As effect / keep'),('AS_KEEP','AS30','Keep effect / As'),('NOAS_KEEP','NOAS30','Keep effect / NoAs')):
        axes[0].plot(xx,[100*(formal['rows'][new][v]['accuracy']-formal['rows'][old][v]['accuracy']) for v in c.VIEWS],marker='.',label=label)
    axes[1].bar(xx,[formal['interactions'][v]['accuracy_interaction_pp'] for v in c.VIEWS],color='#2563eb')
    for ax,title in zip(axes,('Formal paired changes','Optimizer x As: B11-B10-B01+B00')):
        ax.set(xticks=xx,xticklabels=['clean','high sub','mid sub','low sub','high urb','mid urb','low urb'],title=title,ylabel='Absolute percentage points');ax.tick_params(axis='x',labelrotation=30);ax.axhline(0,color='grey',lw=.7);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);save(fig,'formal_paired_interaction')
    if training:
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for arm in NEW_ARMS:
            records=training['rows'][arm]['optimizer_readout']['epochs']
            for ax,key,title in zip(axes,('state_entries','step_max','exp_avg_l2'),('AdamW state entries','Largest stored AdamW step','First moment norm (not accuracy)')):
                ax.plot(range(21,31),[optimizer_plot_value(r,key) for r in records],label=arm,color=colors[arm]);ax.set(title=title,xlabel='Cumulative epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
        save(fig,'optimizer_state_readout')
    return files


def optimizer_plot_value(record,key):
    return c.finite(record[key],key)


def report(out,result):
    f=result['formal'];rows=f['rows'];cost=result['cost'];stage=f['stage']['AS_KEEP']
    text=['# P69：固定学习率日程的 AdamW 状态继承','',
        '两个新尾段分别严格继承原 P43 NoAs20/As20 student 和 ownEMA；只保留全模型 AdamW 的一阶矩、二阶矩及各参数 step，新段学习率仍从0.0002按原cos10降至0.000001。原Sinc单位/clip、普通增强、同IQ RDrop、ORD128 SupCon、native U、6L及默认cosine30头保持。原NoAs30/As30为相同父来源、清空AdamW状态的既存对照。', '',
        '源判据仅两项：AS_KEEP对NOAS_KEEP fullV clean≥−0.1绝对百分点；pooled mid/low对原As30≥+0.5百分点。source verdict在正式读取前固定，任一失败不扩E50，但全部七环境仍完成；本分析不按正式结果晋级。','',
        '| 环境 | NoAs reset % | As reset % | NoAs keep % | As keep % | As×keep交互 pp |','|---|---:|---:|---:|---:|---:|']
    for v in c.VIEWS:text.append('| '+v+' | '+' | '.join(f'{100*rows[a][v]["accuracy"]:.6f}' for a in ARMS)+f' | {f["interactions"][v]["accuracy_interaction_pp"]:+.6f} |')
    text += ['',f'阶段比较 AS_KEEP−NOAS_KEEP：clean {stage["clean_delta_pp"]:+.6f} pp，mid urban {stage["mid_urban_delta_pp"]:+.6f} pp，low urban {stage["low_urban_delta_pp"]:+.6f} pp，urban均值 {stage["urban_mean_delta_pp"]:+.6f} pp；clean下降≤0.1pp：{stage["clean_within_0p1pp"]}。',
        '四格交互、AS_KEEP−AS30与NOAS_KEEP−NOAS30分开报告。只有一个模型seed，两个源信道draw是重复观测；单seed或不足0.1个百分点的正变化不能证明稳定双提升。AdamW状态规模、moment范数、loss变化与U接受率不等于身份收益或伪标签精度。', '',
        '全部七环境 accuracy/Macro-F1/最差RX、RX/day/TX、修复与新增错误、margin/30、固定NoAs30 clean正确集合保持率见CSV。精确P43 As20 student源clean正确集合另列；它不是AS_CE20，也不是NoAs30或EMA。缓存缺失不新增推理，不伪造父集合。源域state/lock/质量/added-noise分层是关联，正式逐包状态未保存；新增噪声比不是总SNR。', '',
        f'父来源共40 model-epochs复用；四条尾段共40，其中本次新增20／4440更新，因此独有总训练80 model-epochs／17760更新。新增native NN={cost["new_native_NN"]}，source={cost["new_source_NN"]}，formal={cost["new_formal_NN"]}，共享smoke={cost["shared_remote_smoke_NN"]}，远端总计={cost["new_total_remote_NN"]}，本地NN=0；新增训练行耗时之和={cost["new_training_wall_seconds_sum"]:.3f}秒。父源cohort历史12602NN单列，本分析复用新增0NN。预算耗尽不等于收敛。','',
        '已存源判定：'+json.dumps(result['stored_source_decision'],ensure_ascii=False),'',
        '未完成／限制：'+json.dumps(result['missing'],ensure_ascii=False),'','实际图表：']
    text += ['- '+p for p in result['figures']]
    text += ['','分析本身NN、IQ/checkpoint读取、训练更新、重预测与重新评分均为0。']
    (out/'final_analysis.md').write_text('\n'.join(text)+'\n',encoding='utf8')


def run(project,out,truth=None,code_commit=None,local_nn=0,smoke_nn=8):
    project=Path(project);decision=read_source_decision(project)
    out=Path(out);c.require(not out.exists(),'Immutable analysis output required');out.mkdir(parents=False)
    c.write(out/'stored_source_decision.json',decision);started=time.perf_counter();specs=row_specs(project)
    missing=[];training=None;curves=[];source=None;source_rows=[];parent=None
    if decision['status'] in c.DECISION_PARTIAL:missing.append(dict(component='source_decision',error='Stored source decision PARTIAL; not recomputed'))
    for name in ('training','source','parent_cohort'):
        try:
            if name=='training':training,curves=training_analysis(project,specs,out)
            elif name=='source':
                source,source_rows=c.source_analysis(project,specs,out,decision)
                q.interaction_strata(source_rows,out,'source_interactions.csv')
            else:parent=parent_cohort_analysis(project,specs,out)
        except Exception as error:
            failure=dict(component=name,error=repr(error),traceback=traceback.format_exc(),formal_still_required=True)
            missing.append(failure);c.write(out/(name+'_failure.json'),failure)
    truth=Path(truth) if truth else project/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
    formal=c.formal_analysis(project,specs,out,truth);q.interaction_strata(c.read(out/'formal_confusions.json'),out,'formal_interactions.csv')
    cost=costs(project,specs,local_nn,smoke_nn);c.write(out/'costs.json',cost)
    try:figures=plots(out,curves,source_rows,formal,training)
    except Exception as error:
        figures=[];failure=dict(component='plots',error=repr(error),traceback=traceback.format_exc());missing.append(failure);c.write(out/'plot_failure.json',failure)
    result=dict(status='ANALYZED' if not missing else 'FORMAL_ANALYZED_WITH_PARTIAL_COMPONENTS',run_id=RUN,
        code_commit=code_commit,stored_source_decision=decision,training=training,source=source,parent_cohort=parent,
        formal=formal,cost=cost,figures=figures,missing=missing,model_seed_count=1,crossseed_confirmed=False,
        reproducible_dual_gain=False,target_feedback=False,source_decision_recomputed=False,goal_complete=False,
        NN=0,IQ_reads=0,checkpoint_reads=0,training_updates=0,wall_seconds=time.perf_counter()-started)
    report(out,result);c.write(out/'summary.json',{k:v for k,v in result.items() if k!='training'});c.write(out/'complete.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--truth',type=Path)
    parser.add_argument('--code-commit');parser.add_argument('--local-nn',type=int,default=0);parser.add_argument('--smoke-nn',type=int,default=8)
    parser.add_argument('--checks',action='store_true');args=parser.parse_args()
    if args.checks:
        path=Path(__file__).with_name('cvs_optimizer_continuity69_analysis_checks_20261018.py')
        spec=importlib.util.spec_from_file_location('_p69_analysis_checks',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.checks();return
    if args.project is None or args.out is None:parser.error('--project and --out required unless --checks')
    result=run(args.project,args.out,args.truth,args.code_commit,args.local_nn,args.smoke_nn)
    print(json.dumps(dict(status=result['status'],out=str(args.out),missing=[r['component'] for r in result['missing']],NN=0,IQ_reads=0,checkpoint_reads=0)),flush=True)


if __name__=='__main__':main()
