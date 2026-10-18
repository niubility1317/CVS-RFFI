"""P77 saved-output analysis: physical-quality target entropy versus permutation.

No Torch import, IQ/checkpoint read, inference, optimizer update or scorer launch.
The source producer's immutable decision is read before any formal artifacts.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
import argparse
import importlib.util
import json
import math
import time
import traceback
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np

RUN='20261019-phase1-FcrossRX-qualityEntropy-e30-s2026092701-m2-r01'
REFERENCE='20261019-phase1-FcrossRX-propContrast-e30-s2026092701-m4-r01'
REF='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
PARENT_RUN='20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01'
PARENT_COHORT_RUN='20261018-diagnostic-FcrossRX-ASCE20cohort-sourceV-s2026092701-m1-r01'
SEED,STEPS,UPDATES,PER_VIEW=2026092701,222,2220,64
ARMS=('NOAS30','AS30','C00','R','Q');NEW_ARMS=('R','Q');PRIMARY=('Q','R')
PAIRS=(PRIMARY,('Q','C00'),('R','C00'),('Q','AS30'),('Q','NOAS30'),
       ('R','AS30'),('R','NOAS30'),('C00','AS30'),('C00','NOAS30'),('AS30','NOAS30'))


def load_helpers():
    path=Path(__file__).with_name('cvs_prop_contrast76_analysis_20261019.py')
    spec=importlib.util.spec_from_file_location('_p77_existing76',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


p=load_helpers();q=p.q;h=p.h;c=p.c;_write=p._write


def row_specs(project):
    runs=Path(project)/'runs';rows=[]
    for key in ARMS:
        shared=key not in ('NOAS30','AS30')
        arm=('B00' if key=='NOAS30' else 'B01') if not shared else key
        run=REF if not shared else REFERENCE if key=='C00' else RUN
        parent_run=PARENT_RUN if shared else REF;parent_arm='AS_CE' if shared else arm
        rows.append(dict(key=key,run=run,arm=arm,row_id=arm+'-s'+str(SEED),root=runs/run/(arm+'-s'+str(SEED)),
            reused=key not in NEW_ARMS,parent=runs/parent_run/(parent_arm+'-s'+str(SEED))/'stage20',
            parent_run=parent_run,parent_row_id=parent_arm+'-s'+str(SEED),parent_As_KL_off=shared))
    return rows


training_complete=h.training_complete


def source_criteria(values):
    clean=100*(values['Q']['fullV_clean']-values['NOAS30']['fullV_clean'])
    best=max(values[a]['midlow'] for a in ('R','C00','AS30'))
    gain=100*(values['Q']['midlow']-best)
    checks=dict(clean_within_NoAs_0p1pp=clean>=-.1-1e-12,
        midlow_vs_max_all_controls_at_least_0p5pp=gain>=.5-1e-12)
    return dict(checks=checks,source_qualified=all(checks.values()),clean_delta_pp=clean,midlow_gain_pp=gain)


def read_source_decision(project):
    value=c.read(Path(project)/'runs'/RUN/'source_matrix_decision.json')
    c.require(value.get('status') in c.DECISION_COMPLETE|c.DECISION_PARTIAL and
        value.get('formal_scores_read') is False and value.get('target_feedback') is False and
        value.get('NN')==0 and value.get('model_seed_count')==1,'Stored source verdict must precede formal readout')
    if value['status'] in c.DECISION_COMPLETE:
        checks=value.get('checks');c.require(isinstance(checks,dict) and len(checks)==2 and
            all(type(v)is bool for v in checks.values()) and type(value.get('source_qualified'))is bool and
            value['source_qualified']==all(checks.values()),'Exactly two stored source checks and verdict required')
    return value


def enrich_formal(obj):
    rows=obj['rows'];obj.pop('all4_rows_all7_verified_before_truth',None)
    obj['all5_rows_all7_verified_before_truth']=True
    obj['comparisons']={a+'__'+b:q.effects(rows,a,b) for a,b in PAIRS}
    obj['primary']=dict(new='Q',control='R',same_parent=True,same_epsilon_multiset=True,
        attribution='Physical-quality/target-entropy association versus within-As permutation',effects=q.effects(rows,'Q','R'))
    obj['secondary']=dict(new='Q',control='C00',same_parent=True,
        attribution='Overall entropy distribution plus physical association',effects=q.effects(rows,'Q','C00'))
    obj['comparison_context']=[dict(new=a,control=b,same_parent=a in ARMS[2:] and b in ARMS[2:],
        first20_As_KL_history_confounded=a in ARMS[2:] and b=='AS30',
        matched_noAs_parent=False if a in ARMS[2:] and b=='NOAS30' else None) for a,b in PAIRS]
    obj['suburban_gap_from_own_clean_pp']={a:{v:100*(rows[a]['clean']['accuracy']-rows[a][v]['accuracy']) for v in c.VIEWS[1:4]} for a in ARMS}
    obj['urban_accuracy_mean']={a:float(np.mean([rows[a][v]['accuracy'] for v in c.VIEWS[-3:]])) for a in ARMS}
    for a in NEW_ARMS:obj['stage'][a].update(NoAs30_is_whole_recipe_reference=True,same_parent_NoAs_control_available=False,reproducible_dual_gain=False)
    obj.update(Ag_claim=False,epsilon_assignment_is_not_observability_proof=True,actual_gradient_norms_matched=False,
        formal_channel_changed=False,model_seed_count=1,reproducible_dual_gain=False)


def write(path,obj):
    if Path(path).name=='formal_summary.json':enrich_formal(obj)
    _write(path,obj)


c.RUN,c.NEW_ARMS,c.PAIRS=RUN,NEW_ARMS,PAIRS
c.training_complete,c.row_specs,c.write=training_complete,row_specs,write


def epsilon(db):
    """The fixed native added-noise ratio law; no total-SNR interpretation."""
    if db=='+inf':return .01
    if db=='-inf':return .20
    value=c.finite(db,'signal_to_added_noise_db')
    power=value*math.log(10.)/10.
    # Stable evaluation of 1/(1+exp(power)); finite inputs never overflow.
    fraction=math.exp(-power)/(1.+math.exp(-power)) if power>=0 else 1./(1.+math.exp(power))
    return .01+.19*fraction


def numeric_vector(value,n,name,integer=False):
    c.require(isinstance(value,list) and len(value)==n,'Complete vector required '+name)
    if integer:c.require(all(type(x)is int for x in value),'Integer vector required '+name)
    else:c.require(all(type(x)in(int,float) and math.isfinite(x) for x in value),'Finite vector required '+name)
    return np.asarray(value,dtype=np.int64 if integer else np.float64)


def resolved_native_snr(observation):
    c.require('signal_to_added_noise_db' in observation,'Native added-noise ratio field is required')
    value=observation['signal_to_added_noise_db']
    if value is not None:
        if value in ('+inf','-inf'):return value
        c.finite(value,'native signal_to_added_noise_db');return value
    power=c.finite(observation.get('measured_added_noise_power_pre_agc'),'native added-noise power')
    c.require(power>=0,'Native added-noise power cannot be negative')
    return '-inf' if power>0 else '+inf'


def native_snr_kind(observation):
    value=observation['signal_to_added_noise_db'];resolved=resolved_native_snr(observation)
    if value is None:return 'zero_signal' if resolved=='-inf' else 'zero_added_noise'
    if resolved=='+inf':return 'explicit_positive_infinity'
    if resolved=='-inf':return 'explicit_negative_infinity'
    return 'finite'


def entropy_record(record,physics,arm,step,packets=PER_VIEW):
    """Independently verify one saved assignment and scalar CE replacement."""
    c.require(record is not None and physics is not None and arm in NEW_ARMS and
        record.get('step')==physics.get('step')==step and
        record.get('cumulative_step')==physics.get('cumulative_step')==step+4440 and
        record.get('method')==physics.get('arm')==arm,'Complete target-entropy step/physics identity required')
    entropy=record.get('entropy');assignment=physics.get('entropy')
    c.require(isinstance(entropy,dict) and isinstance(assignment,dict) and
        entropy.get('mode')==assignment.get('mode')==arm,'Saved entropy objective/assignment required')
    original=numeric_vector(assignment.get('epsilon_original'),packets,'epsilon_original')
    assigned=numeric_vector(assignment.get('epsilon_assigned'),packets,'epsilon_assigned')
    permutation=numeric_vector(assignment.get('permutation'),packets,'permutation',True)
    c.require(np.array_equal(np.sort(permutation),np.arange(packets)) and
        np.all((original>=.01-1e-9)&(original<=.20+1e-9)) and
        np.all((assigned>=.01-1e-9)&(assigned<=.20+1e-9)),'Bounded epsilon and exact permutation required')
    db=assignment.get('signal_to_added_noise_db');first=physics.get('view0')
    c.require(isinstance(db,list) and isinstance(first,list) and len(db)==len(first)==packets,'Every native As observation needs its actual added-noise ratio')
    observed=[resolved_native_snr(r) for r in first]
    c.require(observed==db,'Entropy may use only the actual engine added-noise ratio including its zero-power limit')
    if 'signal_to_added_noise_db_native' in assignment:
        c.require(assignment['signal_to_added_noise_db_native']==[r['signal_to_added_noise_db'] for r in first],
            'Raw native ratio record must remain unchanged')
    kinds=[native_snr_kind(r) for r in first]
    c.require(assignment.get('snr_boundary_kinds')==kinds,'Every legal zero-power limit must retain all CE samples')
    boundary=dict(snr_finite_count=kinds.count('finite'),snr_explicit_infinite_count=sum(k.startswith('explicit_') for k in kinds),
        snr_zero_signal_count=kinds.count('zero_signal'),snr_zero_added_noise_count=kinds.count('zero_added_noise'))
    c.require(sum(boundary.values())==packets,'Native added-noise boundary counts must include every As slot')
    for key,value in boundary.items():
        c.require(assignment.get(key)==value and entropy.get(key)==value,'Saved target-entropy boundary count differs '+key)
    recomputed=np.asarray([epsilon(x) for x in db]);c.require(np.allclose(original,recomputed,atol=1e-8,rtol=0),'Original epsilon law differs')
    namespace=f'FcrossRX/train/e{(step-1)//STEPS+21}/step{step+4440}/L_S'
    c.require(assignment.get('namespace')==namespace and all(r.get('namespace')==namespace for r in first),'Original cumulative source namespace required')
    if arm=='Q':
        c.require(assignment.get('permutation_seed') is None and np.array_equal(permutation,np.arange(packets)),'Q must preserve physical epsilon assignment')
    else:
        seed=q.stable_seed(20261016421,namespace,'quality_entropy77/permutation')
        expected=np.random.default_rng(seed).permutation(packets)
        c.require(assignment.get('permutation_seed')==seed and np.array_equal(permutation,expected),'R needs the independent fixed label-free permutation stream')
    c.require(np.array_equal(assigned,original[permutation]) and np.array_equal(np.sort(original),np.sort(assigned)),
        'Assigned epsilon must be the exact same multiset')
    c.require(entropy.get('epsilon_multiset_preserved') is True and entropy.get('As_CE_passes')==2 and
        entropy.get('epsilon_same_both_passes') is True,'Both existing CE passes must reuse one complete epsilon assignment')
    for prefix,values in (('epsilon',original),('assigned_epsilon',assigned)):
        for suffix,expected in (('min',values.min()),('max',values.max()),('mean',values.mean()),('std',values.std())):
            # The producer may omit redundant assigned/std diagnostics; whenever
            # present they must match the saved vector, never substitute for it.
            name=prefix+'_'+suffix
            if name in entropy:c.require(abs(c.finite(entropy[name],name)-expected)<1e-7,'Saved epsilon scalar differs '+name)
    old0=c.finite(entropy.get('nominal_CE0'),'nominal As CE0');old1=c.finite(entropy.get('nominal_CE1'),'nominal As CE1')
    new0=c.finite(entropy.get('new_CE0'),'smoothed As CE0');new1=c.finite(entropy.get('new_CE1'),'smoothed As CE1')
    c.require(min(old0,old1,new0,new1)>=0,'Nonnegative true-label CE required')
    change=c.finite(entropy.get('weighted_delta'),'epsilon correction')
    c.require(abs(change-.125*(new0-old0+new1-old1))<1e-6,'Exact .25 mean-two-pass target replacement required')
    base=c.finite(entropy.get('base_total_L'),'base total L');total=c.finite(entropy.get('total_L'),'smoothed total L')
    c.require(abs(total-base-change)<3e-5,'Smoothed total L must restore the original base plus correction')
    if 'entropy_weighted_delta' in record:c.require(abs(record['entropy_weighted_delta']-change)<1e-7,'Top-level correction disagrees')
    packet_rows=[]
    for i,(obs,a,b) in enumerate(zip(first,original,assigned)):
        pid=obs['sample_id'];value=db[i]
        quality='<3' if value=='-inf' or type(value)in(int,float) and value<3 else '3-8' if type(value)in(int,float) and value<8 else '8-15' if type(value)in(int,float) and value<15 else '>=15'
        packet_rows.append(dict(TX=physics['labels_L_As'][i],RX=pid[1],day=pid[2],scene=obs['scene'],state=obs['state_end'],
            lock=obs['receiver_lock_status'],signal_to_added_noise_bin=quality,epsilon=float(a),assigned=float(b)))
    return dict(step=step,cumulative_step=step+4440,epoch=(step-1)//STEPS+21,mode=arm,
        base_total_L=base,total_L=total,loss_delta=change,old_As_CE=.5*(old0+old1),new_As_CE=.5*(new0+new1),
        old_As_CE0=old0,old_As_CE1=old1,new_As_CE0=new0,new_As_CE1=new1,
        epsilon_mean=float(original.mean()),assigned_epsilon_mean=float(assigned.mean()),epsilon_min=float(original.min()),
        epsilon_max=float(original.max()),epsilon_std=float(original.std()),assigned_epsilon_std=float(assigned.std()),
        assignment_absolute_difference_mean=float(abs(original-assigned).mean()),permutation_moved=int((permutation!=np.arange(packets)).sum()),
        multiset_exact=True,packet_rows=packet_rows,**boundary)


def epsilon_strata(records):
    grouped=defaultdict(lambda:dict(count=0,epsilon_sum=0.,assigned_sum=0.,epsilon_min=1.,epsilon_max=0.,assigned_min=1.,assigned_max=0.))
    for record in records:
        for row in record['packet_rows']:
            contexts=[('overall','ALL'),*((key,str(row[key])) for key in ('TX','RX','day','scene','state','lock','signal_to_added_noise_bin')),
                ('scene_state_lock',row['scene']+'/'+row['state']+'/'+row['lock'])]
            for context in contexts:
                g=grouped[context];g['count']+=1;g['epsilon_sum']+=row['epsilon'];g['assigned_sum']+=row['assigned']
                g['epsilon_min']=min(g['epsilon_min'],row['epsilon']);g['epsilon_max']=max(g['epsilon_max'],row['epsilon'])
                g['assigned_min']=min(g['assigned_min'],row['assigned']);g['assigned_max']=max(g['assigned_max'],row['assigned'])
    return [dict(dimension=key[0],stratum=key[1],**value,epsilon_mean=value['epsilon_sum']/value['count'],
        assigned_epsilon_mean=value['assigned_sum']/value['count']) for key,value in sorted(grouped.items())]


def matched_assignments(first,second,expected_steps=UPDATES):
    """Q/R must have the same original epsilons and native physical metadata."""
    steps=0
    for steps,(a,b) in enumerate(zip_longest(first,second),1):
        c.require(a is not None and b is not None and a.get('step')==b.get('step')==steps,
            'Both complete Q/R physics streams required')
        c.require(all(a.get(key)==b.get(key) for key in ('view0','view1','labels_L_As','group_stats')),
            'Q/R physical observations/labels must match before epsilon assignment')
        c.require(all(a['entropy'].get(key)==b['entropy'].get(key) for key in
            ('epsilon_original','signal_to_added_noise_db','namespace')),'Q/R original epsilon multiset differs')
    c.require(steps==expected_steps,'All Q/R original assignment steps required')
    return dict(steps=steps,exact_original_epsilon=True,exact_native_metadata=True,NN=0)


def mapped_physics(records,arm):
    for record in records:
        c.require(record.get('arm')==arm,'Actual physics row arm differs')
        yield dict(record,arm='C00')


def mapped_prop(records,arm):
    for record in records:
        c.require(record.get('method')==arm,'Actual prop row arm differs')
        yield dict(record,method='C00')


def parse_tail(folder,is_as,arm=None):
    if arm in ('NOAS30','AS30'):return h.parse_tail(folder,is_as)
    if arm=='C00':return p.parse_tail(folder,True,'C00')
    c.require(arm in NEW_ARMS,'Only fixed five logical rows')
    folder=Path(folder);records=c.lines(folder/'prop_contrast_steps.jsonl.gz')
    physics=c.lines(folder/'prop_contrast_physics.jsonl.gz')
    checked=[entropy_record(r,v,arm,i) for i,(r,v) in enumerate(zip_longest(records,physics),1)]
    c.require(len(checked)==UPDATES,'All2220 target-entropy records required')
    by_step={r['step']:r for r in checked};original_lines=c.lines
    def adapted(path):
        for original in original_lines(path):
            if Path(path)==folder/'reference_steps.jsonl.gz':
                e=by_step[original['step']]
                c.require(abs(original['total_L']-e['total_L'])<3e-5,'Native smoothed total must equal entropy record')
                yield dict(original,total_L=original['total_L']-e['loss_delta'])
            else:yield original
    try:
        c.lines=adapted;tail,ordinary=h.parse_tail(folder,True)
    finally:c.lines=original_lines
    for epoch in tail['curve']:
        rr=checked[(epoch['local_epoch']-1)*STEPS:epoch['local_epoch']*STEPS]
        epoch['base_total_L']=epoch['total_L']
        for key in ('loss_delta','old_As_CE','new_As_CE','epsilon_mean','assigned_epsilon_mean'):
            epoch[key]=float(np.mean([r[key] for r in rr]))
        epoch['total_L']+=epoch['loss_delta']
    strata=epsilon_strata(checked)
    c.require(next(r['count'] for r in strata if r['dimension']=='overall')==142080,'Complete assignment coverage required')
    for row in checked:row.pop('packet_rows')
    tail.update(smoothed_total_L_restored=True,base_decomposition_verified_separately=True,
        entropy_records=checked,epsilon_strata=strata,actual_parameter_gradient_norms_matched=False)
    return tail,ordinary


def training_analysis(project,specs,out):
    tails={};curves=[];ordinary={};by_key={s['key']:s for s in specs}
    for spec in specs:
        arm=spec['key'];folder=spec['root']/'source';tail,ordinary[arm]=parse_tail(folder,arm!='NOAS30',arm)
        if arm in ARMS[2:]:
            result=c.read(folder/'result.json')
            records=c.lines(folder/'prop_contrast_steps.jsonl.gz');physics=c.lines(folder/'prop_contrast_physics.jsonl.gz')
            if arm in NEW_ARMS:records,physics=mapped_prop(records,arm),mapped_physics(physics,arm)
            proof,steps=p.prop_recount(records,physics,'C00')
            saved=dict(result['prop_contrast_summary'],arm='C00');p.validate_summary(saved,proof,result)
            tail['propagation_physics']=proof
            c.csv_write(out/(arm+'_physical_strata.csv'),proof['physical_strata'])
            c.csv_write(out/(arm+'_labeled_coverage.csv'),proof['label_coverage'])
            c.csv_write(out/(arm+'_group_coverage.csv'),proof['group_TX_RX_day_coverage'])
        for r in c.epoch_curve(spec['parent'],0)+tail['curve']:
            curves.append(dict(row_id=arm,model_seed=SEED,reused=spec['reused'],parent_reused=True,**r))
        if arm in NEW_ARMS:
            c.csv_write(out/(arm+'_entropy_steps.csv'),tail.pop('entropy_records'))
            c.csv_write(out/(arm+'_epsilon_strata.csv'),tail['epsilon_strata'])
        tails[arm]=tail
    sequences={}
    for arm in NEW_ARMS:
        sequences[arm]=h.same_sequence(by_key[arm]['root']/'source/training_sequence.jsonl.gz',by_key['C00']['root']/'source/training_sequence.jsonl.gz')
        c.require(ordinary[arm]==ordinary['C00'],'Original ordinary input stream must match C00')
    assignment_match=matched_assignments(c.lines(by_key['Q']['root']/'source/prop_contrast_physics.jsonl.gz'),
        c.lines(by_key['R']['root']/'source/prop_contrast_physics.jsonl.gz'))
    c.require(len(curves)==150,'Five complete logical E30 curves required')
    c.csv_write(out/'training_curves.csv',curves)
    result=dict(status='FULL_SAVED_TRAINING_VERIFIED',rows=tails,new_model_epochs=20,new_updates=4440,
        matched_L_U_sequences=sequences,matched_Q_R_original_assignment=assignment_match,
        ordinary_streams_match=True,nominal_total_L_CE_scale=1.25,As_CE_coefficient=.25,
        extra_NN=0,extra_parameters=0,As_contrast=False,U_truth_read=False,parent_As_KL_off=True,
        default_head_joint=True,actual_parameter_gradient_norms_matched=False)
    c.write(out/'training_summary.json',result);return result,curves


def pooled_source(records):
    pooled={}
    for arm in ARMS:
        rr=[r for r in records if r['row_id']==arm and r['scene'] in c.SCENES[1:] and r['draw']==-1 and r['dimension']=='overall']
        c.require(len(rr)==2,'Both pooled source urban scenes required')
        cm=sum((np.asarray(r['confusion']) for r in rr),np.zeros((6,6),int));den=cm.sum(0)+cm.sum(1)
        f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0);n=int(cm.sum());nc=sum(r['fixed_NOAS30_clean_correct_count'] for r in rr)
        rx=[]
        for receiver in c.RX:
            cells=[r for r in records if r['row_id']==arm and r['scene'] in c.SCENES[1:] and r['draw']==-1 and r['dimension']=='RX' and r['stratum']==str(receiver)]
            rx.append(sum(r['correct'] for r in cells)/sum(r['count'] for r in cells))
        pooled[arm]=dict(count=n,accuracy=float(np.trace(cm)/n),macro_f1=float(f.mean()),worst_RX=min(rx),
            mean_margin_over30=sum(r['mean_margin_over30']*r['count'] for r in rr)/n,
            satellite_retention=sum(r['fixed_NOAS30_clean_correct_retained'] for r in rr)/nc if nc else None)
    return dict(rows=pooled,unique_source_physical=1800,repeated_channel_draws=2,model_seed_count=1,verdict_recomputed=False)


def validate_parent_cohort_metadata(meta,project):
    project=Path(project);parent=project/'runs'/PARENT_RUN/'AS_CE-s2026092701/stage20/final_ssdg.pth'
    panel=project/'runs'/c.PANEL_RUN/'diagnostic'
    exact=dict(status='SOURCE_READOUT_COMPLETE',run_id=PARENT_COHORT_RUN,model_seed=SEED,checkpoint=str(parent),
        checkpoint_run=PARENT_RUN,checkpoint_row='AS_CE-s2026092701',checkpoint_stage='stage20',checkpoint_key='model',
        checkpoint_epoch=20,checkpoint_optimizer_steps=4440,checkpoint_selection='final_only',
        source_contract=str(project/'runs'/c.CONTRACT_RUN/'source_contract.json'),source_role='V',source_unique_V=1800,
        source_role_counts=dict(L_s=6300,U_s=56700,V=27000),source_panel=str(panel),channel_draws=2,
        scenes=list(c.SCENES),condition='full',condition_index=0,source_NN=12600,smoke_NN=2,NN=12602,
        training_updates=0,head_fit_updates=0,teacher_forwards=0,U_truth_read=False,U_records_used=0,
        target_access=False,formal_scores_read=False,query_access=False,model_state_unchanged=True,
        all_parameters_frozen=True,default_head_shape=[6,160],default_head_scale=30.)
    c.require(all(meta.get(key)==value for key,value in exact.items()),'Actual frozen AS_CE20 source cohort provenance differs')


def cohort_metric(y,parent_logits,current_logits,cohort):
    y=np.asarray(y);mask=np.asarray(cohort,bool);parent=np.asarray(parent_logits);current=np.asarray(current_logits)
    c.require(parent.shape==current.shape==(len(y),6) and mask.shape==y.shape and np.isfinite(parent).all() and np.isfinite(current).all(),
        'Finite aligned actual-parent and current logits required')
    a=parent.argmax(1)==y;b=current.argmax(1)==y;n=int(mask.sum())
    other=current.copy()
    if len(y):other[np.arange(len(y)),y]=-np.inf;margin=(current[np.arange(len(y)),y]-other.max(1))/30.
    else:margin=np.asarray([],float)
    return dict(count=len(y),parent_clean_correct_count=n,parent_retained=int((a&mask).sum()),current_retained=int((b&mask).sum()),
        parent_retention=float(a[mask].mean()) if n else None,current_retention=float(b[mask].mean()) if n else None,
        retention_delta_pp=100*float(b[mask].mean()-a[mask].mean()) if n else None,
        repairs_within_parent_cohort=int((~a&b&mask).sum()),new_errors_within_parent_cohort=int((a&~b&mask).sum()),
        mean_margin_over30_within_parent_cohort=float(margin[mask].mean()) if n else None,empty_cohort=n==0,small_cohort=n<30)


def parent_source_cohort(project,specs,out):
    """Existing real parent predictions only; missing cache never changes promotion."""
    project=Path(project);root=project/'runs'/PARENT_COHORT_RUN;meta=c.read(root/'complete.json')
    validate_parent_cohort_metadata(meta,project)
    def arrays(path):
        with np.load(path,allow_pickle=False) as f:return {key:f[key].copy() for key in ('ids','TX','RX','day','clean','satellite')}
    parent=arrays(root/'predictions.npz');n=1800
    role=c.read(project/'runs'/c.PANEL_RUN/'diagnostic/data_roles.json')
    c.require(np.array_equal(parent['ids'],np.asarray(role['physical_ids'])) and len(set(parent['ids'].tolist()))==n and
        parent['clean'].shape==(n,6) and parent['satellite'].shape==(2,3,n,6) and
        np.isfinite(parent['clean']).all() and np.isfinite(parent['satellite']).all(),'Actual parent panel IDs/logit shape differs')
    for key,allowed in (('TX',range(6)),('RX',c.RX),('day',c.DAYS)):
        c.require(parent[key].shape==(n,) and parent[key].dtype.kind in 'iu' and set(parent[key])==set(allowed),
            'Actual parent role labels/RX/day differ '+key)
    cohort=parent['clean'].argmax(1)==parent['TX'];c.require(int(cohort.sum())==meta.get('parent_clean_correct_count'),
        'Parent clean-correct cohort does not recount')
    c.require(abs(float(cohort.mean())-meta['clean_accuracy'])<1e-12 and
        np.allclose((parent['satellite'].argmax(-1)==parent['TX']).mean(-1),meta['source_accuracies'],atol=1e-12,rtol=0),
        'Saved actual-parent accuracy does not recount')
    data={spec['key']:arrays(spec['root']/'source_qualification/predictions.npz') for spec in specs}
    for values in data.values():
        c.require(all(np.array_equal(values[key],parent[key]) for key in ('ids','TX','RX','day')),
            'Actual parent and all source readouts must share exact physical IDs/labels/RX/day')
        c.require(values['clean'].shape==(n,6) and values['satellite'].shape==(2,3,n,6) and
            np.isfinite(values['clean']).all() and np.isfinite(values['satellite']).all(),'Candidate source logits differ from actual-parent panel')
    with np.load(project/'runs'/c.PANEL_RUN/'diagnostic/observations/physics.npz',allow_pickle=False) as f:
        physics={key:f[key].copy() for key in ('state','lock','signal_to_added_noise_db')}
    rows=[]
    for scene in ('clean',*c.SCENES):
        si=c.SCENES.index(scene) if scene!='clean' else None
        for draw in ((-1,) if scene=='clean' else (-1,0,1)):
            draws=(0,1) if draw==-1 else (draw,);copies=1 if scene=='clean' else len(draws)
            y=np.tile(parent['TX'],copies);cc=np.tile(cohort,copies);rx=np.tile(parent['RX'],copies);day=np.tile(parent['day'],copies)
            zparent=parent['clean'] if scene=='clean' else np.concatenate([parent['satellite'][d,si] for d in draws])
            masks=[('overall','ALL',np.ones(len(y),bool))]
            for dimension,values,allowed in (('RX',rx,c.RX),('day',day,c.DAYS),('TX',y,range(6))):masks.extend((dimension,str(v),values==v) for v in allowed)
            if scene!='clean':
                states=np.concatenate([physics['state'][d,si,0] for d in draws]);locks=np.concatenate([physics['lock'][d,si,0] for d in draws]);snr=np.concatenate([physics['signal_to_added_noise_db'][d,si,0] for d in draws])
                masks.extend(('state',s,states==i) for i,s in enumerate(c.STATES));masks.extend(('lock',s,locks==i) for i,s in enumerate(c.LOCKS))
                masks.extend(('signal_to_added_noise_db',label,(snr>=lo)&(snr<hi)) for label,lo,hi in (('<3',-np.inf,3),('3-8',3,8),('8-15',8,15),('>=15',15,np.inf)));masks.append(('signal_to_added_noise_db','MISSING',np.isnan(snr)))
            for arm,values in data.items():
                z=values['clean'] if scene=='clean' else np.concatenate([values['satellite'][d,si] for d in draws])
                for dimension,label,mask in masks:rows.append(dict(row_id=arm,scene=scene,draw=draw,dimension=dimension,stratum=label,
                    cohort_reference='Actual frozen SAT18 AS_CE20 student clean-correct; distinct from NoAs30 cohort',
                    **cohort_metric(y[mask],zparent[mask],z[mask],cc[mask])))
    c.csv_write(out/'actual_parent_source_cohort.csv',rows)
    summary=dict(status='ACTUAL_PARENT_COHORT_REUSED',checkpoint=meta['checkpoint'],parent_clean_correct_count=int(cohort.sum()),
        source_unique_physical=n,channel_draws=2,model_seed_count=1,cohort_reference='SAT18 AS_CE20 student; not EMA or P43 As20',
        historical_source_readout_NN=meta['NN'],new_NN=0,source_promotion_gate=False,formal_parent_cohort_available=False,
        overview=[row for row in rows if row['dimension']=='overall'],NN=0,IQ_reads=0,checkpoint_reads=0)
    c.write(out/'actual_parent_source_cohort.json',summary);return summary


def costs(project,specs,local_nn=0,smoke_nn=1540,training=None):
    rows={};parents={};new={};history_tails={}
    for spec in specs:
        arm=spec['key'];root=spec['root'];tail=c.native_cost(root/'source');parent=c.native_cost(spec['parent'])
        source=c.read(root/'source_qualification/complete.json');prediction=c.read(root/'prediction/complete.json')
        c.require(source.get('NN')==12600 and prediction.get('actual_NN')==1176002,'Actual frozen readout counters differ')
        parents[str(spec['parent'])]=parent
        row=dict(parent=parent,tail=tail,parent_run=spec['parent_run'],parent_row_id=spec['parent_row_id'],
            source_NN=12600,formal_NN=1176002,source_wall_seconds=source.get('wall_seconds'),formal_wall_seconds=prediction.get('wall_seconds'),reused=spec['reused'])
        rows[arm]=row
        if arm in NEW_ARMS:
            c.require(tail['native_NN']==tail['training_total_NN']==3162962 and not tail['extra_student_U_NN'] and not tail['extra_frozen_teacher_U_NN'],'No extra target-entropy network calls allowed')
            if training:row.update(native_waveforms=training['rows'][arm]['propagation_physics']['native_waveforms'],
                environment_initializations=training['rows'][arm]['propagation_physics']['environment_initializations'],personal_initializations=142080)
            new[arm]=row
        else:history_tails[str(root/'source')]=tail
    smoke=c.read(Path(project)/'runs'/RUN/'real_checkpoint_smoke.json')
    c.require(smoke.get('status')=='PASS' and smoke.get('NN')==smoke_nn==1540 and local_nn==0,'Shared1540 no-query smoke/local0 required')
    total=sum(r['tail']['training_total_NN']+r['source_NN']+r['formal_NN'] for r in new.values())+smoke_nn
    c.require(total==8704668 and len(new)==2 and len(parents)==3 and len(history_tails)==3,'Exact two-tail/shared-parent cost arithmetic differs')
    historical=sum(x['training_total_NN'] for x in parents.values())+sum(x['training_total_NN'] for x in history_tails.values())+sum(r['source_NN']+r['formal_NN'] for a,r in rows.items() if a not in NEW_ARMS)
    return dict(rows=rows,unique_reused_parents=parents,new_native_NN=6325924,new_source_NN=25200,new_formal_NN=2352004,
        shared_remote_smoke_NN=1540,local_verification_NN=0,new_total_remote_NN=total,new_model_epochs=20,new_updates=4440,
        historical_reference_and_shared_parent_model_epochs=90,unique_total_model_epochs_including_reused=110,
        unique_total_updates_including_reused=24420,logical_curve_epoch_records=150,shared_parent_counted_once=True,
        historical_known_NN=historical,unique_total_known_remote_NN=historical+total,historical_shared_smokes_unallocated=True,
        student_parent_epochs=20,student_parent_updates=4440,student_cumulative_epochs=30,student_cumulative_updates=6660,
        new_training_wall_seconds_sum=sum(r['tail']['wall_seconds'] for r in new.values()),
        new_source_wall_seconds_sum=sum(r['source_wall_seconds'] for r in new.values()),
        new_formal_wall_seconds_sum=sum(r['formal_wall_seconds'] for r in new.values()),
        new_training_native_waveforms=284160,new_training_As_forward_slots=568320,
        reference_reprediction=0,reference_rescoring=0,extra_student_NN=0,extra_teacher_NN=0,analysis_NN=0,
        same_NN_is_not_same_wall=True,budget_exhaustion_is_not_convergence=True)


def plots(out,curves,source_rows,formal,training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(ARMS,('#64748b','#d97706','#8b5cf6','#16a34a','#2563eb')));files=[]
    def save(fig,name):
        fig.tight_layout()
        for extension in ('png','pdf'):
            path=out/(name+'.'+extension);fig.savefig(path,dpi=180);files.append(str(path))
        plt.close(fig)
    if curves:
        fig,axes=plt.subplots(1,3,figsize=(16,4.7))
        for arm in ARMS:
            seq=[r for r in curves if r['row_id']==arm]
            for ax,key,title in zip(axes,('source_val_accuracy_pct','native_loss','grad_backbone'),('Source V accuracy (%)','Native total loss','Backbone gradient norm')):
                ax.plot([r['epoch'] for r in seq],[r[key] for r in seq],label=arm,color=colors[arm]);ax.set(xlabel='Cumulative epoch',title=title);ax.axvline(20.5,color='grey',ls='--',alpha=.4);ax.grid(alpha=.2)
        axes[0].legend();save(fig,'training_curves')
    if source_rows:
        fig,axes=plt.subplots(2,3,figsize=(17,9))
        for si,scene in enumerate(c.SCENES[1:]):
            for di,dimension in enumerate(('state_lock','signal_to_added_noise_db','receiver_quality_proxy_db')):
                ax=axes[si,di];rr=[r for r in source_rows if r['scene']==scene and r['draw']==-1 and r['dimension']==dimension];labels=list(dict.fromkeys(r['stratum'] for r in rr));xx=np.arange(len(labels))
                for i,arm in enumerate(ARMS):
                    chosen=[next(r for r in rr if r['row_id']==arm and r['stratum']==label) for label in labels]
                    ax.bar(xx+(i-2)*.16,[100*r['accuracy'] if r['accuracy'] is not None else np.nan for r in chosen],.16,label=arm,color=colors[arm])
                ns=[next(r['count'] for r in rr if r['row_id']=='Q' and r['stratum']==label) for label in labels]
                ax.set(xticks=xx,xticklabels=[l+'\nn='+str(n) for l,n in zip(labels,ns)],title=scene+' / '+dimension,ylabel='Source accuracy (%)');ax.tick_params(axis='x',labelrotation=40);ax.grid(axis='y',alpha=.2)
        axes[0,0].legend(fontsize=7);save(fig,'source_quality_states')
    fig,axes=plt.subplots(1,2,figsize=(15,5));xx=np.arange(7)
    for arm in ARMS:axes[0].plot(xx,[100*formal['rows'][arm][v]['accuracy'] for v in c.VIEWS],marker='.',label=arm,color=colors[arm])
    for a,b in (('Q','R'),('Q','C00'),('R','C00')):axes[1].plot(xx,[100*(formal['rows'][a][v]['accuracy']-formal['rows'][b][v]['accuracy']) for v in c.VIEWS],marker='.',label=a+' - '+b)
    for ax,title in zip(axes,('Frozen default-head accuracy (%)','Paired absolute percentage points')):
        ax.set(xticks=xx,xticklabels=['clean','high sub','mid sub','low sub','high urb','mid urb','low urb'],title=title);ax.tick_params(axis='x',labelrotation=30);ax.legend();ax.grid(alpha=.2)
    axes[1].axhline(0,color='grey',lw=.7);save(fig,'formal_all7_paired')
    if training:
        fig,axes=plt.subplots(1,3,figsize=(16,5))
        for arm in NEW_ARMS:
            rr=[r for r in curves if r['row_id']==arm and r['epoch']>20]
            axes[0].plot([r['epoch'] for r in rr],[r['epsilon_mean'] for r in rr],label=arm+' original',color=colors[arm])
            axes[0].plot([r['epoch'] for r in rr],[r['assigned_epsilon_mean'] for r in rr],label=arm+' assigned',color=colors[arm],ls='--')
            for key in ('old_As_CE','new_As_CE'):axes[1].plot([r['epoch'] for r in rr],[r[key] for r in rr],label=arm+' '+key)
            axes[2].plot([r['epoch'] for r in rr],[r['loss_delta'] for r in rr],label=arm,color=colors[arm])
        for ax,title in zip(axes,('Original / assigned epsilon mean','Same logits: original / smoothed CE','Exact correction to total L')):
            ax.set(xlabel='Cumulative epoch',title=title);ax.legend(fontsize=7);ax.grid(alpha=.2)
        save(fig,'target_entropy_assignment')
    return files


def report(out,result):
    formal=result['formal'];cost=result['cost'];rows=formal['rows'];stage=formal['stage']['Q']
    text=['# P77：实际新增噪声比与目标熵的关联检验','',
        'Q/R 从同一已核实 SAT18 AS_CE20 student/own EMA 独立追加 E10，总 E30。复用 P76 C00 的按 RX/day/eq 共场景、独立传播 H 与无额外 As 对比项路径。ε=.01+.19/(1+10^(实际新增噪声比/10))；Q 直接对应物理观测，R 在同批 As64 内按独立命名流打乱同一 ε 多重集。两次既有 As 前向使用同一 IQ 和同一 assigned ε，CE 各自计算后平均；As 系数 .25、ordinary/raw 目标、原 U、ordinary SupCon .1、默认 cosine30 头和 6L/768 不变。','',
        '主比较 Q−R 控制每批 ε 的完整分布及均值；Q−C00 是平均平滑分布和质量关联的总效应。没有匹配实际参数梯度范数，不能将 R 置换控制称为纯物理因果识别。AsLS 固定 .05 的历史失败保持关闭，本轮不据正式分数选择 ε、种子或晋级。','',
        '| 环境 | NoAs % | 原As % | C00 % | R % | Q % | Q−R pp | Q−C00 pp | Q F1 % | Q 最差RX % |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for view in c.VIEWS:
        text.append('| '+view+' | '+' | '.join(f'{100*rows[a][view]["accuracy"]:.6f}' for a in ARMS)+f' | {100*(rows["Q"][view]["accuracy"]-rows["R"][view]["accuracy"]):+.6f} | {100*(rows["Q"][view]["accuracy"]-rows["C00"][view]["accuracy"]):+.6f} | {100*rows["Q"][view]["macro_f1"]:.6f} | {100*rows["Q"][view]["worst_RX"]:.6f} |')
    text+=['','Q 相对整体 NoAs30 的 clean 变化：'+f'{stage["clean_delta_vs_NoAs30_pp"]:+.6f} 个百分点；下降不超过 0.1 个百分点：{stage["clean_within_0p1pp"]}。该 NoAs 不是同 AS_CE20 父的尾段对照。',
        '只有一个模型种子，两个源信道 draw 不是两个模型种子。单种子正差不能证明可重复双提升。所有逐行/逐环境 acc、F1、最差 RX、RX/day/TX 分层、修复及新增错误、margin/30、固定 NoAs30 clean 正确集合保持率见 CSV。实际 AS_CE20 父的源域 clean 正确集合由既有同 ID 缓存另外计算，见 actual_parent_source_cohort；二者不混称，正式父正确集合缺失不另做推理。','',
        '训练分解先减去 ε 替换 correction，独立核对原 CE/RDrop/SupCon 总式，再还原实际 smoothed total L；没有把 correction 当额外辅助损失或新增 CE 视图。源状态/lock/质量结果为关联描述。实际新增噪声比不是总 SNR，正式每包状态标签不可用。','',
        '已存源域决定（formal 前读取，不重算晋级）：'+json.dumps(result['stored_source_decision'],ensure_ascii=False),'',
        f'新增20模型轮/4440更新，每行累计30轮/6660更新；新增 native NN={cost["new_native_NN"]}，source={cost["new_source_NN"]}，formal={cost["new_formal_NN"]}，共享 no-query smoke={cost["shared_remote_smoke_NN"]}，合计={cost["new_total_remote_NN"]}，本地NN=0。三个不同父E20与三个复用尾E10单列且去重；唯一历史加新增共110模型轮/24420更新，五条30轮曲线共150行不等于150轮唯一训练。',
        f'新尾训练过程秒数合计={cost["new_training_wall_seconds_sum"]:.3f}；source={cost["new_source_wall_seconds_sum"]:.3f}；formal prediction={cost["new_formal_wall_seconds_sum"]:.3f}。并行过程秒数和不等于矩阵实际经过时间；相同NN不等于相同墙钟。预算结束不等于收敛。','',
        '未完成/限制：'+json.dumps(result['missing'],ensure_ascii=False),'','实际图表：']
    text+=['- '+name for name in result['figures']]
    text+=['','本分析只读既有保存产物；新增NN/IQ/checkpoint读取/更新/重推理/重评分均为0。']
    (out/'final_analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def run(project,out,truth=None,code_commit=None,local_nn=0,smoke_nn=1540):
    project=Path(project);decision=read_source_decision(project);out=Path(out)
    c.require(not out.exists(),'Immutable fresh analysis directory required');out.mkdir(parents=False)
    started=time.perf_counter();c.write(out/'stored_source_decision.json',decision);specs=row_specs(project)
    missing=[];training=None;curves=[];source=None;source_rows=[];pooled=None;parent_cohort=None
    if decision['status'] in c.DECISION_PARTIAL:missing.append(dict(component='source_decision',error='Stored source verdict PARTIAL; not recomputed'))
    for component in ('training','source'):
        try:
            if component=='training':training,curves=training_analysis(project,specs,out)
            else:
                source,source_rows=c.source_analysis(project,specs,out,decision);pooled=pooled_source(source_rows);c.write(out/'source_pooled_midlow.json',pooled)
        except Exception as error:
            failure=dict(component=component,error=repr(error),traceback=traceback.format_exc(),formal_still_required=True)
            missing.append(failure);c.write(out/(component+'_failure.json'),failure)
    try:parent_cohort=parent_source_cohort(project,specs,out)
    except Exception as error:
        failure=dict(component='actual_parent_source_cohort',error=repr(error),traceback=traceback.format_exc(),
            source_promotion_gate=False,formal_still_required=True,new_inference_forbidden=True)
        missing.append(failure);c.write(out/'actual_parent_source_cohort_missing.json',failure)
    truth=Path(truth) if truth else project/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
    formal=c.formal_analysis(project,specs,out,truth);cost=costs(project,specs,local_nn,smoke_nn,training)
    cohort_NN=parent_cohort['historical_source_readout_NN'] if parent_cohort else None
    cost['actual_parent_cohort_historical_NN']=cohort_NN
    cost['actual_parent_cohort_new_NN']=0
    cost['unique_total_known_remote_NN_including_cohort']=cost['unique_total_known_remote_NN']+cohort_NN if cohort_NN is not None else None
    c.write(out/'costs.json',cost)
    try:
        figures=plots(out,curves,source_rows,formal,training)
        if training and source_rows:c.require(len(figures)==8,'Four actual PNG/PDF figure pairs required')
    except Exception as error:
        figures=[];failure=dict(component='plots',error=repr(error),traceback=traceback.format_exc());missing.append(failure);c.write(out/'plot_failure.json',failure)
    result=dict(status='ANALYZED' if not missing else 'FORMAL_ANALYZED_WITH_PARTIAL_COMPONENTS',run_id=RUN,code_commit=code_commit,
        stored_source_decision=decision,training=training,source=source,source_pooled_midlow=pooled,
        actual_parent_source_cohort=parent_cohort,formal=formal,cost=cost,figures=figures,
        missing=missing,model_seed_count=1,crossseed_confirmed=False,reproducible_dual_gain=False,target_feedback=False,
        source_decision_recomputed=False,goal_complete=False,NN=0,IQ_reads=0,checkpoint_reads=0,training_updates=0,
        wall_seconds=time.perf_counter()-started)
    report(out,result);c.write(out/'summary.json',{k:v for k,v in result.items() if k!='training'});c.write(out/'complete.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--truth',type=Path);parser.add_argument('--code-commit');parser.add_argument('--local-nn',type=int,default=0)
    parser.add_argument('--smoke-nn',type=int,default=1540);parser.add_argument('--checks',action='store_true');args=parser.parse_args()
    if args.checks:
        path=Path(__file__).with_name('cvs_quality_entropy77_analysis_checks_20261019.py')
        spec=importlib.util.spec_from_file_location('_p77_checks',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        print(json.dumps(module.checks()));return
    if args.project is None or args.out is None:parser.error('--project and --out required unless --checks')
    result=run(args.project,args.out,args.truth,args.code_commit,args.local_nn,args.smoke_nn)
    print(json.dumps(dict(status=result['status'],out=str(args.out),missing=[x['component'] for x in result['missing']],NN=0,IQ_reads=0,checkpoint_reads=0)),flush=True)


if __name__=='__main__':main()
