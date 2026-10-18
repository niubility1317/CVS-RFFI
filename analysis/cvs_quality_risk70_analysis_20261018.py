"""P70 finite nominal-pool quality readout. CPU only; no IQ/checkpoint reads."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import importlib.util
import json
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

RUN = '20261018-phase1-FcrossRX-qualityStrat-e30-s2026092701-m2-r01'
REF = '20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
SEED, UPDATES, BATCH, POOL_N = 2026092701, 2220, 64, 142080
ARMS = ('NOAS30', 'AS30', 'POOL_RANDOM', 'QUALITY_STRAT')
NEW_ARMS = ARMS[2:]
PAIRS = (('AS30', 'NOAS30'), ('POOL_RANDOM', 'AS30'), ('QUALITY_STRAT', 'POOL_RANDOM'),
         ('QUALITY_STRAT', 'AS30'), ('POOL_RANDOM', 'NOAS30'), ('QUALITY_STRAT', 'NOAS30'))
GROUPS = ('<3', '3-8', '8-15', '>=15')


def load_common():
    # CPU helper dependencies are explicit and copied unchanged into the release.
    path = Path(__file__).with_name('cvs_optimizer_continuity69_analysis_20261018.py')
    spec = importlib.util.spec_from_file_location('_p70_common69', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


h = load_common(); q = h.q; c = h.c


def row_specs(project):
    runs = Path(project)/'runs'; specs = []
    for key, actual, reused in (('NOAS30','B00',True), ('AS30','B01',True),
                                ('POOL_RANDOM','M1',False), ('QUALITY_STRAT','M2',False)):
        run = REF if reused else RUN; parent_arm = 'B00' if key == 'NOAS30' else 'B01'
        specs.append(dict(key=key, run=run, arm=parent_arm, actual_arm=actual,
            row_id=actual+'-s'+str(SEED), reused=reused,
            root=runs/run/(actual+'-s'+str(SEED)),
            parent=runs/REF/(parent_arm+'-s'+str(SEED))/'stage20'))
    return specs


def training_complete(spec, project):
    roles = c.read(Path(project)/'runs'/c.CONTRACT_RUN/'source_contract.json')['role_ids']
    for folder, parent, epochs, labels, pseudo, updates in (
            (spec['parent'],True,20,13,7,4440), (spec['root']/'source',False,10,0,10,2220)):
        done=c.read(folder/'completion.json'); cfg=done['config']; init=c.read(folder/'initialization.json')
        result=c.read(folder/'result.json'); frozen=c.read(folder/'source_frozen.json')
        args=c.read(folder/'actual_native_args.json'); args=args.get('args',args)
        row=spec['arm']+'-s'+str(SEED) if parent else spec['row_id']
        expected=[] if parent else [str(spec['parent']/'final_ssdg.pth')]
        c.require((done.get('epoch'),done.get('optimizer_steps'),cfg.get('epochs'),cfg.get('label_epochs'),
            cfg.get('pseudo_epochs'),cfg.get('updates'))==(epochs,updates,epochs,labels,pseudo,updates), 'Exact E20+10 calendar required')
        c.require(cfg.get('run_id')==(REF if parent else spec['run']) and cfg.get('row_id')==row and
            cfg.get('model_seed')==SEED and cfg.get('As') is (spec['key']!='NOAS30'), 'Exact actual parent/tail row required')
        c.require(cfg.get('checkpoint_sources')==init.get('checkpoint_sources')==expected and
            init.get('scratch_only') is parent and init.get('source_roles')=='EXACT_MATCH' and
            init.get('target_contact') is False, 'Exact source-fixed P43 student/ownEMA parent required')
        c.require(frozen.get('status')=='SOURCE_FROZEN' and frozen.get('config')==cfg and
            frozen.get('selection')=='final_only' and frozen.get('target_access') is False, 'Final-only default head required')
        c.require(done.get('target_access') is False and done.get('head_fit_updates')==0 and
            done.get('native_fitted_probes_disabled') is True and result.get('config')==cfg and
            result.get('U_truth_read') is False and result.get('target_access') is False and result.get('head_fit_updates')==0,
            'Joint default head/hidden U/source boundary differs')
        c.require(c.read(folder/'source_contract.json')['role_ids']==roles and result.get('source_RX')==list(c.RX) and
            result.get('source_RX_holdout') is None and done['final_source_metrics'].get('source_val_count')==27000,
            'Complete fullsource roles required')
        c.require(cfg.get('Ag_mode')=='NONE' and cfg.get('new_Ag') is False and cfg.get('L_NN_per_step')==768 and
            cfg.get('total_L_loss_scale')==1.25 and args.get('pseudo_temporal_mode')=='batch_neighbor' and
            args.get('pseudo_temporal_gate') is True and args.get('pseudo_temporal_bank_min_streak')==2,
            'Original sixL/1.25/native U rules required')
        if not parent:
            c.require(cfg.get('parent_source')==str(spec['parent']) and cfg.get('epoch_offset')==20 and
                cfg.get('step_offset')==4440, 'Same legal own20/reset tail clocks required')
            if not spec['reused']:
                pool=cfg.get('quality_pool',{})
                c.require(cfg.get('method')==spec['key'] and pool.get('N')==POOL_N and pool.get('updates')==UPDATES and
                    pool.get('batch_size')==BATCH and pool.get('one_use') is True and pool.get('extra_proposals')==0 and
                    pool.get('satellite_CE_coefficient')==pool.get('satellite_KL_coefficient')==.25,
                    'Same finite nominal pool and original CE/KL mass required')
    return c.read(spec['root']/'source/completion.json')


def effects(rows,new,old):
    result={v:{k+'_delta_pp':100*(rows[new][v][k]-rows[old][v][k])
        for k in ('accuracy','macro_f1','worst_RX','satellite_retention')} for v in c.VIEWS}
    result['urban_accuracy_mean_delta_pp']=float(np.mean([result[v]['accuracy_delta_pp'] for v in c.VIEWS[-3:]]))
    return result


def enrich_formal(obj):
    rows=obj['rows']; obj['comparisons']={new+'__'+old:effects(rows,new,old) for new,old in PAIRS}
    obj['primary']=dict(new='QUALITY_STRAT',control='POOL_RANDOM',effects=effects(rows,'QUALITY_STRAT','POOL_RANDOM'))
    obj['stage']={}
    for key in ('AS30',*NEW_ARMS):
        delta=effects(rows,key,'NOAS30'); clean=delta['clean']['accuracy_delta_pp']
        obj['stage'][key]=dict(control='NOAS30',clean_delta_pp=clean,
            low_urban_delta_pp=delta[c.VIEWS[-1]]['accuracy_delta_pp'],
            mid_urban_delta_pp=delta[c.VIEWS[-2]]['accuracy_delta_pp'],
            urban_mean_delta_pp=delta['urban_accuracy_mean_delta_pp'],clean_within_0p1pp=bool(clean>=-.1-1e-10),
            single_seed_observed_dual_gain=bool(clean>0 and delta['urban_accuracy_mean_delta_pp']>0),reproducible_dual_gain=False)
    obj['four_rows_are_not_a_two_by_two_factorial']=True


def write(path,obj):
    if Path(path).name=='formal_summary.json':enrich_formal(obj)
    q._write(path,obj)


c.RUN,c.NEW_ARMS,c.PAIRS,c.CANDIDATE=RUN,NEW_ARMS,PAIRS,'QUALITY_STRAT'
c.row_specs,c.training_complete,c.write=row_specs,training_complete,write


def source_criteria(clean_m2,clean_noas,midlow_m2,midlow_m1):
    clean=100*(clean_m2-clean_noas); gain=100*(midlow_m2-midlow_m1)
    checks=dict(fullV_clean_M2_vs_NoAs_ge_minus_0p1pp=clean>=-.1-1e-10,
                pooled_midlow_M2_vs_M1_ge_0p5pp=gain>=.5-1e-10)
    return dict(checks=checks,source_qualified=all(checks.values()),clean_delta_pp=clean,midlow_delta_pp=gain)


def read_source_decision(project):
    value=c.read(Path(project)/'runs'/RUN/'source_matrix_decision.json')
    c.require(value.get('status') in c.DECISION_COMPLETE|c.DECISION_PARTIAL and value.get('formal_scores_read') is False and
        value.get('target_feedback') is False and value.get('NN')==0 and value.get('model_seed_count')==1,
        'Already-fixed source-only decision required before formal reads')
    if value['status'] in c.DECISION_COMPLETE:
        checks=value.get('checks'); c.require(isinstance(checks,dict) and len(checks)==2 and
            all(type(x) is bool for x in checks.values()) and type(value.get('source_qualified')) is bool and
            value['source_qualified']==all(checks.values()), 'Exactly the two fixed source criteria required')
    return value


def validate_pool_steps(records, counts, method, expected_updates=UPDATES, batch=BATCH, pool_groups=None):
    """Pure-array recount of one-use allocations and saved finite-pool CE estimators."""
    counts=np.asarray(counts)
    c.require(counts.shape==(4,) and counts.dtype.kind in 'iu' and np.all(counts>=0) and
        np.all((counts==0)|(counts>=expected_updates)),
        'Each positive-mass pool group must support every step without replacement')
    n=int(counts.sum()); c.require(n==expected_updates*batch,'Pool size must equal consumed As occurrences')
    weights=counts/n; seen=set(); totals=np.zeros(4,dtype=np.int64); matrix=[]; output=[]
    lo=counts//expected_updates; hi=(counts+expected_updates-1)//expected_updates
    for step,r in enumerate(records,1):
        c.require(r.get('step')==step and r.get('cumulative_step')==step+4440 and
            r.get('method')==method,'Contiguous actual quality-estimator clock/method required')
        idx=r.get('used_pool_indices'); group=r.get('groups'); cnt=np.asarray(r.get('counts'))
        c.require(isinstance(idx,list) and len(idx)==batch and all(type(i) is int and 0<=i<n for i in idx) and
            len(set(idx))==batch and not seen.intersection(idx),'Each full nominal observation must be used exactly once')
        c.require(isinstance(group,list) and len(group)==batch and all(type(i) is int and 0<=i<4 for i in group) and
            cnt.shape==(4,) and cnt.dtype.kind in 'iu' and np.array_equal(cnt,np.bincount(group,minlength=4)),
            'Saved per-step bins/counts must match all 64 records')
        if pool_groups is not None:c.require(np.array_equal(np.asarray(pool_groups)[idx],group),'Allocation bins differ from the fixed metadata-only pool array')
        origin_steps=[i//batch+1 for i in idx]
        origin_slots=[2*(i%batch)+1-(e%2) for i,e in zip(idx,origin_steps)]
        c.require(r.get('origin_steps')==origin_steps and r.get('origin_slots')==origin_slots,
            'Original nominal epoch/step/slot mapping must move with each observation')
        c.require(np.allclose(r.get('p_g'),weights,atol=1e-14,rtol=0),'Fixed train-pool weights must not vary')
        if method=='QUALITY_STRAT':c.require(np.all((cnt==lo)|(cnt==hi)) and
            np.all(cnt[counts>0]>0) and np.all(cnt[counts==0]==0),'Exact positive-mass floor/ceil group quotas required')
        unweighted=c.finite(r.get('As_CE_unweighted'),'As mean CE'); weighted=c.finite(r.get('As_CE_weighted'),'As used CE')
        means=r.get('group_CE'); c.require(isinstance(means,list) and len(means)==4,'All group CE readouts required')
        for mean,num in zip(means,cnt):c.require((mean is None and num==0) or (num>0 and c.finite(mean,'group mean CE')>=0),'Group means must preserve empty groups')
        expected_unweighted=sum((float(mean)*int(num) if num else 0.) for mean,num in zip(means,cnt))/batch
        expected_used=sum(float(weights[g])*float(means[g]) for g in range(4) if weights[g]>0) if method=='QUALITY_STRAT' else unweighted
        correction=.25*(weighted-unweighted)
        c.require(abs(unweighted-expected_unweighted)<3e-5 and abs(weighted-expected_used)<3e-5 and
            abs(c.finite(r.get('CE_correction_to_total'),'CE correction')-correction)<3e-5 and
            abs(c.finite(r.get('used_total_L'),'actual total L')-c.finite(r.get('original_total_L'),'hypothetical ordinary mean total L')-correction)<3e-5,
            'Actual .25 CE estimator/correction differs')
        for key in ('As_KL_unweighted','ordinary64_CE','ordinary64_KL'):c.require(c.finite(r.get(key),key)>=0,'Actual unweighted RD group readouts required')
        c.require(r.get('same_IQ_each_RD_pair') is True and r.get('all_labeled_CE_retained') is True and
            r.get('As_KL_quality_weighted') is False and r.get('U_truth_read') is False and
            r.get('extra_NN')==0,'Original same-IQ KL/all true labels/no-extra-NN required')
        seen.update(idx); totals+=cnt; matrix.append(cnt)
        output.append(dict(step=step,cumulative_step=step+4440,epoch=(step-1)//222+21,
            As_CE_unweighted=unweighted,As_CE_used=weighted,CE_correction=correction,
            **{'count_'+str(g):int(cnt[g]) for g in range(4)},
            **{'CE_'+str(g):means[g] for g in range(4)}))
    c.require(len(output)==expected_updates and len(seen)==n and np.array_equal(totals,counts),'Complete one-use pool consumption required')
    matrix=np.asarray(matrix)
    return dict(method=method,steps=len(output),packets=n,unique_pool_observations=len(seen),one_use=True,
        group_counts=counts.tolist(),p_g=weights.tolist(),positive_groups=np.flatnonzero(counts>0).tolist(),
        empty_groups=np.flatnonzero(counts==0).tolist(),batch_count_mean=matrix.mean(0).tolist(),
        batch_count_variance=matrix.var(0).tolist(),batch_count_min=matrix.min(0).tolist(),batch_count_max=matrix.max(0).tolist(),
        As_CE_unweighted_mean=float(np.mean([r['As_CE_unweighted'] for r in output])),
        As_CE_used_mean=float(np.mean([r['As_CE_used'] for r in output])),
        CE_correction_mean=float(np.mean([r['CE_correction'] for r in output])),
        fixed_finite_pool_risk=True,continuous_population_exact_unbiased=False,
        trajectory_conditional_unbiased_claimed=False,extra_NN=0),output


def pool_readout(project,spec):
    folder=spec['root']/'source'; initialization=c.read(folder/'quality_pool_initialization.json')
    actual=c.read(folder/'quality_training_actual.json'); saved=c.read(folder/'result.json')
    summary=initialization['summary']; counts=summary['group_counts']
    shared=Path(project)/'runs'/RUN/'nominal_pool'; canonical=c.read(shared/'complete.json')
    c.require(summary==canonical and summary.get('status')=='READY' and summary.get('written')==POOL_N and
        summary.get('N')==POOL_N and summary.get('real_CVS_NN')==summary.get('extra_proposals')==summary.get('dropped')==0 and
        summary.get('source_IQ_observations')==POOL_N and summary.get('occurrence_one_use') is True,
        'Exact complete shared train-only nominal pool required')
    # This file contains only integer quality-bin metadata, never IQ.
    pool_groups=np.load(shared/'groups.npy',allow_pickle=False)
    c.require(pool_groups.shape==(POOL_N,) and pool_groups.dtype.kind in 'iu' and np.all((pool_groups>=0)&(pool_groups<4)) and
        np.array_equal(np.bincount(pool_groups,minlength=4),counts),'Metadata-only pool bins/counts differ')
    c.require(initialization.get('fixed_before_first_update') is True and initialization.get('shared_pool') is True and
        initialization.get('V_read') is False and initialization.get('target_read') is False and
        initialization.get('no_extra_neural_forward') is True,'Pool must precede all updates and exclude V/target')
    c.require(actual.get('active_BatchNorm')==0 and actual.get('satellite_CE_coefficient')==.25 and
        actual.get('satellite_KL_coefficient')==.25 and actual.get('extra_NN')==0,
        'Original per-packet graph and fixed CE/KL coefficients required')
    records=list(c.lines(folder/'quality_steps.jsonl.gz'))
    result,flat=validate_pool_steps(records,counts,spec['key'],pool_groups=pool_groups)
    c.require(saved.get('quality_method')==spec['key'] and saved.get('quality_pool')==c.read(folder/'quality_pool_final.json'),
        'Final reader summary must be the saved training result')
    reader=saved['quality_pool']
    c.require(reader.get('calls')==UPDATES and reader.get('packets')==reader.get('pool_N')==reader.get('unique_occurrences_used')==POOL_N and
        reader.get('group_counts')==list(counts) and np.allclose(reader.get('p_g'),np.asarray(counts)/POOL_N,atol=1e-14,rtol=0) and
        reader.get('fixed_pool') is True and reader.get('new_channel_calls')==reader.get('dropped')==0,
        'Actual reader summary must reconcile with every one-use step')
    result.update(initialization=initialization,pool_summary=summary,reader_summary=saved['quality_pool'],
        training_actual=actual,load_or_prepare_seconds=c.finite(initialization['load_or_prepare_seconds'],'pool setup wall'))
    return result,flat,records


def undo_quality_CE(reference,quality):
    """Return an in-memory ordinary-mean view for the original log checker."""
    delta=quality['CE_correction_to_total']; record=dict(reference)
    c.require(abs(record['total_L']-quality['used_total_L'])<3e-5 and
        abs(record['extra_CE']-.5*(quality['ordinary64_CE']+quality['As_CE_weighted']))<3e-5 and
        abs(record['extra_KL']-.5*(quality['ordinary64_KL']+quality['As_KL_unweighted']))<3e-5,
        'Quality and reference CE/KL/actual L totals disagree')
    record['total_L']-=delta; record['base_L_before_shape']-=delta
    record['extra_CE']-=2*delta
    if 'extra_RD' in record:record['extra_RD']-=2*delta
    return record


def parse_new_tail(spec,records):
    # Reuse original full-log checker after undoing only the independently
    # recounted CE correction in a transient view; saved files are untouched.
    source=spec['root']/'source'; by_step={r['step']:r for r in records}; old_lines=c.lines
    def adjusted(path):
        for record in old_lines(path):
            if Path(path)==source/'reference_steps.jsonl.gz':
                record=undo_quality_CE(record,by_step[record['step']])
            yield record
    c.lines=adjusted
    try:tail,ordinary,_=q.parse_tail(source,True,False)
    finally:c.lines=old_lines
    for point in tail['curve']:
        values=[r for r in records if (r['step']-1)//222+1==point['local_epoch']]
        delta=float(np.mean([r['CE_correction_to_total'] for r in values])); point['total_L']+=delta
        point['extra_CE_same_logits_unweighted']=point['extra_CE']; point['extra_CE']+=2*delta
        point.update(As_CE_used=float(np.mean([r['As_CE_weighted'] for r in values])),
            As_CE_unweighted=float(np.mean([r['As_CE_unweighted'] for r in values])),CE_correction=delta)
    tail['original_decomposition_checked_after_removing_only_quality_CE_correction']=True
    return tail,ordinary


def training_analysis(project,specs,out):
    tails={}; curves=[]; coverage=[]; quality_rows=[]; new_pool={}; by_key={s['key']:s for s in specs}
    for spec in specs:
        key=spec['key']
        if spec['reused']:tail,ordinary,_=q.parse_tail(spec['root']/'source',spec['arm']=='B01',False)
        else:
            pool,flat,records=pool_readout(project,spec); new_pool[key]=pool
            tail,ordinary=parse_new_tail(spec,records); tail['quality_pool_readout']=pool
            tail['physical_sequence']=q.same_sequence(spec['root']/'source/training_sequence.jsonl.gz',by_key['AS30']['root']/'source/training_sequence.jsonl.gz')
            c.require(ordinary==tails['AS30']['ordinary'],'Original ordinary augmentation stream must remain fixed')
            quality_rows.extend(dict(row_id=key,**r) for r in flat)
        tail['ordinary']=ordinary; result=c.read(spec['root']/'source/result.json')
        if spec['arm']=='B01':
            channel=result['L_channel']; cells=channel['strata']
            c.require(channel.get('packets')==POOL_N and channel.get('session_count')==15 and
                sum(r['count'] for r in cells)==POOL_N,'Full true-label nominal As coverage required')
            coverage.extend(dict(row_id=key,**r) for r in cells)
            if not spec['reused']:
                joint=result['TX_RX_scene_state_lock']; c.require(sum(r['count'] for r in joint)==POOL_N and
                    len({(r['TX'],r['RX'],r['scene']) for r in joint})==90,'All TX x source RX x train scene support required')
                c.csv_write(out/(key+'_TX_RX_scene_state_lock.csv'),joint)
        curves.extend(dict(row_id=key,reused=spec['reused'],model_seed=SEED,**r) for r in c.epoch_curve(spec['parent'],0)+tail['curve'])
        tails[key]=tail
    c.require(new_pool['POOL_RANDOM']['group_counts']==new_pool['QUALITY_STRAT']['group_counts'] and
        new_pool['POOL_RANDOM']['pool_summary']==new_pool['QUALITY_STRAT']['pool_summary'],'M1/M2 must share the exact nominal pool')
    for tail in tails.values():tail.pop('ordinary')
    c.csv_write(out/'training_curves.csv',curves); c.csv_write(out/'quality_steps.csv',quality_rows)
    c.csv_write(out/'training_As_coverage.csv',coverage)
    result=dict(status='FULL_TRAINING_LOGS_VERIFIED',rows=tails,new_model_epochs=20,new_updates=4440,
        shared_parent_model_epochs=40,unique_total_model_epochs=80,U_truth_read=False,extra_NN=0,
        nominal_pool_observations_generated_once=POOL_N,nominal_pool_shared_across_two_rows=True,
        native_U_temporal_mode='batch_neighbor',quality_balance_is_not_identity_gain=True)
    c.write(out/'training_summary.json',result); c.write(out/'pool_summary.json',new_pool)
    return result,curves


def costs(project,specs,local_nn=0,smoke_nn=4):
    c.require(local_nn==0 and type(smoke_nn) is int and smoke_nn>=0,'Actual shared smoke/local0 required')
    rows={}; new={}
    for spec in specs:
        source=c.read(spec['root']/'source_qualification/complete.json'); prediction=c.read(spec['root']/'prediction/complete.json')
        c.require(source.get('NN')==12600 and prediction.get('actual_NN')==1176002,'Source/formal forward counts differ')
        record=dict(parent=c.native_cost(spec['parent']),tail=c.native_cost(spec['root']/'source'),source_NN=12600,
            formal_NN=1176002,source_wall_seconds=source.get('wall_seconds'),formal_wall_seconds=prediction.get('wall_seconds'))
        rows[spec['key']]=record
        if not spec['reused']:
            init=c.read(spec['root']/'source/quality_pool_initialization.json')
            record['pool_load_or_prepare_seconds']=init['load_or_prepare_seconds']; new[spec['key']]=record
    smoke=c.read(Path(project)/'runs'/RUN/'real_checkpoint_smoke.json')
    c.require(smoke.get('status')=='PASS' and smoke.get('NN')==smoke_nn,'Actual no-query shared smoke differs')
    subtotal=sum(v['tail']['training_total_NN']+v['source_NN']+v['formal_NN'] for v in new.values())
    pool=c.read(Path(project)/'runs'/RUN/'nominal_pool/complete.json')
    c.require(subtotal==8703128 and len(new)==2 and all(v['tail']['native_NN']==3162962 and
        v['tail']['extra_student_U_NN']==v['tail']['extra_frozen_teacher_U_NN']==0 for v in new.values()),'Two native-only tails required')
    return dict(rows=rows,new_native_NN=6325924,new_source_NN=25200,new_formal_NN=2352004,
        shared_remote_smoke_NN=smoke_nn,new_total_remote_NN=subtotal+smoke_nn,local_verification_NN=0,
        new_model_epochs=20,new_updates=4440,unique_shared_parent_model_epochs=40,
        unique_total_model_epochs_including_reused=80,student_cumulative_epochs_each=30,
        student_cumulative_updates_each=6660,parent_training_reused_not_repeated=True,
        shared_nominal_As_generated=POOL_N,per_row_As_consumed=POOL_N,extra_simulator_proposals=0,
        shared_pool_generation_wall_seconds=pool.get('wall_seconds'),shared_pool_generation_NN=pool.get('real_CVS_NN'),
        pool_preparation_wall_is_shared_not_twice_counted=True,
        new_training_wall_seconds_sum=sum(v['tail']['wall_seconds'] for v in new.values()),
        pool_load_or_prepare_seconds_by_row={k:v['pool_load_or_prepare_seconds'] for k,v in new.items()},
        extra_student_NN=0,extra_teacher_NN=0,reference_reprediction=0,reference_rescoring=0,analysis_NN=0,
        historical_parent_source_cohort_NN=12602,parent_cohort_new_NN=0,budget_exhaustion_is_not_convergence=True)


def plots(out,curves,source_rows,formal,training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(ARMS,('#64748b','#d97706','#8b5cf6','#2563eb'))); files=[]
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):
            path=out/(name+'.'+ext); fig.savefig(path,dpi=180); files.append(str(path))
        plt.close(fig)
    if curves:
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for arm in ARMS:
            rr=[r for r in curves if r['row_id']==arm]
            for ax,key,title in zip(axes,('native_loss','source_val_accuracy_pct','grad_backbone'),('Native loss','Source V accuracy (%)','Backbone gradient norm')):
                ax.plot([r['epoch'] for r in rr],[r[key] for r in rr],label=arm,color=colors[arm]); ax.set(title=title,xlabel='Cumulative epoch'); ax.grid(alpha=.2)
        axes[0].legend(fontsize=8); save(fig,'full_training')
    if source_rows:
        fig,axes=plt.subplots(2,4,figsize=(20,8))
        for si,scene in enumerate(c.SCENES[1:]):
            for di,dim in enumerate(('state','lock','receiver_quality_proxy_db','signal_to_added_noise_db')):
                ax=axes[si,di]; rr=[r for r in source_rows if r['scene']==scene and r['draw']==-1 and r['dimension']==dim]; labels=list(dict.fromkeys(r['stratum'] for r in rr)); xx=np.arange(len(labels))
                for i,arm in enumerate(ARMS):
                    zz=[next(r for r in rr if r['row_id']==arm and r['stratum']==label) for label in labels]
                    ax.bar(xx+(i-1.5)*.2,[100*r['accuracy'] if r['accuracy'] is not None else np.nan for r in zz],.2,label=arm,color=colors[arm])
                ax.set(xticks=xx,xticklabels=labels,title=scene+' / '+dim,ylabel='Source accuracy (%)'); ax.tick_params(axis='x',labelrotation=25); ax.grid(axis='y',alpha=.2)
        axes[0,0].legend(fontsize=8); save(fig,'source_state_quality_lock_noise')
    fig,axes=plt.subplots(1,2,figsize=(15,5)); xx=np.arange(7)
    for new,old,label in (('QUALITY_STRAT','POOL_RANDOM','M2-M1'),('QUALITY_STRAT','AS30','M2-original As'),('QUALITY_STRAT','NOAS30','M2-NoAs')):
        axes[0].plot(xx,[100*(formal['rows'][new][v]['accuracy']-formal['rows'][old][v]['accuracy']) for v in c.VIEWS],marker='.',label=label)
    axes[1].plot(xx,[100*(formal['rows']['QUALITY_STRAT'][v]['macro_f1']-formal['rows']['POOL_RANDOM'][v]['macro_f1']) for v in c.VIEWS],marker='.',label='Macro-F1')
    axes[1].plot(xx,[100*(formal['rows']['QUALITY_STRAT'][v]['worst_RX']-formal['rows']['POOL_RANDOM'][v]['worst_RX']) for v in c.VIEWS],marker='.',label='Worst RX')
    for ax,title in zip(axes,('Formal paired accuracy changes','M2-M1 paired F1 / worst RX')):
        ax.set(xticks=xx,xticklabels=['clean','high sub','mid sub','low sub','high urb','mid urb','low urb'],title=title,ylabel='Absolute percentage points'); ax.tick_params(axis='x',labelrotation=30); ax.axhline(0,color='grey',lw=.7); ax.grid(axis='y',alpha=.2); ax.legend(fontsize=8)
    save(fig,'formal_paired')
    if training:
        fig,axes=plt.subplots(1,3,figsize=(15,4.5)); xx=np.arange(4)
        for i,arm in enumerate(NEW_ARMS):
            pool=training['rows'][arm]['quality_pool_readout']
            axes[0].bar(xx+(i-.5)*.35,pool['group_counts'],.35,label=arm,color=colors[arm])
            axes[1].bar(xx+(i-.5)*.35,pool['batch_count_variance'],.35,label=arm,color=colors[arm])
            rr=[r for r in curves if r['row_id']==arm and r['epoch']>20]
            axes[2].plot([r['epoch'] for r in rr],[r['As_CE_used'] for r in rr],label=arm,color=colors[arm])
        for ax,title in zip(axes[:2],('Same one-use nominal pool','Batch group-count variance')):
            ax.set(xticks=xx,xticklabels=GROUPS,title=title,xlabel='Added-noise ratio (dB)'); ax.grid(axis='y',alpha=.2); ax.legend(fontsize=8)
        axes[2].set(title='Actual satellite CE estimator',xlabel='Cumulative epoch'); axes[2].grid(alpha=.2); axes[2].legend(fontsize=8)
        save(fig,'pool_allocation_readout')
    return files


def report(out,result):
    f=result['formal']; stage=f['stage']['QUALITY_STRAT']; cost=result['cost']; primary=f['primary']['effects']
    text=['# P70：固定 nominal 训练池的质量分层','',
        'POOL_RANDOM和QUALITY_STRAT严格使用同一P43 As20 student/ownEMA，各追加10轮，重置AdamW并采用相同cos10。共享预生成142080个完整nominal As观测，各行每个观测恰用一次；ordinary/raw、ORD128 SupCon、native U及默认cosine30头保持。M1随机排列并使用普通均值CE；M2按新增噪声比四组分层，固定p_g=N_g/N，使用组均值CE。As CE与原未加权same-IQ RDrop KL系数均为.25；M3未运行。','',
        'M2只声称固定参数下对有限池风险的分层估计；不声称连续总体精确无偏、参数变化后的条件无偏或单次排列逐观测权重完全相同。质量批次方差、loss、U接受率与覆盖不是身份收益。新增噪声比不是总SNR。','',
        '| 环境 | NoAs30 % | 原As30 % | M1随机池 % | M2分层 % | M2-M1 pp |','|---|---:|---:|---:|---:|---:|']
    for v in c.VIEWS:text.append('| '+v+' | '+' | '.join(f'{100*f["rows"][a][v]["accuracy"]:.6f}' for a in ARMS)+f' | {primary[v]["accuracy_delta_pp"]:+.6f} |')
    text.extend(['',f'主要机制比较M2-M1：clean {primary["clean"]["accuracy_delta_pp"]:+.6f} pp，mid urban {primary[c.VIEWS[-2]]["accuracy_delta_pp"]:+.6f} pp，low urban {primary[c.VIEWS[-1]]["accuracy_delta_pp"]:+.6f} pp，urban均值 {primary["urban_accuracy_mean_delta_pp"]:+.6f} pp。',
        f'阶段比较M2-NoAs30：clean {stage["clean_delta_pp"]:+.6f} pp，urban均值 {stage["urban_mean_delta_pp"]:+.6f} pp；clean下降不超过0.1绝对百分点：{stage["clean_within_0p1pp"]}。',
        '只有一个模型seed；两个源信道draw不是跨seed重复。单seed及不足0.1个百分点的微小上涨均不能证明可重复双提升。原As30是配方参照，M1才是同池匹配控制；四行不是2×2因子矩阵。', '',
        'source决定先于正式读取固定，仅两项：M2对NoAs30 fullV clean≥−0.1pp；pooled mid/low M2对M1≥+0.5pp。任一FAIL不扩E50，全部七个正式环境仍报告；正式结果不用于重新晋级。',
        '全部accuracy、Macro-F1、最差RX、RX/day/TX、修复/新增错误、margin/30、固定NoAs30 clean正确集合保持率见CSV。精确P43 As20 student源clean正确集合另列，缓存缺失不新增推理或替换。状态/质量分层仅源域可用，不作物理因果归因。','',
        f'新增20 model-epochs／4440更新，复用父与旧尾段后独有训练总计80 model-epochs。新增native NN={cost["new_native_NN"]}，source={cost["new_source_NN"]}，formal={cost["new_formal_NN"]}，共享smoke={cost["shared_remote_smoke_NN"]}，远端合计={cost["new_total_remote_NN"]}，本地NN=0。完整As只共享生成142080次，各行消耗142080次；两行原本分别在线生成的模拟总数不可再计两份共享生成。新增训练行墙钟之和={cost["new_training_wall_seconds_sum"]:.3f}秒，pool准备/等待另列。预算耗尽不等于收敛。','',
        '已固定源决定：'+json.dumps(result['stored_source_decision'],ensure_ascii=False),'',
        '未完成／限制：'+json.dumps(result['missing'],ensure_ascii=False),'','实际图表：'])
    text.extend('- '+p for p in result['figures']); text.extend(['','分析自身NN、IQ/checkpoint读取、训练更新、重预测及重新评分均为0。'])
    (out/'final_analysis.md').write_text('\n'.join(text)+'\n',encoding='utf8')


def run(project,out,truth=None,code_commit=None,local_nn=0,smoke_nn=4):
    project=Path(project); decision=read_source_decision(project); out=Path(out)
    c.require(not out.exists(),'Immutable analysis output required'); out.mkdir(parents=False)
    c.write(out/'stored_source_decision.json',decision); started=time.perf_counter(); specs=row_specs(project)
    missing=[]; training=None; curves=[]; source=None; source_rows=[]; parent=None
    if decision['status'] in c.DECISION_PARTIAL:missing.append(dict(component='source_decision',error='Stored decision PARTIAL; no recomputation'))
    for name in ('training','source','parent_cohort'):
        try:
            if name=='training':training,curves=training_analysis(project,specs,out)
            elif name=='source':source,source_rows=c.source_analysis(project,specs,out,decision)
            else:parent=h.parent_cohort_analysis(project,specs,out)
        except Exception as error:
            failure=dict(component=name,error=repr(error),traceback=traceback.format_exc(),formal_still_required=True)
            missing.append(failure); c.write(out/(name+'_failure.json'),failure)
    truth=Path(truth) if truth else project/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
    formal=c.formal_analysis(project,specs,out,truth); cost=costs(project,specs,local_nn,smoke_nn); c.write(out/'costs.json',cost)
    try:figures=plots(out,curves,source_rows,formal,training)
    except Exception as error:
        figures=[]; failure=dict(component='plots',error=repr(error),traceback=traceback.format_exc()); missing.append(failure); c.write(out/'plot_failure.json',failure)
    result=dict(status='ANALYZED' if not missing else 'FORMAL_ANALYZED_WITH_PARTIAL_COMPONENTS',run_id=RUN,code_commit=code_commit,
        stored_source_decision=decision,training=training,source=source,parent_cohort=parent,formal=formal,cost=cost,
        figures=figures,missing=missing,model_seed_count=1,crossseed_confirmed=False,reproducible_dual_gain=False,
        target_feedback=False,source_decision_recomputed=False,goal_complete=False,NN=0,IQ_reads=0,checkpoint_reads=0,
        training_updates=0,wall_seconds=time.perf_counter()-started)
    report(out,result); c.write(out/'summary.json',{k:v for k,v in result.items() if k!='training'}); c.write(out/'complete.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path); parser.add_argument('--out',type=Path); parser.add_argument('--truth',type=Path)
    parser.add_argument('--code-commit'); parser.add_argument('--local-nn',type=int,default=0); parser.add_argument('--smoke-nn',type=int,default=4)
    parser.add_argument('--checks',action='store_true'); args=parser.parse_args()
    if args.checks:
        path=Path(__file__).with_name('cvs_quality_risk70_analysis_checks_20261018.py')
        spec=importlib.util.spec_from_file_location('_p70_checks',path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.checks(); return
    if args.project is None or args.out is None:parser.error('--project and --out required unless --checks')
    result=run(args.project,args.out,args.truth,args.code_commit,args.local_nn,args.smoke_nn)
    print(json.dumps(dict(status=result['status'],out=str(args.out),missing=[r['component'] for r in result['missing']],NN=0,IQ_reads=0,checkpoint_reads=0)),flush=True)


if __name__=='__main__':main()
