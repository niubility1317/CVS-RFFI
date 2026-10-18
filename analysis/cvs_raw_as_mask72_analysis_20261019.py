"""P72 saved-output CPU analysis; no model, IQ, checkpoint or new scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
import argparse
import importlib.util
import json
import time
import traceback
from pathlib import Path
import numpy as np

RUN='20261019-phase1-FcrossRX-rawAsMask-e30-s2026092701-m2-r01'
REF='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
POOL_RUN='20261018-phase1-FcrossRX-qualityStrat-e30-s2026092701-m2-r01'
SEED,UPDATES,AS_PACKETS=2026092701,2220,142080
TARGET='id_backbone.encoder.core.id_backbone.fuse.2'
MASK_SLOTS=64*160
ARMS=('NOAS30','AS30','MASK_INDEPENDENT','MASK_SHARED')
MODES={'MASK_INDEPENDENT':'INDEPENDENT_FUSE','MASK_SHARED':'SHARED_RAW_AS_FUSE'}
PAIRS=(('AS30','NOAS30'),('MASK_INDEPENDENT','AS30'),('MASK_SHARED','MASK_INDEPENDENT'),
       ('MASK_SHARED','AS30'),('MASK_INDEPENDENT','NOAS30'),('MASK_SHARED','NOAS30'))


def load_common():
    path=Path(__file__).with_name('cvs_optimizer_continuity69_analysis_20261018.py')
    spec=importlib.util.spec_from_file_location('_p72_common69',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


h=load_common();q=h.q;c=h.c


def row_specs(project):
    runs=Path(project)/'runs';rows=[]
    for key,arm,reused in (('NOAS30','B00',True),('AS30','B01',True),('MASK_INDEPENDENT','I1',False),('MASK_SHARED','S1',False)):
        parent='B00' if key=='NOAS30' else 'B01';run=REF if reused else RUN
        rows.append(dict(key=key,run=run,row_id=arm+'-s'+str(SEED),actual_arm=arm,arm=parent,reused=reused,
            root=runs/run/(arm+'-s'+str(SEED)),parent=runs/REF/(parent+'-s'+str(SEED))/'stage20'))
    return rows


def training_complete(spec,project):
    roles=c.read(Path(project)/'runs'/c.CONTRACT_RUN/'source_contract.json')['role_ids']
    for folder,parent,epochs,labels,pseudo,updates in ((spec['parent'],True,20,13,7,4440),(spec['root']/'source',False,10,0,10,2220)):
        done=c.read(folder/'completion.json');cfg=done['config'];init=c.read(folder/'initialization.json')
        frozen=c.read(folder/'source_frozen.json');result=c.read(folder/'result.json')
        args=c.read(folder/'actual_native_args.json');args=args.get('args',args)
        row=spec['arm']+'-s'+str(SEED) if parent else spec['row_id'];expected=[] if parent else [str(spec['parent']/'final_ssdg.pth')]
        c.require((done.get('epoch'),done.get('optimizer_steps'),cfg.get('epochs'),cfg.get('label_epochs'),cfg.get('pseudo_epochs'),cfg.get('updates'))==
            (epochs,updates,epochs,labels,pseudo,updates),'Exact E20+10 calendar required')
        c.require(cfg.get('run_id')==(REF if parent else spec['run']) and cfg.get('row_id')==row and cfg.get('model_seed')==SEED and
            cfg.get('As') is (spec['key']!='NOAS30'),'Exact parent/tail identity and As role required')
        c.require(cfg.get('checkpoint_sources')==init.get('checkpoint_sources')==expected and init.get('scratch_only') is parent and
            init.get('source_roles')=='EXACT_MATCH' and init.get('target_contact') is False,'Only the corresponding legal P43 parent student/ownEMA required')
        c.require(frozen.get('status')=='SOURCE_FROZEN' and frozen.get('config')==cfg and frozen.get('selection')=='final_only' and
            frozen.get('target_access') is False,'Final-only joint default head required')
        c.require(done.get('target_access') is False and done.get('head_fit_updates')==0 and done.get('native_fitted_probes_disabled') is True and
            result.get('config')==cfg and result.get('target_access') is False and result.get('U_truth_read') is False and result.get('head_fit_updates')==0,
            'Default head/source-only/hidden U boundary differs')
        c.require(c.read(folder/'source_contract.json')['role_ids']==roles and result.get('source_RX')==list(c.RX) and
            result.get('source_RX_holdout') is None and done['final_source_metrics'].get('source_val_count')==27000,'Complete original source roles required')
        c.require(cfg.get('Ag_mode')=='NONE' and cfg.get('new_Ag') is False and cfg.get('L_NN_per_step')==768 and cfg.get('total_L_loss_scale')==1.25 and
            args.get('pseudo_temporal_mode')=='batch_neighbor' and args.get('pseudo_temporal_gate') is True and
            args.get('pseudo_temporal_bank_min_streak')==2,'Original sixL/loss scale/native U rules required')
        if not parent:c.require(cfg.get('parent_source')==str(spec['parent']) and cfg.get('epoch_offset')==20 and cfg.get('step_offset')==4440,
            'Exact ownAs20/reset tail clocks required')
    return c.read(spec['root']/'source/completion.json')


def effects(rows,new,old):return h.effects(rows,new,old)


def enrich_formal(obj):
    rows=obj['rows'];obj['comparisons']={new+'__'+old:effects(rows,new,old) for new,old in PAIRS}
    obj['primary']=dict(new='MASK_SHARED',control='MASK_INDEPENDENT',effects=effects(rows,'MASK_SHARED','MASK_INDEPENDENT'))
    obj['stage']={}
    for arm in ('AS30','MASK_INDEPENDENT','MASK_SHARED'):
        delta=effects(rows,arm,'NOAS30');clean=delta['clean']['accuracy_delta_pp'];urban=delta['urban_accuracy_mean_delta_pp']
        obj['stage'][arm]=dict(control='NOAS30',clean_delta_pp=clean,mid_urban_delta_pp=delta[c.VIEWS[-2]]['accuracy_delta_pp'],
            low_urban_delta_pp=delta[c.VIEWS[-1]]['accuracy_delta_pp'],urban_mean_delta_pp=urban,clean_within_0p1pp=bool(clean>=-.1-1e-10),
            single_seed_observed_dual_gain=bool(clean>0 and urban>0),reproducible_dual_gain=False)
    obj['four_rows_are_not_a_two_by_two_factorial']=True


def write(path,obj):
    if Path(path).name=='formal_summary.json':enrich_formal(obj)
    q._write(path,obj)


c.RUN,c.NEW_ARMS,c.PAIRS,c.CANDIDATE=RUN,ARMS[2:],PAIRS,'MASK_SHARED'
c.row_specs,c.training_complete,c.write=row_specs,training_complete,write


def source_criteria(clean_shared,clean_noas,midlow_shared,midlow_independent):
    clean=100*(clean_shared-clean_noas);urban=100*(midlow_shared-midlow_independent)
    checks=dict(fullV_clean_shared_vs_NoAs_ge_minus_0p1pp=clean>=-.1-1e-10,
        pooled_midlow_shared_vs_independent_ge_0p5pp=urban>=.5-1e-10)
    return dict(checks=checks,source_qualified=all(checks.values()),clean_delta_pp=clean,midlow_delta_pp=urban)


def read_source_decision(project):
    value=c.read(Path(project)/'runs'/RUN/'source_matrix_decision.json')
    c.require(value.get('status') in c.DECISION_COMPLETE|c.DECISION_PARTIAL and value.get('formal_scores_read') is False and
        value.get('target_feedback') is False and value.get('model_seed_count')==1 and value.get('NN')==0,
        'Already fixed source-only decision required before formal reads')
    if value['status'] in c.DECISION_COMPLETE:
        checks=value.get('checks');c.require(isinstance(checks,dict) and len(checks)==2 and all(type(v) is bool for v in checks.values()) and
            type(value.get('source_qualified')) is bool and value['source_qualified']==all(checks.values()),'Exactly two preregistered source criteria required')
    return value


def validate_mask_records(records,mode,expected_updates=UPDATES):
    c.require(mode in MODES.values(),'Only the two fixed mask laws are allowed')
    steps=[];epochs={};private_kept={k:0 for k in ('As0','As1','raw0','raw1')}
    for step,r in enumerate(records,1):
        epoch=(step-1)//222+1
        c.require(r.get('step')==step and r.get('cumulative_step')==step+4440 and r.get('epoch')==epoch and
            r.get('cumulative_epoch')==epoch+20 and r.get('method')==mode and
            r.get('logical_method')==('MASK_SHARED' if mode=='SHARED_RAW_AS_FUSE' else 'MASK_INDEPENDENT'),
            'Complete native mask clocks/method required')
        c.require(r.get('target')==TARGET and r.get('dropout_p')==.45 and r.get('calls')==r.get('native_dropout_calls')==5 and
            r.get('native_dropout_slots')==5*128*160 and r.get('private_draws')==r.get('overridden_calls')==4 and
            r.get('private_draw_slots')==r.get('overridden_slots')==4*MASK_SLOTS and r.get('overridden_rows_per_call')==64,
            'Only existing base160 fusion with retained native draws and four matched private masks required')
        seeds=r.get('mask_seeds');kept=r.get('private_kept_counts');keys={'As0','As1','raw0','raw1'}
        c.require(isinstance(seeds,dict) and set(seeds)==keys and all(type(v) is int for v in seeds.values()) and len(set(seeds.values()))==4 and
            isinstance(kept,dict) and set(kept)==keys and all(type(v) is int and 0<=v<=MASK_SLOTS for v in kept.values()),
            'Four separate named private streams and complete kept counts required')
        ham=r.get('As_raw_pair_hamming');rd=[r.get(k) for k in ('As_RD_hamming','raw_RD_hamming','unused_raw_RD_hamming')]
        c.require(isinstance(ham,list) and len(ham)==2 and all(type(v) is int and 0<=v<=MASK_SLOTS for v in ham+rd) and
            r.get('paired_same')==all(v==0 for v in ham) and r.get('RD_different')==(rd[0]>0 and rd[1]>0),
            'Saved raw/As and RDrop mask Hamming counts must reconcile')
        if mode=='SHARED_RAW_AS_FUSE':c.require(ham==[0,0] and rd[0]==rd[1],'Corresponding raw/As masks must be identical, not complementary')
        else:c.require(rd[1]==rd[2],'Independent raw RDrop masks must actually be used')
        calls=r.get('call_records');c.require(isinstance(calls,list) and len(calls)==5,'All five original scoped calls required')
        for index,call in enumerate(calls):
            c.require(call.get('index')==index and call.get('native_dropout_executed') is True,'Every original native Dropout call must execute')
            if index==0:c.require(call.get('role')=='ordinary' and call.get('overridden_rows')==0,'Ordinary forward must remain untouched')
            else:
                role,pair=('As',index-1) if index<=2 else ('raw',index-3)
                used='As' if role=='raw' and mode=='SHARED_RAW_AS_FUSE' else role
                c.require(call.get('role')==role and call.get('pair')==pair and call.get('overridden_rows')==64 and
                    call.get('used_mask_role')==used and call.get('used_mask_seed')==seeds[used+str(pair)] and
                    call.get('independent_role_seed')==seeds[role+str(pair)],'Physical counterpart and RDrop pair mask mapping differs')
        c.require(r.get('As_origin_verified') is True and r.get('As_origin_count')==64 and r.get('same_IQ_each_RD_pair') is True and
            r.get('As_redraws')==r.get('new_channel_calls')==r.get('extra_NN')==r.get('persistent_mask_state')==r.get('new_parameters')==0 and
            r.get('CE_KL_unchanged') is True,'Original physical pairing, CE/KL and zero extra work required')
        point=dict(step=step,cumulative_step=step+4440,epoch=epoch,cumulative_epoch=epoch+20,
            raw_As_hamming=sum(ham),As_RD_hamming=rd[0],raw_RD_hamming=rd[1],unused_raw_RD_hamming=rd[2],
            raw_As_equal_fraction=1-sum(ham)/(2*MASK_SLOTS),RDrop_pair_equal_fraction=1-(rd[0]+rd[1])/(2*MASK_SLOTS))
        steps.append(point);bucket=epochs.setdefault(epoch,dict(epoch=epoch,steps=0,raw_As_hamming=0,As_RD_hamming=0,raw_RD_hamming=0))
        bucket['steps']+=1
        for key in ('raw_As_hamming','As_RD_hamming','raw_RD_hamming'):bucket[key]+=point[key]
        for key in keys:private_kept[key]+=kept[key]
    c.require(len(steps)==expected_updates,'All mask-step records required')
    for bucket in epochs.values():
        n=bucket['steps']*2*MASK_SLOTS
        bucket.update(raw_As_equal_fraction=1-bucket['raw_As_hamming']/n,
            RDrop_pair_equal_fraction=1-(bucket['As_RD_hamming']+bucket['raw_RD_hamming'])/n)
    return dict(method=mode,target=TARGET,dropout_p=.45,base_width=160,steps=len(steps),As_packets=64*len(steps),
        native_dropout_calls=5*len(steps),native_dropout_slots=5*128*160*len(steps),private_draws=4*len(steps),
        private_draw_slots=4*MASK_SLOTS*len(steps),private_kept_counts=private_kept,
        private_keep_fractions={k:v/(MASK_SLOTS*len(steps)) for k,v in private_kept.items()},
        epoch_summaries=list(epochs.values()),physical_raw_As_correspondence=True,all_RD_pairs_distinct_streams=True,
        native_draws_retained=True,mask_statistics_are_not_accuracy=True,extra_NN=0),steps


def validate_ordered_records(ordered,masks,expected_updates=UPDATES,metadata=None):
    c.require(len(ordered)==len(masks)==expected_updates,'Complete original-order and mask logs required')
    for step,(r,m) in enumerate(zip(ordered,masks),1):
        indices=list(range(64*(step-1),64*step));slots=[i for i in range(128) if (i+step)%2!=0]
        c.require(r.get('step')==m.get('step')==step and r.get('used_pool_indices')==m.get('used_pool_indices')==indices and
            r.get('origin_verified') is True and r.get('no_quality_weighting') is True and
            r.get('order')=='ORIGINAL_GENERATION_ORDER' and r.get('selected_raw_indices')==m.get('selected_raw_indices')==slots,
            'Every original As64 occurrence must be consumed once in native order and raw slot')
        for key in ('physical_ids','source_labels','source_domains'):
            c.require(r.get(key)==m.get(key) and len(r.get(key,[]))==64,'Actual source labels/domains/IDs must move with paired IQ: '+key)
        c.require(all(type(v) is int and 0<=v<6 for v in m['source_labels']) and
            all(type(v) is int and 0<=v<15 for v in m['source_domains']) and
            all(isinstance(v,list) and len(v)==5 and all(type(z) is int for z in v) for v in m['physical_ids']),
            'Only visible true L labels and five-field physical metadata required')
        if metadata is not None:
            for key,array in (('physical_ids','physical_ids'),('source_labels','labels'),('source_domains','domains')):
                c.require(np.array_equal(r[key],metadata[array][indices]),'Original nominal pool pairing metadata differs')
        c.require(abs(c.finite(r['As_CE_weighted'],'flat satellite CE')-c.finite(r['As_CE_unweighted'],'ordinary satellite CE'))<3e-5 and
            abs(c.finite(r['CE_correction_to_total'],'CE correction'))<3e-5 and
            abs(c.finite(r['used_total_L'],'actual L')-c.finite(r['original_total_L'],'original L'))<3e-5 and
            r.get('As_KL_quality_weighted') is False,'No CE/KL risk change or new loss is allowed')
    return dict(steps=expected_updates,observations=64*expected_updates,one_use=True,original_generation_order=True,
        labels_domains_physical_IDs_match=True,no_quality_weighting=True,new_simulations=0)


def mask_readout(folder,mode,project):
    folder=Path(folder);initial=c.read(folder/'mask_initialization.json');actual=c.read(folder/'mask_training_actual.json');result=c.read(folder/'result.json')
    c.require(initial.get('nominal_pool_reused') is True and initial.get('original_generation_order') is True and
        initial.get('fixed_before_first_update') is True and initial.get('V_read') is False and initial.get('target_read') is False and
        initial.get('new_channel_calls')==initial.get('extra_neural_forward')==0,'Read-only original pool must bind before all updates')
    policy=initial['mask_coupling'];c.require(policy.get('mode')==mode and policy.get('module')==TARGET and policy.get('p')==.45 and policy.get('width')==160 and
        policy.get('RDrop_pairs_independent') is True and policy.get('native_global_RNG_draws_preserved') is True,'Fixed base160-only coupling required')
    c.require(actual.get('satellite_CE_coefficient')==actual.get('satellite_KL_coefficient')==.25 and
        actual.get('original_expected_risk_unchanged') is True and actual.get('ordinary_raw_U_objectives_unchanged') is True and
        actual.get('extra_NN')==actual.get('real_channel_regeneration')==0,'Original expected per-view CE/KL/native U required')
    records=list(c.lines(folder/'mask_steps.jsonl.gz'));ordered=list(c.lines(folder/'ordered_steps.jsonl.gz'))
    summary,flat=validate_mask_records(records,mode)
    pool=Path(project)/'runs'/POOL_RUN/'nominal_pool'
    # These are only fixed labels/domains/physical-ID metadata arrays, never IQ.
    metadata={key:np.load(pool/(key+'.npy'),allow_pickle=False) for key in ('physical_ids','labels','domains')}
    summary['ordered_pool']=validate_ordered_records(ordered,records,metadata=metadata)
    reader=result['ordered_pool'];c.require(reader==c.read(folder/'ordered_pool_final.json') and
        reader.get('origin_validated_steps')==reader.get('calls')==UPDATES and reader.get('origin_validated_observations')==reader.get('packets')==AS_PACKETS and
        reader.get('consumed_pool_range')==[0,AS_PACKETS] and reader.get('complete') is True and reader.get('one_use') is True and
        reader.get('new_channel_calls')==reader.get('pool_generated_this_run')==0 and reader.get('no_quality_weighting') is True,
        'Actual cache consumption must be complete, original-order and simulation-free')
    c.require(result.get('mask_record_count')==UPDATES and result.get('mask_coupling')==policy and result.get('nominal_pool_reused') is True and
        result.get('new_channel_calls')==result.get('pool_generated_this_run')==0,'Final row mask/cache counters must reconcile')
    summary.update(initialization=initial,training_actual=actual,reader_summary=reader,
        pool_load_seconds=c.finite(initial['pool_load_seconds'],'pool load wall'),new_simulations=0)
    return summary,flat


def training_analysis(project,specs,out):
    tails={};curves=[];coverage=[];mask_rows=[];by_key={s['key']:s for s in specs}
    for spec in specs:
        key=spec['key'];tail,ordinary,_=q.parse_tail(spec['root']/'source',spec['arm']=='B01',False)
        if not spec['reused']:
            readout,steps=mask_readout(spec['root']/'source',MODES[key],project);tail['mask_readout']=readout
            mask_rows.extend(dict(row_id=key,**r) for r in steps)
            tail['physical_sequence']=q.same_sequence(spec['root']/'source/training_sequence.jsonl.gz',by_key['AS30']['root']/'source/training_sequence.jsonl.gz')
            c.require(ordinary==tails['AS30']['ordinary'],'Original ordinary augmentation stream must be preserved')
        tail['ordinary']=ordinary;result=c.read(spec['root']/'source/result.json')
        if spec['arm']=='B01':
            channel=result['L_channel'];cells=channel['strata']
            c.require(channel.get('packets')==AS_PACKETS and channel.get('session_count')==15 and sum(r['count'] for r in cells)==AS_PACKETS,
                'Original full labeled As coverage required')
            coverage.extend(dict(row_id=key,**r) for r in cells)
            if not spec['reused']:
                joint=result['TX_RX_scene_state_lock'];c.require(sum(r['count'] for r in joint)==AS_PACKETS and
                    len({(r['TX'],r['RX'],r['scene']) for r in joint})==90,'All true TX x source RX x scene support required')
                c.csv_write(out/(key+'_TX_RX_scene_state_lock.csv'),joint)
        curves.extend(dict(row_id=key,reused=spec['reused'],model_seed=SEED,**r) for r in c.epoch_curve(spec['parent'],0)+tail['curve']);tails[key]=tail
    first=list(c.lines(by_key['MASK_INDEPENDENT']['root']/'source/mask_steps.jsonl.gz'))
    second=list(c.lines(by_key['MASK_SHARED']['root']/'source/mask_steps.jsonl.gz'))
    for a,b in zip(first,second):
        c.require(a['mask_seeds']==b['mask_seeds'] and a['private_kept_counts']==b['private_kept_counts'] and
            a['As_RD_hamming']==b['As_RD_hamming'] and a['unused_raw_RD_hamming']==b['unused_raw_RD_hamming'],
            'Both arms must retain the same As and unused independent-raw private draws')
    for tail in tails.values():tail.pop('ordinary')
    c.csv_write(out/'training_curves.csv',curves);c.csv_write(out/'mask_steps.csv',mask_rows);c.csv_write(out/'training_As_coverage.csv',coverage)
    value=dict(status='FULL_TRAINING_LOGS_VERIFIED',rows=tails,new_model_epochs=20,new_updates=4440,
        unique_shared_parent_model_epochs=40,unique_total_model_epochs=80,extra_NN=0,U_truth_read=False,
        no_new_physical_information=True,mask_coupling_is_not_identity_gain=True,new_simulations=0,
        both_arms_same_named_private_draws=True,historical_nominal_pool_generated=AS_PACKETS)
    c.write(out/'training_summary.json',value);return value,curves


def expected_new_NN(smoke_nn=4):
    c.require(type(smoke_nn) is int and smoke_nn>=0,'Nonnegative actual smoke counter required')
    return 2*(3162962+12600+1176002)+smoke_nn


def costs(project,specs,local_nn=0,smoke_nn=4):
    c.require(local_nn==0,'No local neural verification in this round')
    rows={};new={}
    for spec in specs:
        source=c.read(spec['root']/'source_qualification/complete.json');prediction=c.read(spec['root']/'prediction/complete.json')
        c.require(source.get('NN')==12600 and prediction.get('actual_NN')==1176002,'Saved source/formal counters differ')
        row=dict(reused=spec['reused'],parent=c.native_cost(spec['parent']),tail=c.native_cost(spec['root']/'source'),source_NN=12600,
            formal_NN=1176002,source_wall_seconds=source.get('wall_seconds'),formal_wall_seconds=prediction.get('wall_seconds'))
        rows[spec['key']]=row
        if not spec['reused']:
            row['pool_load_seconds']=c.read(spec['root']/'source/mask_initialization.json')['pool_load_seconds'];new[spec['key']]=row
    smoke=c.read(Path(project)/'runs'/RUN/'real_checkpoint_smoke.json')
    c.require(smoke.get('status')=='PASS' and smoke.get('NN')==smoke_nn and len(new)==2 and all(
        r['tail']['native_NN']==r['tail']['training_total_NN']==3162962 and r['tail']['extra_student_U_NN']==r['tail']['extra_frozen_teacher_U_NN']==0
        for r in new.values()),'Two native-only tails and one actual no-query smoke required')
    return dict(rows=rows,new_native_NN=6325924,new_source_NN=25200,new_formal_NN=2352004,shared_remote_smoke_NN=smoke_nn,
        new_total_remote_NN=expected_new_NN(smoke_nn),local_verification_NN=0,new_model_epochs=20,new_updates=4440,
        unique_shared_parent_model_epochs=40,unique_total_model_epochs_including_reused=80,
        student_cumulative_epochs_each=30,student_cumulative_updates_each=6660,parent_training_reused_not_repeated=True,
        new_training_wall_seconds_sum=sum(r['tail']['wall_seconds'] for r in new.values()),
        extra_student_NN=0,extra_teacher_NN=0,reference_reprediction=0,reference_rescoring=0,analysis_NN=0,
        historical_parent_source_cohort_NN=12602,parent_cohort_new_NN=0,budget_exhaustion_is_not_convergence=True,
        new_simulations=0,historical_nominal_pool_generated=AS_PACKETS,
        pool_load_seconds_by_row={k:r['pool_load_seconds'] for k,r in new.items()},
        mask_private_draw_slots_each=UPDATES*4*MASK_SLOTS,mask_native_draw_slots_in_scope_each=UPDATES*5*128*160)


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
    for old,label in (('MASK_INDEPENDENT','Shared-independent'),('AS30','Shared-original As'),('NOAS30','Shared-NoAs')):
        axes[0].plot(xx,[100*(formal['rows']['MASK_SHARED'][v]['accuracy']-formal['rows'][old][v]['accuracy']) for v in c.VIEWS],marker='.',label=label)
    for key,label in (('macro_f1','Macro-F1'),('worst_RX','Worst RX')):
        axes[1].plot(xx,[100*(formal['rows']['MASK_SHARED'][v][key]-formal['rows']['MASK_INDEPENDENT'][v][key]) for v in c.VIEWS],marker='.',label=label)
    for ax,title in zip(axes,('Formal paired accuracy changes','Shared-independent paired F1 / worst RX')):
        ax.set(xticks=xx,xticklabels=['clean','high sub','mid sub','low sub','high urb','mid urb','low urb'],title=title,ylabel='Absolute percentage points');ax.tick_params(axis='x',labelrotation=30);ax.axhline(0,color='grey',lw=.7);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    save(fig,'formal_paired')
    if training:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for arm in ARMS[2:]:
            readout=training['rows'][arm]['mask_readout'];points=readout['epoch_summaries']
            for ax,key,title in zip(axes,('raw_As_equal_fraction','RDrop_pair_equal_fraction'),('Physical raw / As mask equality','Within-view RDrop-pair mask equality')):
                ax.plot([r['epoch']+20 for r in points],[r[key] for r in points],label=arm,color=colors[arm]);ax.set(title=title,xlabel='Cumulative epoch');ax.legend(fontsize=8);ax.grid(alpha=.2)
        save(fig,'raw_As_mask_readout')
    return files


def report(out,result):
    formal=result['formal'];rows=formal['rows'];primary=formal['primary']['effects'];stage=formal['stage']['MASK_SHARED'];cost=result['cost']
    text=['# P72：现有 base160 融合 Dropout 的 raw／As 相关性','',
        '两行均从原P43 As20 student/ownEMA独立追加10轮，重置AdamW并使用相同cos10。只在现有fuse.2的base160 Dropout上比较独立掩码和对应物理raw/As共享掩码；RDrop两次之间仍独立。作用点不是所有最终分支的joint特征。其余原生随机抽样、CE/KL、ordinary/raw、ORD128 SupCon、native U和默认cosine30分类头保持；不增加物理信息、部署结构或新损失。','',
        '| 环境 | NoAs30 % | 原As30 % | 独立mask % | raw/As同mask % | 共享−独立 pp |','|---|---:|---:|---:|---:|---:|']
    for view in c.VIEWS:text.append('| '+view+' | '+' | '.join(f'{100*rows[arm][view]["accuracy"]:.6f}' for arm in ARMS)+f' | {primary[view]["accuracy_delta_pp"]:+.6f} |')
    text.extend(['',f'主要比较共享−独立：clean {primary["clean"]["accuracy_delta_pp"]:+.6f} pp，mid urban {primary[c.VIEWS[-2]]["accuracy_delta_pp"]:+.6f} pp，low urban {primary[c.VIEWS[-1]]["accuracy_delta_pp"]:+.6f} pp，urban均值 {primary["urban_accuracy_mean_delta_pp"]:+.6f} pp。',
        f'阶段比较共享−NoAs30：clean {stage["clean_delta_pp"]:+.6f} pp，urban均值 {stage["urban_mean_delta_pp"]:+.6f} pp；clean下降≤0.1绝对百分点：{stage["clean_within_0p1pp"]}。',
        '仅一个模型seed；两个源信道draw不是模型复现。单seed或不足0.1个百分点的正变化不能证明可重复双提升。四行不是2×2因子矩阵。', '',
        'source仅两项：共享对NoAs30 fullV clean≥−0.1pp；pooled mid/low共享对独立≥+0.5pp。判定先于正式读取固定；任一FAIL关闭该配方且不扩E50，全部七个正式环境仍完成，不按正式成绩重新晋级。','',
        '全部accuracy、Macro-F1、最差RX、RX/day/TX、修复/新增错误、margin/30和固定NoAs30 clean正确集合保持率见CSV。精确原P43 As20 student源clean正确集合另列；状态/lock/质量分层只在源域可用，不作物理因果归因。新增噪声比不是总SNR，mask相关性与loss变化不是身份收益。','',
        f'新增20 model-epochs／4440更新，复用两父及两个旧尾段后四行独有训练总计80 model-epochs。新增native NN={cost["new_native_NN"]}，source={cost["new_source_NN"]}，formal={cost["new_formal_NN"]}，smoke={cost["shared_remote_smoke_NN"]}，远端总计={cost["new_total_remote_NN"]}；本地和分析NN均0。新增训练行墙钟之和={cost["new_training_wall_seconds_sum"]:.3f}秒。模拟生成／缓存与mask算子成本另列，预算耗尽不等于收敛。','',
        '已固定源判定：'+json.dumps(result['stored_source_decision'],ensure_ascii=False),'',
        '未完成／限制：'+json.dumps(result['missing'],ensure_ascii=False),'','实际图表：'])
    text.extend('- '+path for path in result['figures']);text.extend(['','分析自身无模型、IQ/checkpoint读取、训练、重预测或重新评分。'])
    (out/'final_analysis.md').write_text('\n'.join(text)+'\n',encoding='utf8')


def run(project,out,truth=None,code_commit=None,local_nn=0,smoke_nn=4):
    project=Path(project);decision=read_source_decision(project);out=Path(out)
    c.require(not out.exists(),'Immutable analysis output required');out.mkdir(parents=False)
    c.write(out/'stored_source_decision.json',decision);started=time.perf_counter();specs=row_specs(project)
    missing=[];training=None;curves=[];source=None;source_rows=[];parent=None
    if decision['status'] in c.DECISION_PARTIAL:missing.append(dict(component='source_decision',error='Stored decision PARTIAL; not recomputed'))
    for name in ('training','source','parent_cohort'):
        try:
            if name=='training':training,curves=training_analysis(project,specs,out)
            elif name=='source':source,source_rows=c.source_analysis(project,specs,out,decision)
            else:parent=h.parent_cohort_analysis(project,specs,out)
        except Exception as error:
            failure=dict(component=name,error=repr(error),traceback=traceback.format_exc(),formal_still_required=True)
            missing.append(failure);c.write(out/(name+'_failure.json'),failure)
    truth=Path(truth) if truth else project/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
    formal=c.formal_analysis(project,specs,out,truth);cost=costs(project,specs,local_nn,smoke_nn);c.write(out/'costs.json',cost)
    try:figures=plots(out,curves,source_rows,formal,training)
    except Exception as error:
        figures=[];failure=dict(component='plots',error=repr(error),traceback=traceback.format_exc());missing.append(failure);c.write(out/'plot_failure.json',failure)
    result=dict(status='ANALYZED' if not missing else 'FORMAL_ANALYZED_WITH_PARTIAL_COMPONENTS',run_id=RUN,code_commit=code_commit,
        stored_source_decision=decision,training=training,source=source,parent_cohort=parent,formal=formal,cost=cost,figures=figures,missing=missing,
        model_seed_count=1,crossseed_confirmed=False,reproducible_dual_gain=False,target_feedback=False,source_decision_recomputed=False,goal_complete=False,
        NN=0,IQ_reads=0,checkpoint_reads=0,training_updates=0,wall_seconds=time.perf_counter()-started)
    report(out,result);c.write(out/'summary.json',{k:v for k,v in result.items() if k!='training'});c.write(out/'complete.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--truth',type=Path)
    parser.add_argument('--code-commit');parser.add_argument('--local-nn',type=int,default=0);parser.add_argument('--smoke-nn',type=int,default=4)
    parser.add_argument('--checks',action='store_true');args=parser.parse_args()
    if args.checks:
        path=Path(__file__).with_name('cvs_raw_as_mask72_analysis_checks_20261019.py')
        spec=importlib.util.spec_from_file_location('_p72_checks',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.checks();return
    if args.project is None or args.out is None:parser.error('--project and --out required unless --checks')
    result=run(args.project,args.out,args.truth,args.code_commit,args.local_nn,args.smoke_nn)
    print(json.dumps(dict(status=result['status'],out=str(args.out),missing=[r['component'] for r in result['missing']],NN=0,IQ_reads=0,checkpoint_reads=0)),flush=True)


if __name__=='__main__':main()
