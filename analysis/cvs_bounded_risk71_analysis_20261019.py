"""P71 bounded true-label As CE risk: saved-data CPU readout, zero NN/IQ/CP."""
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

RUN='20261019-phase1-FcrossRX-boundedRisk-e30-s2026092701-m1-r01'
POOL_RUN='20261018-phase1-FcrossRX-qualityStrat-e30-s2026092701-m2-r01'
REF='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
SEED,POOL_N,UPDATES=2026092701,142080,2220
GAMMA=.25
V_G=(.15,.40,.25,.20)
ARMS=('NOAS30','AS30','QUALITY_STRAT','BOUNDED_RISK')
PAIRS=(('AS30','NOAS30'),('QUALITY_STRAT','AS30'),('BOUNDED_RISK','QUALITY_STRAT'),
       ('BOUNDED_RISK','AS30'),('QUALITY_STRAT','NOAS30'),('BOUNDED_RISK','NOAS30'))


def load_common():
    path=Path(__file__).with_name('cvs_quality_risk70_analysis_20261018.py')
    spec=importlib.util.spec_from_file_location('_p71_common70',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


p=load_common(); h=p.h; q=p.q; c=p.c


def row_specs(project):
    root=Path(project)/'runs'; result=[]
    for key,run,arm,reused in (('NOAS30',REF,'B00',True),('AS30',REF,'B01',True),
        ('QUALITY_STRAT',POOL_RUN,'M2',True),('BOUNDED_RISK',RUN,'M3',False)):
        parent='B00' if key=='NOAS30' else 'B01'
        result.append(dict(key=key,run=run,arm=parent,actual_arm=arm,row_id=arm+'-s'+str(SEED),reused=reused,
            root=root/run/(arm+'-s'+str(SEED)),parent=root/REF/(parent+'-s'+str(SEED))/'stage20'))
    return result


def training_complete(spec,project):return p.training_complete(spec,project)


def risk_weights(p_g):
    nominal=np.asarray(p_g,float); v=np.asarray(V_G)
    c.require(nominal.shape==(4,) and np.isfinite(nominal).all() and np.all(nominal>0) and
        abs(float(nominal.sum())-1)<1e-12,'The fixed P70 pool must support all four risk groups')
    pi=(1-GAMMA)*nominal+GAMMA*v
    return dict(p_g=nominal.tolist(),v_g=v.tolist(),pi_g=pi.tolist(),gamma=GAMMA,
        relative_to_nominal=(pi/nominal).tolist(),As_CE_group_coefficients=(.25*pi).tolist(),
        As_CE_total_coefficient=.25,As_KL_quality_weighted=False)


def nominal_record(record):
    """Same M3 logits under the M2 objective; never the old M2 model's loss."""
    r=dict(record); nominal=c.finite(r['nominal_M2_As_CE'],'same-logits nominal As CE')
    r.update(method='QUALITY_STRAT',As_CE_weighted=nominal,
        CE_correction_to_total=.25*(nominal-r['As_CE_unweighted']),
        used_total_L=r['nominal_M2_same_logits_total_L'])
    return r


def validate_risk_steps(records,counts,m2_records,expected_updates=UPDATES,batch=64,pool_groups=None):
    counts=np.asarray(counts); fixed=risk_weights(counts/counts.sum())
    c.require(len(records)==len(m2_records)==expected_updates,'Every M3 and reused M2 quality step required')
    normalized=[nominal_record(r) for r in records]
    allocation,base=p.validate_pool_steps(normalized,counts,'QUALITY_STRAT',expected_updates,batch,pool_groups)
    flat=[]
    for r,m2,point in zip(records,m2_records,base):
        c.require(r.get('method')=='BOUNDED_RISK' and r.get('allocation_method')=='QUALITY_STRAT' and
            r.get('gamma')==GAMMA and np.allclose(r.get('v_g'),V_G,atol=1e-14,rtol=0) and
            np.allclose(r.get('pi_g'),fixed['pi_g'],atol=1e-14,rtol=0),'One fixed bounded risk formula required; no scan/adaptation')
        for key in ('used_pool_indices','counts','groups','origin_steps','origin_slots'):
            c.require(r.get(key)==m2.get(key),'M3 must use exactly the reused M2 nominal observations in the same order: '+key)
        means=np.asarray(r['group_CE'],float)
        c.require(means.shape==(4,) and np.isfinite(means).all(),'All positive-mass group CE means required')
        nominal=float(np.dot(fixed['p_g'],means)); risk=float(np.dot(fixed['pi_g'],means)); increment=.25*(risk-nominal)
        c.require(abs(r['nominal_M2_As_CE']-nominal)<3e-5 and
            abs(c.finite(r.get('used_risk_As_CE'),'risk As CE')-risk)<3e-5 and
            abs(r['As_CE_weighted']-risk)<3e-5 and
            abs(c.finite(r.get('risk_vs_nominal_total_correction'),'risk minus nominal correction')-increment)<3e-5 and
            abs(r['used_total_L']-r['nominal_M2_same_logits_total_L']-increment)<3e-5 and
            abs(r['CE_correction_to_total']-.25*(risk-r['As_CE_unweighted']))<3e-5,
            'Actual risk, nominal same-logits counterfactual and .25 CE-only increment must reconcile')
        point.update(risk_CE=risk,nominal_M2_CE=nominal,risk_vs_nominal_total_correction=increment,
            used_total_L=r['used_total_L'],nominal_M2_same_logits_total_L=r['nominal_M2_same_logits_total_L'])
        flat.append(point)
    allocation.update(fixed,method='BOUNDED_RISK',allocation_method='QUALITY_STRAT',same_M2_pool_order=True,
        risk_CE_mean=float(np.mean([r['risk_CE'] for r in flat])),
        nominal_M2_same_logits_CE_mean=float(np.mean([r['nominal_M2_CE'] for r in flat])),
        risk_vs_nominal_total_correction_mean=float(np.mean([r['risk_vs_nominal_total_correction'] for r in flat])),
        fixed_finite_pool_risk=False,fixed_bounded_group_risk=True,nominal_risk_unbiased_claimed=False,
        counterfactual_is_same_logits_not_reused_M2_model=True)
    return allocation,flat


def risk_readout(project,spec,m2_spec,m2_records):
    folder=spec['root']/'source'; initialization=c.read(folder/'risk_initialization.json')
    actual=c.read(folder/'risk_training_actual.json'); result=c.read(folder/'result.json')
    shared=Path(project)/'runs'/POOL_RUN/'nominal_pool'; summary=c.read(shared/'complete.json')
    c.require(summary.get('status')=='READY' and summary.get('N')==POOL_N and
        summary.get('written')==POOL_N and summary.get('extra_proposals')==summary.get('dropped')==0,
        'Only the already-fixed P70 nominal pool may be reused')
    c.require(initialization.get('fixed_before_first_update') is True and initialization.get('nominal_pool_reused') is True and
        initialization.get('V_read') is False and initialization.get('target_read') is False and
        initialization.get('new_channel_calls')==initialization.get('source_IQ_generated')==initialization.get('extra_neural_forward')==0,
        'No V/target, regeneration or extra NN during pool binding')
    initial_reader=initialization['reader']; fixed=risk_weights(summary['p_g'])
    c.require(initial_reader.get('calls')==initial_reader.get('packets')==0 and initial_reader.get('pool_N')==POOL_N and
        np.allclose(initial_reader.get('p_g'),summary['p_g'],atol=1e-14,rtol=0) and
        np.allclose(initial_reader.get('pi_g'),fixed['pi_g'],atol=1e-14,rtol=0),
        'Risk reader must bind the original pool before its first consumption')
    c.require(actual.get('satellite_CE_coefficient')==.25 and actual.get('satellite_KL_coefficient')==.25 and
        actual.get('extra_NN')==actual.get('real_channel_regeneration')==0 and actual.get('ordinary_raw_U_unchanged') is True,
        'Original CE/KL mass and unchanged ordinary/raw/U work required')
    # Only integer bin metadata; iq.npy is deliberately never opened.
    groups=np.load(shared/'groups.npy',allow_pickle=False)
    c.require(groups.shape==(POOL_N,) and groups.dtype.kind in 'iu' and np.all((groups>=0)&(groups<4)) and
        np.array_equal(np.bincount(groups,minlength=4),summary['group_counts']),'Existing nominal pool bin metadata differs')
    records=list(c.lines(folder/'risk_steps.jsonl.gz'))
    readout,flat=validate_risk_steps(records,summary['group_counts'],m2_records,pool_groups=groups)
    reader=result['risk']
    c.require(reader==c.read(folder/'risk_final.json') and reader.get('calls')==UPDATES and
        reader.get('packets')==reader.get('pool_N')==reader.get('unique_occurrences_used')==POOL_N and
        reader.get('group_counts')==summary['group_counts'] and reader.get('new_channel_calls')==reader.get('dropped')==0,
        'One-use saved reader summary must reconcile with every risk step')
    c.require(result.get('new_channel_calls')==result.get('pool_generated_this_run')==0 and result.get('nominal_pool_reused') is True,
        'No new simulation is permitted in this risk-only experiment')
    readout.update(initialization=initialization,pool_summary=summary,reader_summary=reader,training_actual=actual,
        load_or_prepare_seconds=c.finite(initialization['pool_load_seconds'],'pool load wall'),
        new_simulations=0,new_pool_generation=False)
    return readout,flat,records


def effects(rows,new,old):return p.effects(rows,new,old)


def enrich_formal(obj):
    rows=obj['rows']; obj['comparisons']={new+'__'+old:effects(rows,new,old) for new,old in PAIRS}
    obj['primary']=dict(new='BOUNDED_RISK',control='QUALITY_STRAT',effects=effects(rows,'BOUNDED_RISK','QUALITY_STRAT'))
    obj['stage']={}
    for arm in ('AS30','QUALITY_STRAT','BOUNDED_RISK'):
        value=effects(rows,arm,'NOAS30'); clean=value['clean']['accuracy_delta_pp']
        obj['stage'][arm]=dict(control='NOAS30',clean_delta_pp=clean,mid_urban_delta_pp=value[c.VIEWS[-2]]['accuracy_delta_pp'],
            low_urban_delta_pp=value[c.VIEWS[-1]]['accuracy_delta_pp'],urban_mean_delta_pp=value['urban_accuracy_mean_delta_pp'],
            clean_within_0p1pp=bool(clean>=-.1-1e-10),single_seed_observed_dual_gain=bool(clean>0 and value['urban_accuracy_mean_delta_pp']>0),
            reproducible_dual_gain=False)
    obj['four_rows_are_not_a_two_by_two_factorial']=True


def write(path,obj):
    if Path(path).name=='formal_summary.json':enrich_formal(obj)
    q._write(path,obj)


c.RUN,c.NEW_ARMS,c.PAIRS,c.CANDIDATE=RUN,('BOUNDED_RISK',),PAIRS,'BOUNDED_RISK'
c.row_specs,c.training_complete,c.write=row_specs,training_complete,write


def source_criteria(clean_m3,clean_noas,midlow_m3,midlow_m2):
    clean=100*(clean_m3-clean_noas); gain=100*(midlow_m3-midlow_m2)
    checks=dict(fullV_clean_M3_vs_NoAs_ge_minus_0p1pp=clean>=-.1-1e-10,pooled_midlow_M3_vs_M2_ge_0p5pp=gain>=.5-1e-10)
    return dict(checks=checks,source_qualified=all(checks.values()),clean_delta_pp=clean,midlow_delta_pp=gain)


def read_source_decision(project):
    value=c.read(Path(project)/'runs'/RUN/'source_matrix_decision.json')
    c.require(value.get('status') in c.DECISION_COMPLETE|c.DECISION_PARTIAL and value.get('formal_scores_read') is False and
        value.get('target_feedback') is False and value.get('NN')==0 and value.get('model_seed_count')==1,
        'Stored source-only decision must precede formal reads')
    if value['status'] in c.DECISION_COMPLETE:
        checks=value.get('checks'); c.require(isinstance(checks,dict) and len(checks)==2 and
            all(type(x) is bool for x in checks.values()) and type(value.get('source_qualified')) is bool and
            value['source_qualified']==all(checks.values()),'Only the two fixed source criteria are allowed')
    return value


def training_analysis(project,specs,out):
    tails={}; curves=[]; coverage=[]; risk_rows=[]; by_key={s['key']:s for s in specs}
    m2_records=list(c.lines(by_key['QUALITY_STRAT']['root']/'source/quality_steps.jsonl.gz'))
    for spec in specs:
        key=spec['key']
        if key in ('NOAS30','AS30'):tail,ordinary,_=q.parse_tail(spec['root']/'source',spec['arm']=='B01',False)
        elif key=='QUALITY_STRAT':
            readout,_,records=p.pool_readout(project,spec); tail,ordinary=p.parse_new_tail(spec,records)
            tail['quality_pool_readout']=readout
        else:
            readout,flat,records=risk_readout(project,spec,by_key['QUALITY_STRAT'],m2_records)
            tail,ordinary=p.parse_new_tail(spec,records); tail['quality_pool_readout']=readout
            risk_rows.extend(dict(row_id=key,**r) for r in flat)
            for point in tail['curve']:
                rr=[r for r in flat if r['epoch']==point['epoch']]
                point.update(nominal_M2_same_logits_CE=float(np.mean([r['nominal_M2_CE'] for r in rr])),
                    risk_vs_nominal_total_correction=float(np.mean([r['risk_vs_nominal_total_correction'] for r in rr])))
                point.update({'group_CE_'+str(g):float(np.mean([r['CE_'+str(g)] for r in rr])) for g in range(4)})
        if key in ('QUALITY_STRAT','BOUNDED_RISK'):
            tail['physical_sequence']=q.same_sequence(spec['root']/'source/training_sequence.jsonl.gz',by_key['AS30']['root']/'source/training_sequence.jsonl.gz')
            c.require(ordinary==tails['AS30']['ordinary'],'Original ordinary augmentation stream must remain fixed')
        tail['ordinary']=ordinary; result=c.read(spec['root']/'source/result.json')
        if spec['arm']=='B01':
            channel=result['L_channel']; cells=channel['strata']
            c.require(channel.get('packets')==POOL_N and channel.get('session_count')==15 and sum(r['count'] for r in cells)==POOL_N,
                'All original labeled nominal As observations must remain covered')
            coverage.extend(dict(row_id=key,**r) for r in cells)
            if key in ('QUALITY_STRAT','BOUNDED_RISK'):
                joint=result['TX_RX_scene_state_lock']; c.require(sum(r['count'] for r in joint)==POOL_N and
                    len({(r['TX'],r['RX'],r['scene']) for r in joint})==90,'All true TX x source RX x train scene support required')
                c.csv_write(out/(key+'_TX_RX_scene_state_lock.csv'),joint)
        curves.extend(dict(row_id=key,reused=spec['reused'],model_seed=SEED,**r) for r in c.epoch_curve(spec['parent'],0)+tail['curve'])
        tails[key]=tail
    for tail in tails.values():tail.pop('ordinary')
    c.csv_write(out/'training_curves.csv',curves); c.csv_write(out/'risk_steps.csv',risk_rows)
    c.csv_write(out/'training_As_coverage.csv',coverage)
    result=dict(status='FULL_TRAINING_LOGS_VERIFIED',rows=tails,new_model_epochs=10,new_updates=2220,
        unique_shared_parent_model_epochs=40,unique_total_model_epochs=80,U_truth_read=False,extra_NN=0,
        new_simulations=0,historical_shared_pool_generated=POOL_N,same_M2_pool_order=True,
        native_U_temporal_mode='batch_neighbor',risk_changes_target=True,nominal_unbiased_claimed=False,
        loss_differences_and_coverage_are_not_identity_gain=True)
    c.write(out/'training_summary.json',result); c.write(out/'risk_summary.json',tails['BOUNDED_RISK']['quality_pool_readout'])
    return result,curves


def expected_new_NN(smoke_nn=4):
    c.require(type(smoke_nn) is int and smoke_nn>=0,'Nonnegative actual smoke NN required')
    return 3162962+12600+1176002+smoke_nn


def costs(project,specs,local_nn=0,smoke_nn=4):
    c.require(local_nn==0,'No local neural verification in this round')
    rows={}; new=None
    for spec in specs:
        source=c.read(spec['root']/'source_qualification/complete.json'); prediction=c.read(spec['root']/'prediction/complete.json')
        c.require(source.get('NN')==12600 and prediction.get('actual_NN')==1176002,'Saved source/formal forward counts differ')
        row=dict(reused=spec['reused'],parent=c.native_cost(spec['parent']),tail=c.native_cost(spec['root']/'source'),
            source_NN=12600,formal_NN=1176002,source_wall_seconds=source.get('wall_seconds'),formal_wall_seconds=prediction.get('wall_seconds'))
        rows[spec['key']]=row
        if not spec['reused']:
            c.require(new is None,'Exactly one new M3 training row permitted'); new=row
            row['pool_load_or_prepare_seconds']=c.read(spec['root']/'source/risk_initialization.json')['pool_load_seconds']
    smoke=c.read(Path(project)/'runs'/RUN/'real_checkpoint_smoke.json')
    c.require(smoke.get('status')=='PASS' and smoke.get('NN')==smoke_nn and new is not None and
        new['tail']['native_NN']==new['tail']['training_total_NN']==3162962 and
        new['tail']['extra_student_U_NN']==new['tail']['extra_frozen_teacher_U_NN']==0,'One original-native-only tail and actual shared smoke required')
    historical=c.read(Path(project)/'runs'/POOL_RUN/'nominal_pool/complete.json')
    return dict(rows=rows,new_native_NN=3162962,new_source_NN=12600,new_formal_NN=1176002,shared_remote_smoke_NN=smoke_nn,
        new_total_remote_NN=expected_new_NN(smoke_nn),local_verification_NN=0,new_model_epochs=10,new_updates=2220,
        unique_shared_parent_model_epochs=40,unique_total_model_epochs_including_reused=80,
        student_cumulative_epochs_each=30,student_cumulative_updates_each=6660,parent_training_reused_not_repeated=True,
        new_simulations=0,historical_nominal_pool_generated=POOL_N,historical_pool_generation_wall_seconds=historical.get('wall_seconds'),
        pool_load_or_prepare_seconds=new['pool_load_or_prepare_seconds'],new_training_wall_seconds=new['tail']['wall_seconds'],
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
            target=out/(name+'.'+ext); fig.savefig(target,dpi=180); files.append(str(target))
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
    for old,label in (('QUALITY_STRAT','M3-M2'),('AS30','M3-original As'),('NOAS30','M3-NoAs')):
        axes[0].plot(xx,[100*(formal['rows']['BOUNDED_RISK'][v]['accuracy']-formal['rows'][old][v]['accuracy']) for v in c.VIEWS],marker='.',label=label)
    for key,label in (('macro_f1','Macro-F1'),('worst_RX','Worst RX')):
        axes[1].plot(xx,[100*(formal['rows']['BOUNDED_RISK'][v][key]-formal['rows']['QUALITY_STRAT'][v][key]) for v in c.VIEWS],marker='.',label=label)
    for ax,title in zip(axes,('Formal paired accuracy changes','M3-M2 paired F1 / worst RX')):
        ax.set(xticks=xx,xticklabels=['clean','high sub','mid sub','low sub','high urb','mid urb','low urb'],title=title,ylabel='Absolute percentage points'); ax.tick_params(axis='x',labelrotation=30); ax.axhline(0,color='grey',lw=.7); ax.grid(axis='y',alpha=.2); ax.legend(fontsize=8)
    save(fig,'formal_paired')
    if training:
        readout=training['rows']['BOUNDED_RISK']['quality_pool_readout']; fig,axes=plt.subplots(2,2,figsize=(13,9)); axes=axes.ravel(); xx=np.arange(4)
        for i,(key,label) in enumerate((('p_g','Nominal p'),('v_g','Fixed v'),('pi_g','Bounded risk pi'))):axes[0].bar(xx+(i-1)*.25,readout[key],.25,label=label)
        axes[0].set(xticks=xx,xticklabels=p.GROUPS,title='Fixed As CE group-risk mass',xlabel='Added-noise ratio (dB)'); axes[0].legend(fontsize=8)
        rr=[r for r in curves if r['row_id']=='BOUNDED_RISK' and r['epoch']>20]
        for g,label in enumerate(p.GROUPS):axes[1].plot([r['epoch'] for r in rr],[r['group_CE_'+str(g)] for r in rr],label=label+' dB')
        axes[1].set(title='Per-group same-IQ CE means (not accuracy)',xlabel='Cumulative epoch'); axes[1].legend(fontsize=8)
        for key,label in (('As_CE_used','Risk CE'),('nominal_M2_same_logits_CE','Nominal CE / same logits')):axes[2].plot([r['epoch'] for r in rr],[r[key] for r in rr],label=label)
        axes[2].set(title='M3 same-logits objectives (not accuracy)',xlabel='Cumulative epoch'); axes[2].legend(fontsize=8)
        axes[3].plot([r['epoch'] for r in rr],[r['risk_vs_nominal_total_correction'] for r in rr],color=colors['BOUNDED_RISK'])
        axes[3].set(title='0.25 x (risk CE - nominal CE)',xlabel='Cumulative epoch'); axes[3].axhline(0,color='grey',lw=.7)
        for ax in axes:ax.grid(axis='y',alpha=.2)
        save(fig,'bounded_risk_readout')
    return files


def report(out,result):
    formal=result['formal']; rows=formal['rows']; primary=formal['primary']['effects']; stage=formal['stage']['BOUNDED_RISK']; cost=result['cost']
    text=['# P71：固定 As CE 总量的有界质量组风险','',
        '仅新增M3，严格从原P43 As20 student/ownEMA追加10轮，重置AdamW并采用相同cos10。P70 M2完整模型仅作匹配结果参照，不作为父权重。M3复用P70 nominal池以及M2的观测顺序，每个观测恰用一次，新增模拟为0；ordinary/raw、ORD128 SupCon、native U、原same-IQ RDrop及默认cosine30头保持。','',
        '固定gamma=.25，v=[.15,.40,.25,.20]，pi=.75p+.25v；p来自训练nominal池N_g/N。只有As CE按pi组均值重分配，总系数仍.25；KL保持原未加权.25。本方法明确改变组风险，不称为nominal风险的无偏估计。groupCE、loss差及覆盖不等于身份收益，3至15dB也不是真实可恢复性标签。新增噪声比不是总SNR。','',
        '| 环境 | NoAs30 % | 原As30 % | M2 nominal % | M3 bounded % | M3-M2 pp |','|---|---:|---:|---:|---:|---:|']
    for view in c.VIEWS:text.append('| '+view+' | '+' | '.join(f'{100*rows[arm][view]["accuracy"]:.6f}' for arm in ARMS)+f' | {primary[view]["accuracy_delta_pp"]:+.6f} |')
    text.extend(['',f'主要比较M3-M2：clean {primary["clean"]["accuracy_delta_pp"]:+.6f} pp，mid urban {primary[c.VIEWS[-2]]["accuracy_delta_pp"]:+.6f} pp，low urban {primary[c.VIEWS[-1]]["accuracy_delta_pp"]:+.6f} pp，urban均值 {primary["urban_accuracy_mean_delta_pp"]:+.6f} pp。',
        f'阶段比较M3-NoAs30：clean {stage["clean_delta_pp"]:+.6f} pp，urban均值 {stage["urban_mean_delta_pp"]:+.6f} pp；clean下降≤0.1绝对百分点：{stage["clean_within_0p1pp"]}。',
        '仅一个模型seed；两个源信道draw不是跨seed重复。单seed和小于0.1个百分点的变化不能证明可重复双提升。四行不是2×2因子矩阵。', '',
        'source仅两项：M3对NoAs30 fullV clean≥−0.1pp；pooled mid/low M3对M2≥+0.5pp。判定先于正式结果固定；任一FAIL不扩E50，全部七个正式环境仍完成，不能按正式结果重新晋级。','',
        '同logits的nominal_M2_same_logits_total_L只是在当前M3参数下改回M2损失公式，并非已训练M2模型的loss；original_total_L更早对应当前logits下普通As均值公式，二者分开保存。实际M3-M2身份收益来自固定模型预测的配对评分。','',
        '完整accuracy、Macro-F1、最差RX、RX/day/TX、修复/新增错误、margin/30、固定NoAs30 clean正确集合保持见CSV；精确P43 As20 student源clean正确集合另列。源state/lock/质量分层只作关联，正式状态不可用，不作物理因果归因。','',
        f'本次新增10 model-epochs／2220更新，复用两父与三个旧尾段后四行独有总训练80 model-epochs。新增native NN={cost["new_native_NN"]}，source={cost["new_source_NN"]}，formal={cost["new_formal_NN"]}，smoke={cost["shared_remote_smoke_NN"]}，远端总计={cost["new_total_remote_NN"]}；本地和分析NN均0。旧池历史生成142080次，本次新增0次；训练墙钟={cost["new_training_wall_seconds"]:.3f}秒，pool加载另列。预算耗尽不等于收敛。','',
        '已固定源决定：'+json.dumps(result['stored_source_decision'],ensure_ascii=False),'',
        '未完成／限制：'+json.dumps(result['missing'],ensure_ascii=False),'','实际图表：'])
    text.extend('- '+path for path in result['figures']); text.extend(['','分析自身无训练、IQ/checkpoint读取、重预测或重新评分。'])
    (out/'final_analysis.md').write_text('\n'.join(text)+'\n',encoding='utf8')


def run(project,out,truth=None,code_commit=None,local_nn=0,smoke_nn=4):
    project=Path(project); decision=read_source_decision(project); out=Path(out)
    c.require(not out.exists(),'Immutable analysis output required'); out.mkdir(parents=False)
    c.write(out/'stored_source_decision.json',decision); started=time.perf_counter(); specs=row_specs(project)
    missing=[]; training=None; curves=[]; source=None; source_rows=[]; parent=None
    if decision['status'] in c.DECISION_PARTIAL:missing.append(dict(component='source_decision',error='Stored decision PARTIAL; not recomputed'))
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
        path=Path(__file__).with_name('cvs_bounded_risk71_analysis_checks_20261019.py')
        spec=importlib.util.spec_from_file_location('_p71_checks',path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.checks(); return
    if args.project is None or args.out is None:parser.error('--project and --out required unless --checks')
    result=run(args.project,args.out,args.truth,args.code_commit,args.local_nn,args.smoke_nn)
    print(json.dumps(dict(status=result['status'],out=str(args.out),missing=[r['component'] for r in result['missing']],NN=0,IQ_reads=0,checkpoint_reads=0)),flush=True)


if __name__=='__main__':main()
