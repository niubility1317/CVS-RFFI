"""P76 saved-output factorial analysis; no IQ, checkpoint, inference or rescoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
import argparse
import importlib.util
import json
import time
import traceback
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np

RUN='20261019-phase1-FcrossRX-propContrast-e30-s2026092701-m4-r01'
REF='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
PARENT_RUN='20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01'
SEED,STEPS,UPDATES,PER_VIEW=2026092701,222,2220,64
CELLS=('C00','C10','C01','C11');ARMS=('NOAS30','AS30',*CELLS);NEW_ARMS=CELLS
PRIMARY=('C11','C00')
PAIRS=(PRIMARY,('C11','C10'),('C11','C01'),('C10','C00'),('C01','C00'),('AS30','NOAS30'),
       *((a,'NOAS30') for a in CELLS),*((a,'AS30') for a in CELLS))
PROPAGATION=('geometry','probs','state','z','ray_hz','ray_coef','f_doppler')
PRIVATE=('noise','phase','sync','tracking','equalizer')
LEVELS=('time_down','t2.pool')


def load_helpers():
    path=Path(__file__).with_name('cvs_noise_resample75_analysis_20261019.py')
    spec=importlib.util.spec_from_file_location('_p76_existing75',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


q=load_helpers();h=q.h;c=q.c;_write=q._write


def row_specs(project):
    runs=Path(project)/'runs';rows=[]
    for key in ARMS:
        reused=key not in CELLS;arm=('B00' if key=='NOAS30' else 'B01') if reused else key
        run=REF if reused else RUN;prun=REF if reused else PARENT_RUN;parm=arm if reused else 'AS_CE'
        rows.append(dict(key=key,run=run,arm=arm,row_id=arm+'-s'+str(SEED),root=runs/run/(arm+'-s'+str(SEED)),
            reused=reused,parent=runs/prun/(parm+'-s'+str(SEED))/'stage20',parent_run=prun,
            parent_row_id=parm+'-s'+str(SEED),parent_As_KL_off=not reused))
    return rows


training_complete=h.training_complete


def source_criteria(values):
    n=values['C11'];clean=100*(n['fullV_clean']-values['NOAS30']['fullV_clean'])
    gain=100*(n['midlow']-max(values[a]['midlow'] for a in ('C00','C10','C01','AS30')))
    interaction=100*(values['C11']['midlow']-values['C10']['midlow']-values['C01']['midlow']+values['C00']['midlow'])
    checks=dict(clean_within_NoAs_0p1pp=clean>=-.1-1e-12,midlow_vs_max_all_controls_at_least_0p5pp=gain>=.5-1e-12,
        midlow_interaction_at_least_0p5pp=interaction>=.5-1e-12)
    return dict(checks=checks,source_qualified=all(checks.values()),clean_delta_pp=clean,midlow_gain_pp=gain,interaction_pp=interaction)


def read_source_decision(project):
    value=c.read(Path(project)/'runs'/RUN/'source_matrix_decision.json')
    c.require(value.get('status') in c.DECISION_COMPLETE|c.DECISION_PARTIAL and value.get('formal_scores_read') is False and
        value.get('target_feedback') is False and value.get('NN')==0 and value.get('model_seed_count')==1,
        'Already fixed source verdict must precede formal readout')
    if value['status'] in c.DECISION_COMPLETE:
        checks=value.get('checks');c.require(isinstance(checks,dict) and len(checks)==3 and
            all(type(v)is bool for v in checks.values()) and type(value.get('source_qualified'))is bool and
            value['source_qualified']==all(checks.values()),'Exactly three stored source interaction checks required')
    return value


def factorial(rows,key,scale=100.):
    values=[rows[a].get(key) for a in ('C11','C10','C01','C00')]
    return None if any(v is None for v in values) else float(scale*(values[0]-values[1]-values[2]+values[3]))


def interaction(rows):
    return dict(accuracy_interaction_pp=factorial(rows,'accuracy'),macro_f1_interaction_pp=factorial(rows,'macro_f1'),
        worst_RX_interaction_pp=factorial(rows,'worst_RX'),margin_over30_interaction=factorial(rows,'mean_margin_over30',1.),
        retention_interaction_pp=factorial(rows,'satellite_retention'))


def enrich_formal(obj):
    rows=obj['rows'];obj.pop('all4_rows_all7_verified_before_truth',None);obj['all6_rows_all7_verified_before_truth']=True
    obj['primary']=dict(new='C11',control='C00',same_parent=True,parent_run=PARENT_RUN,parent_row='AS_CE-s'+str(SEED),
        parent_epoch=20,parent_As_KL_off=True,effects=q.effects(rows,'C11','C00'))
    obj['comparisons']={a+'__'+b:q.effects(rows,a,b) for a,b in PAIRS}
    obj['interaction']={v:interaction({a:rows[a][v] for a in CELLS}) for v in c.VIEWS}
    obj['interaction']['urban_mean']={k:(None if any(obj['interaction'][v][k] is None for v in c.VIEWS[-3:]) else float(np.mean([obj['interaction'][v][k] for v in c.VIEWS[-3:]])))
        for k in obj['interaction'][c.VIEWS[-1]]}
    obj['suburban_gap_from_own_clean_pp']={a:{v:100*(rows[a]['clean']['accuracy']-rows[a][v]['accuracy']) for v in c.VIEWS[1:4]} for a in ARMS}
    obj['urban_accuracy_mean']={a:float(np.mean([rows[a][v]['accuracy'] for v in c.VIEWS[-3:]])) for a in ARMS}
    obj['comparison_context']=[dict(new=a,control=b,same_parent=a in CELLS and b in CELLS,
        first20_As_KL_history_confounded=a in CELLS and b=='AS30',
        causal_attribution='Shared propagation x As contrast factorial' if a in CELLS and b in CELLS else 'Whole-recipe descriptive reference') for a,b in PAIRS]
    for a in CELLS:obj['stage'][a].update(NoAs30_is_whole_recipe_reference=True,same_parent_NoAs_control_available=False,reproducible_dual_gain=False)
    obj.update(Ag_claim=False,extra_deployment_module=False,formal_channel_changed=False,model_seed_count=1,reproducible_dual_gain=False)


def write(path,obj):
    if Path(path).name=='formal_summary.json':enrich_formal(obj)
    _write(path,obj)


c.RUN,c.NEW_ARMS,c.PAIRS=RUN,NEW_ARMS,PAIRS
c.training_complete,c.row_specs,c.write=training_complete,row_specs,write


def parse_tail(folder,is_as,arm=None):
    if arm not in CELLS:return h.parse_tail(folder,is_as)
    folder=Path(folder);steps=list(c.lines(folder/'prop_contrast_steps.jsonl.gz'));by_step={}
    for step,r in enumerate(steps,1):
        c.require(r.get('step')==step and r.get('cumulative_step')==step+4440 and r.get('method')==arm,'Complete ordered As objective steps required')
        raw=c.finite(r['As_loss_raw'],'As loss');weighted=c.finite(r['As_loss_weighted'],'weighted As loss');on=arm[2]=='1'
        c.require(r.get('ordinary_SupCon_coefficient')==.1 and r.get('As_SupCon_coefficient')==(.05 if on else 0.) and
            r.get('As_loss_evaluated') is on and abs(weighted-(.05*raw if on else 0.))<1e-6 and
            (on or raw==0.),'One extra .05 As term; unchanged ordinary .1 required')
        by_step[step]=r
    c.require(len(steps)==UPDATES,'All2220 actual As objective records required')
    original_lines=c.lines;max_error=0.
    def adapted(path):
        nonlocal max_error
        for original in original_lines(path):
            if Path(path)==folder/'reference_steps.jsonl.gz':
                extra=by_step[original['step']]['As_loss_weighted']
                c.require(abs(c.finite(original.get('As_loss_weighted'),'native As extra')-extra)<1e-12,'Native and objective extra loss must match')
                value=dict(original,total_L=original['total_L']-extra)
                max_error=max(max_error,abs((value['total_L']+extra)-original['total_L']))
                yield value
            else:yield original
    try:
        c.lines=adapted;tail,ordinary=h.parse_tail(folder,is_as)
    finally:c.lines=original_lines
    for epoch in tail['curve']:
        rr=steps[(epoch['local_epoch']-1)*STEPS:epoch['local_epoch']*STEPS]
        epoch['base_total_L']=epoch['total_L']
        for k in ('As_loss_raw','As_loss_weighted'):epoch[k]=float(np.mean([r[k] for r in rr]))
        epoch['total_L']+=epoch['As_loss_weighted']
        epoch['As_aux_evaluated']=arm[2]=='1'
    tail.update(actual_total_L_includes_As_term=True,base_decomposition_verified_separately=True,
        maximum_actual_L_reconstruction_error=max_error,ordinary_SupCon_coefficient=.1,
        As_SupCon_coefficient=.05 if arm[2]=='1' else 0.)
    return tail,ordinary


def eligibility(labels,groups):
    y=np.asarray(labels);g=np.asarray(groups)
    c.require(y.ndim==1 and y.dtype.kind in 'iu' and len(y)>0 and np.all((y>=0)&(y<6)) and g.shape==(len(y),3),
        'Actual legal L labels and RX/day/eq groups required')
    c.require(g.dtype.kind in 'iu' and set(g[:,0]).issubset(c.RX) and np.all((g[:,1]>=1)&(g[:,1]<=3)) and np.all(g[:,2]==1),'Legal source environment groups required')
    same=y[:,None]==y[None,:];group=(g[:,None,:]==g[None,:,:]).all(-1)
    pos=same&(g[:,None,0]!=g[None,:,0]);neg=(~same)&group;valid=pos.any(1)&neg.any(1)
    return dict(valid_anchors=int(valid.sum()),zero_anchors=int((~valid).sum()),positive_pairs=int(pos.sum()),negative_pairs=int(neg.sum()),
        valid_positive_pairs=int(pos[valid].sum()),valid_negative_pairs=int(neg[valid].sum()))


def prop_recount(records,physics,arm,expected_steps=UPDATES,packets=PER_VIEW):
    c.require(arm in CELLS,'Only four fixed cells');share=arm[1]=='1';on=arm[2]=='1'
    counts=Counter();strata=Counter();coverage=Counter();group_coverage=Counter();sums=Counter();potential=Counter();rows=[];sessions={}
    environment_total=personal_total=native_total=0
    for step,(r,p) in enumerate(zip_longest(records,physics),1):
        c.require(r is not None and p is not None and r.get('step')==p.get('step')==step and
            r.get('cumulative_step')==p.get('cumulative_step')==step+4440 and r.get('method')==p.get('arm')==arm,
            'Complete ordered factorial objective/physics stream required')
        c.require(r.get('share_propagation') is share and r.get('As_contrast') is on and p.get('share_propagation') is share and
            p.get('As_contrast') is on and r.get('generated_As0')==packets and r.get('generated_As1')==0 and
            r.get('As_generated_packets')==r.get('actual_residual_snapshots')==packets and r.get('As_forward_packets')==packets*2 and
            r.get('logical_second_slots')==packets and r.get('As_same_IQ') is True and r.get('L_NN')==768 and
            r.get('cross_channel_KL') is False and r.get('two_separate_CE_mean') is True and
            r.get('all_labeled_CE_retained') is True and r.get('bypassed_elements')==0 and r.get('As_IQ_pair_MSE')==0.,
            'One actual native As64 and exact repeated CE view required')
        first,second,labels=p.get('view0'),p.get('view1'),p.get('labels_L_As')
        c.require(isinstance(first,list) and isinstance(second,list) and len(first)==len(second)==packets and isinstance(labels,list) and
            len(labels)==packets and all(type(v)is int and v in range(6) for v in labels),'Never infer TX from sig_i; true L labels required')
        groups={};grouped_labels=defaultdict(Counter);unique_H={};group_vectors=[]
        namespace=f'FcrossRX/train/e{(step-1)//STEPS+21}/step{step+4440}/L_S'
        for a,b,y in zip(first,second,labels):
            pid=a.get('sample_id')
            c.require(isinstance(pid,list) and len(pid)==5 and pid[0]==1 and pid[1] in c.RX and pid[2] in c.DAYS and
                a.get('namespace')==namespace and a.get('scene') in c.SCENES and a.get('state_end') in c.STATES and
                a.get('receiver_lock_status') in c.LOCKS and a.get('channel_equalization_applied') is False,
                'Original legal source/native residual/EQoff metadata required')
            group=(int(pid[1]),int(pid[2]),int(pid[0]));session=f'rx{group[0]}:day{group[1]}:eq{group[2]}'
            c.require(a.get('group')==a.get('environment_group')==list(group) and a.get('session_id')==a.get('group_id')==session and
                a.get('TX_used_in_group') is False and a.get('sig_i_is_TX') is False,'RX/day/eq groups must not use TX or sig_i as labels')
            scene=a['scene'];scene_seed=q.stable_seed(20261016421,namespace,'scene',group)
            personal_seed=q.stable_seed(20261016421,namespace,scene,pid);env_seed=q.stable_seed(20261016421,namespace,scene,'environment',group)
            c.require(a.get('scene_seed')==scene_seed and a.get('personal_seed')==a.get('packet_seed')==a.get('seed')==personal_seed and
                a.get('environment_seed')==env_seed and a.get('share_propagation') is share and
                a.get('propagation_source')==('environment' if share else 'personal') and a.get('propagation_fields')==list(PROPAGATION),
                'No arm/TX-dependent seed; correct propagation-only scope required')
            c.require(a.get('private_streams_preserved') is True and a.get('propagation_arrays_not_aliased') is True and
                a.get('same_raw_input') is True and a.get('native_noise_chain_retained') is True and a.get('native_agc_recomputed') is True and
                a.get('environment_initialized_not_processed') is True and
                a.get('private_stream_seeds')=={name:q.stable_seed(personal_seed,name) for name in PRIVATE},'Private native streams must remain personal')
            c.require(not share or a.get('propagation_matches_environment') is True,'Shared H must match its environment')
            c.require(b.get('view_kind')=='same_IQ_reuse' and b.get('channel_generated') is False and b.get('agc_recomputed') is False and
                all(key in b and b[key]==value for key,value in a.items()),'Second slot must be exact metadata copy; no second native call')
            if session in sessions:c.require(sessions[session]==a['receiver'],'Stable physical RX hardware required across steps')
            else:sessions[session]=a['receiver']
            if group not in groups:groups[group]=dict(scene=scene,env_seed=env_seed,index=len(groups),n=0)
            info=groups[group];info['n']+=1
            c.require((scene,env_seed,a.get('environment_index'))==(info['scene'],info['env_seed'],info['index']),
                'Group scene common in every cell and one environment construction per group required')
            if share:
                signature={k:a[k] for k in ('geometry','state_end','orbital_doppler_hz','residual_kernel_energy','propagation_kernel_length','residual_kernel_length','boundary')}
                if group in unique_H:c.require(unique_H[group]==signature,'Shared propagation observable metadata differ within group')
                else:unique_H[group]=signature
            grouped_labels[group][y]+=1;group_vectors.append(group)
            group_coverage[(y,*group,share,scene,a['state_end'],a['receiver_lock_status'])]+=1
            for view,obs in enumerate((a,b)):
                counts[view]+=1;strata[(view,group[0],group[1],scene,obs['state_end'],obs['receiver_lock_status'])]+=1
                coverage[(view,y,group[0],scene,obs['state_end'],obs['receiver_lock_status'])]+=1
        gcount=len(groups)
        for obs in first:c.require(obs.get('environment_initializations_this_step')==gcount and
            obs.get('personal_initializations_this_step')==obs.get('native_waveforms_this_step')==packets and
            obs.get('environment_processes_this_step')==0,'Count repeated step construction metadata once, not64 times')
        expected_groups={g:dict(v) for g,v in grouped_labels.items()}
        for logged in p.get('group_stats',[]):
            key=(logged['RX'],logged['day'],logged['eq'])
            c.require(key in expected_groups and {int(k):v for k,v in logged['TX_counts'].items()}==expected_groups[key] and
                logged['count']==sum(expected_groups[key].values()) and logged['scenes']==[groups[key]['scene']],
                'Logged group TX counts must use visible L labels')
            del expected_groups[key]
        c.require(not expected_groups and r.get('group_count')==gcount,'Every physical group must be logged')
        eligible=eligibility(labels,group_vectors);potential.update(eligible)
        details=r.get('As_details');raw=c.finite(r['As_loss_raw'],'As loss');weighted=c.finite(r['As_loss_weighted'],'As weighted loss')
        c.require(r.get('ordinary_SupCon_coefficient')==.1 and r.get('As_SupCon_coefficient')==(.05 if on else 0.) and
            r.get('As_loss_evaluated') is on and abs(weighted-(.05*raw if on else 0.))<1e-6,'Exactly one extra .05 term required')
        if on:
            c.require(isinstance(details,list) and len(details)==4 and {(v['phase'],v['level']) for v in details}=={(p,l) for p in (0,1) for l in LEVELS},'Two depths and both existing mixed passes required')
            for v in details:
                c.require(v.get('total_anchors')==packets and all(v.get(k)==value for k,value in eligible.items()) and v.get('temperature')==.2 and
                    v.get('same_RX_positive_pairs')==v.get('cross_group_negative_pairs')==0,'Actual As positive/negative support differs')
            c.require(abs(raw-np.mean([v['loss'] for v in details]))<1e-6,'As raw loss must average two depths x two passes')
            for key in ('valid_anchors','zero_anchors','positive_pairs','negative_pairs'):
                c.require(r.get('As_'+key)==eligible[key],'Saved actual As pair scalar differs')
        else:c.require(details==[] and raw==weighted==0.,'OFF stats are unevaluated, not measured pair absence')
        environment_total+=gcount;personal_total+=packets;native_total+=packets
        for key,value in r.items():
            if type(value)in(int,float):sums[key]+=c.finite(value,key)
        rows.append(dict(arm=arm,step=step,cumulative_step=step+4440,epoch=(step-1)//STEPS+21,share_propagation=share,As_contrast=on,
            group_count=gcount,environment_initializations=gcount,personal_initializations=packets,native_waveforms=packets,
            As_loss_raw=raw,As_loss_weighted=weighted,As_stats_evaluated=on,**{'potential_'+k:v for k,v in eligible.items()},
            multi_TX_groups=sum(len(v)>1 for v in grouped_labels.values())))
    total=expected_steps*packets
    c.require(len(rows)==expected_steps and counts=={0:total,1:total},'Complete native/copy support required')
    physical=[dict(view=v,RX=rx,day=day,scene=sc,state=st,lock=lk,count=n) for (v,rx,day,sc,st,lk),n in sorted(strata.items())]
    label_table=[dict(view=v,TX=tx,RX=rx,scene=sc,state=st,lock=lk,count=n) for (v,tx,rx,sc,st,lk),n in sorted(coverage.items())]
    group_table=[dict(TX=tx,RX=rx,day=day,eq=eq,share_propagation=sh,scene=sc,state=st,lock=lk,count=n)
        for (tx,rx,day,eq,sh,sc,st,lk),n in sorted(group_coverage.items())]
    return dict(status='SAVED_FACTORIAL_PHYSICS_RECOUNTED',arm=arm,steps=len(rows),counts=dict(counts),native_waveforms=native_total,
        environment_initializations=environment_total,personal_initializations=personal_total,environment_processes=0,
        logical_second_slots=total,As_forward_slots=2*total,session_count=len(sessions),share_propagation=share,As_contrast=on,
        scalar_means={k:v/len(rows) for k,v in sums.items()},potential_pair_counts=dict(potential),potential_pairs_offline_from_true_L=True,
        OFF_zero_stats_are_not_pair_absence=True,physical_strata=physical,label_coverage=label_table,group_TX_RX_day_coverage=group_table,
        TX_from_labels_L_As_only=True,total_SNR_known=False,physical_causality_identified=False,NN=0,IQ_reads=0,checkpoint_reads=0),rows


def validate_summary(saved,recount,result):
    n=recount['steps'];total=recount['native_waveforms'];arm=recount['arm'];on=arm[2]=='1'
    expected=dict(arm=arm,steps=n,generated_As0=total,generated_As1=0,native_waveforms=total,As_forward_NN=total*2,
        L_NN=n*768,logical_second_slots=total,share_propagation=arm[1]=='1',As_contrast=on,ordinary_SupCon_coefficient=.1,
        As_SupCon_coefficient=.05 if on else 0.,total_L_CE_scale=1.25,As_CE_coefficient=.25,per_pass_As_CE_coefficient=.125,
        As_KL=False,original_dropout=True,original_ordinary_raw_objectives=True,new_deployment_parameters=0,extra_NN=0,extra_U_NN=0,
        U_truth_read=False,target_access=False,physics_records=n,session_count=recount['session_count'])
    c.require(all(saved.get(k)==v for k,v in expected.items()),'Actual summary counts or fixed objectives differ')
    for key,value in recount['scalar_means'].items():c.require(abs(c.finite(saved.get('means',{}).get(key+'_mean'),key)-value)<3e-5,'Whole-tail saved scalar mean differs '+key)
    actual=Counter()
    for row in result['TX_RX_scene_state_lock']:
        key=tuple(row[k] for k in ('view','TX','RX','scene','state','lock'));c.require(key not in actual,'Duplicate labeled support cell')
        actual[key]=row['count']
    expected_labels=Counter({tuple(row[k] for k in ('view','TX','RX','scene','state','lock')):row['count'] for row in recount['label_coverage']})
    c.require(actual==expected_labels and result.get('L_As_total')==total and result.get('L_As_forward_NN')==2*total,
        'Producer labeled support must equal independently recounted true L labels')


def training_analysis(project,specs,out):
    tails={};curves=[];ordinary={};by_key={s['key']:s for s in specs}
    for spec in specs:
        key=spec['key'];tail,ordinary[key]=parse_tail(spec['root']/'source',key!='NOAS30',key)
        tail.update(parent_run=spec['parent_run'],parent_row_id=spec['parent_row_id'],parent_As_KL_off=spec['parent_As_KL_off'])
        if key in CELLS:
            result=c.read(spec['root']/'source/result.json')
            proof,steps=prop_recount(c.lines(spec['root']/'source/prop_contrast_steps.jsonl.gz'),c.lines(spec['root']/'source/prop_contrast_physics.jsonl.gz'),key)
            validate_summary(result['prop_contrast_summary'],proof,result);tail['propagation_contrast_physics']=proof
            c.csv_write(out/(key+'_prop_steps.csv'),steps);c.csv_write(out/(key+'_group_TX_RX_day_state.csv'),proof['group_TX_RX_day_coverage'])
            c.csv_write(out/(key+'_labeled_coverage.csv'),proof['label_coverage']);c.csv_write(out/(key+'_physical_strata.csv'),proof['physical_strata'])
        for row in c.epoch_curve(spec['parent'],0)+tail['curve']:curves.append(dict(row_id=key,model_seed=SEED,reused=spec['reused'],parent_reused=True,**row))
        tails[key]=tail
    sequences={}
    for arm in CELLS[1:]:
        sequences[arm]=h.same_sequence(by_key[arm]['root']/'source/training_sequence.jsonl.gz',by_key['C00']['root']/'source/training_sequence.jsonl.gz')
        c.require(ordinary[arm]==ordinary['C00'],'All factorial cells must retain identical ordinary augmentation inputs')
    c.csv_write(out/'training_curves.csv',curves)
    result=dict(status='FULL_AVAILABLE_TRAINING_LOGS_VERIFIED',rows=tails,new_model_epochs=40,new_updates=8880,
        matched_L_U_sequences=sequences,ordinary_augmentation_stream_identical=True,ordinary_SupCon_coefficient=.1,
        As_SupCon_coefficient_ON=.05,parent_epoch20_reused=True,parent_history_difference=True,extra_NN=0,U_truth_read=False,
        native_U_temporal_mode='batch_neighbor',coverage_is_not_identity_performance=True)
    c.write(out/'training_summary.json',result);return result,curves


def interaction_rows(records,context):
    grouped=defaultdict(dict)
    for row in records:
        if row['row_id'] in CELLS:grouped[tuple(row[k] for k in context)][row['row_id']]=row
    output=[]
    for key,rows in grouped.items():
        c.require(set(rows)==set(CELLS),'Four matched cells required for every interaction stratum')
        output.append(dict(zip(context,key),count=rows['C00']['count'],**interaction(rows)))
    return output


def pooled_source(records):
    selected=[r for r in records if r['scene'] in c.SCENES[1:] and r['draw']==-1]
    pooled={}
    for arm in ARMS:
        rr=[r for r in selected if r['row_id']==arm and r['dimension']=='overall'];c.require(len(rr)==2,'Both mid/low source rows required')
        cm=sum((np.asarray(r['confusion']) for r in rr),np.zeros((6,6),int));count=int(cm.sum());den=cm.sum(0)+cm.sum(1)
        f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0);nc=sum(r['fixed_NOAS30_clean_correct_count'] for r in rr)
        rx_values=[]
        for rx in c.RX:
            rows=[r for r in selected if r['row_id']==arm and r['dimension']=='RX' and r['stratum']==str(rx)]
            rx_values.append(sum(r['correct'] for r in rows)/sum(r['count'] for r in rows))
        pooled[arm]=dict(count=count,accuracy=float(np.trace(cm)/count),macro_f1=float(f.mean()),worst_RX=min(rx_values),
            mean_margin_over30=sum(r['mean_margin_over30']*r['count'] for r in rr)/count,
            satellite_retention=sum(r['fixed_NOAS30_clean_correct_retained'] for r in rr)/nc if nc else None)
    return dict(rows=pooled,interaction=interaction({a:pooled[a] for a in CELLS}),
        unique_source_physical=1800,repeated_channel_draws=2,scene_count=2,model_seed_count=1,source_verdict_recomputed=False)


def smoke_constructors(records,packets=64):
    environment=personal=native=0
    for record in records:
        first=record.get('view0',[]);second=record.get('view1',[])
        c.require(len(first)==len(second)==packets,'Complete saved smoke physical metadata required')
        groups={tuple(row['group']) for row in first};g=len(groups)
        for row in first:c.require(row.get('environment_initializations_this_step')==g and
            row.get('personal_initializations_this_step')==row.get('native_waveforms_this_step')==packets and
            row.get('environment_processes_this_step')==0,'Smoke construction metadata counted once per step')
        c.require(all(row.get('channel_generated') is False for row in second),'Smoke copy slot must not add native calls')
        environment+=g;personal+=packets;native+=packets
    return dict(environment_initializations=environment,personal_initializations=personal,native_waveforms=native,environment_processes=0)


def costs(project,specs,local_nn=0,smoke_nn=3076,training=None):
    rows={};new={};parents={}
    for spec in specs:
        tail=c.native_cost(spec['root']/'source');qq=c.read(spec['root']/'source_qualification/complete.json');pp=c.read(spec['root']/'prediction/complete.json')
        c.require(qq.get('NN')==12600 and pp.get('actual_NN')==1176002,'Fixed source/formal actual NN required')
        parent=c.native_cost(spec['parent']);parents[str(spec['parent'])]=parent
        row=dict(parent=parent,tail=tail,source_NN=qq['NN'],formal_NN=pp['actual_NN'],source_wall_seconds=qq.get('wall_seconds'),
            formal_wall_seconds=pp.get('wall_seconds'),parent_run=spec['parent_run'],parent_row_id=spec['parent_row_id'])
        if spec['key'] in CELLS:
            result=c.read(spec['root']/'source/result.json');s=result['prop_contrast_summary']
            c.require(s['native_waveforms']==142080 and s['As_forward_NN']==284160 and tail['native_NN']==3162962,'Native calls and NN must remain separate')
            row.update(native_waveforms=142080,logical_As_forward_slots=284160)
            if training:
                proof=training['rows'][spec['key']]['propagation_contrast_physics']
                row.update(environment_initializations=proof['environment_initializations'],personal_initializations=proof['personal_initializations'],environment_processes=0)
            new[spec['key']]=row
        rows[spec['key']]=row
    smoke=c.read(Path(project)/'runs'/RUN/'real_checkpoint_smoke.json')
    c.require(smoke.get('status')=='PASS' and smoke.get('NN')==smoke_nn==3076 and smoke.get('physical_snapshots')==256 and local_nn==0,
        'Shared3076 NN/256 native outputs/local0 smoke required')
    smoke_physics=c.read(Path(project)/'runs'/RUN/'synthetic_physics.json')
    c.require(smoke_physics.get('source_IQ')==0 and smoke_physics.get('target_query') is False and len(smoke_physics.get('records',[]))==4,'Shared smoke metadata must remain synthetic/query-free')
    smoke_cost=smoke_constructors(smoke_physics['records'])
    c.require(smoke_cost['native_waveforms']==256,'Actual smoke native metadata count differs')
    total=sum(r['tail']['training_total_NN']+r['source_NN']+r['formal_NN'] for r in new.values())+smoke_nn
    c.require(len(new)==4 and total==17409332,'Four exact ten-epoch tails plus fixed evaluation/smoke required')
    return dict(rows=rows,new_native_NN=12651848,new_source_NN=50400,new_formal_NN=4704008,shared_remote_smoke_NN=3076,
        new_total_remote_NN=total,local_verification_NN=0,new_model_epochs=40,new_updates=8880,
        student_parent_epochs=20,student_parent_updates=4440,student_cumulative_epochs=30,student_cumulative_updates=6660,
        parent_training_reused_not_repeated=True,parent_As_KL_off=True,parent_run=PARENT_RUN,unique_reused_parents=parents,
        historical_reference_and_shared_parent_model_epochs=80,unique_total_model_epochs_including_reused=120,
        unique_total_updates_including_reused=26640,new_training_wall_seconds_sum=sum(r['tail']['wall_seconds'] for r in new.values()),
        new_training_native_waveforms=568320,new_smoke_native_waveforms=256,shared_smoke_constructors=smoke_cost,
        training_constructors_recounted=training is not None,
        new_training_environment_initializations=sum(r['environment_initializations'] for r in new.values()) if training else None,
        new_training_personal_initializations=sum(r['personal_initializations'] for r in new.values()) if training else None,per_row_native_waveforms=142080,
        As_forward_slots_per_row=284160,constructors_are_not_waveforms=True,extra_student_NN=0,extra_teacher_NN=0,
        reference_reprediction=0,reference_rescoring=0,analysis_NN=0,NN_matching_does_not_imply_equal_wall=True,budget_exhaustion_is_not_convergence=True)


def plots(out,curves,source_rows,formal,training,source_interactions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(ARMS,('#64748b','#d97706','#8b5cf6','#16a34a','#0891b2','#2563eb')));files=[]
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):
            path=out/(name+'.'+ext);fig.savefig(path,dpi=180);files.append(str(path))
        plt.close(fig)
    if curves:
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        for arm in ARMS:
            rr=[r for r in curves if r['row_id']==arm]
            for ax,key,title in zip(axes,('source_val_accuracy_pct','native_loss','grad_backbone'),('Full source V','Native training loss','Backbone gradient norm')):
                ax.plot([r['epoch'] for r in rr],[r[key] for r in rr],label=arm,color=colors[arm]);ax.set(xlabel='Cumulative epoch',title=title);ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);save(fig,'training_curves')
    if source_rows:
        fig,axes=plt.subplots(2,3,figsize=(18,9))
        for si,scene in enumerate(c.SCENES[1:]):
            for di,dim in enumerate(('state','lock','signal_to_added_noise_db')):
                ax=axes[si,di];rr=[r for r in source_rows if r['scene']==scene and r['draw']==-1 and r['dimension']==dim]
                labels=list(dict.fromkeys(r['stratum'] for r in rr));xx=np.arange(len(labels))
                for i,arm in enumerate(ARMS):
                    selected=[next(r for r in rr if r['row_id']==arm and r['stratum']==label) for label in labels]
                    ax.bar(xx+(i-2.5)*.12,[100*r['accuracy'] if r['accuracy'] is not None else np.nan for r in selected],.12,label=arm,color=colors[arm])
                ax.set(xticks=xx,xticklabels=labels,title=scene+' / '+dim,ylabel='Source accuracy (%)');ax.tick_params(axis='x',labelrotation=25);ax.grid(axis='y',alpha=.2)
        axes[0,0].legend(fontsize=7);save(fig,'source_quality_state_lock_noise')
    fig,axes=plt.subplots(1,3,figsize=(19,5));xx=np.arange(7)
    for arm in ARMS:axes[0].plot(xx,[100*formal['rows'][arm][v]['accuracy'] for v in c.VIEWS],marker='.',label=arm,color=colors[arm])
    for old in ('C00','C10','C01'):
        axes[1].plot(xx,[100*(formal['rows']['C11'][v]['accuracy']-formal['rows'][old][v]['accuracy']) for v in c.VIEWS],marker='.',label='C11 - '+old)
    axes[2].plot(np.arange(8),[formal['interaction'][v]['accuracy_interaction_pp'] for v in (*c.VIEWS,'urban_mean')],marker='o')
    labels=['clean','high sub','mid sub','low sub','high urb','mid urb','low urb']
    for i,ax in enumerate(axes):
        ax.set(xticks=np.arange(8) if i==2 else xx,xticklabels=labels+['urban mean'] if i==2 else labels,
            title=('Frozen all7 default head','C11 paired changes','Four-cell interaction')[i],ylabel='Accuracy (%)' if i==0 else 'Absolute percentage points');ax.tick_params(axis='x',labelrotation=30);ax.grid(alpha=.2)
        if i<2:ax.legend(fontsize=7)
        if i>0:ax.axhline(0,color='grey',lw=.7)
    save(fig,'formal_paired')
    if training:
        fig,axes=plt.subplots(2,2,figsize=(16,10));xx=np.arange(4)
        native=[training['rows'][a]['propagation_contrast_physics']['native_waveforms'] for a in CELLS]
        env=[training['rows'][a]['propagation_contrast_physics']['environment_initializations'] for a in CELLS]
        axes[0,0].bar(xx-.17,native,.34,label='Native waveforms');axes[0,0].bar(xx+.17,env,.34,label='Unused environment constructions')
        axes[0,0].set(xticks=xx,xticklabels=CELLS,title='Actual simulation and constructor costs');axes[0,0].legend(fontsize=8)
        for arm in ('C01','C11'):
            rr=[r for r in curves if r['row_id']==arm and r['epoch']>20]
            axes[0,1].plot([r['epoch'] for r in rr],[r['As_loss_raw'] for r in rr],label=arm,color=colors[arm])
        axes[0,1].set(xlabel='Cumulative epoch',title='Extra As loss (actual coefficient .05)');axes[0,1].legend()
        hist=Counter()
        for r in training['rows']['C11']['propagation_contrast_physics']['group_TX_RX_day_coverage']:hist[(r['TX'],r['RX'],r['day'])]+=r['count']
        columns=[(rx,day) for rx in c.RX for day in c.DAYS];matrix=np.asarray([[hist[(tx,rx,day)] for rx,day in columns] for tx in range(6)])
        im=axes[1,0].imshow(matrix,aspect='auto');axes[1,0].set(xticks=np.arange(len(columns)),xticklabels=[str(rx)+'/'+str(day) for rx,day in columns],yticks=range(6),ylabel='Visible L TX',title='C11 native coverage: RX/day groups');axes[1,0].tick_params(axis='x',labelrotation=45);fig.colorbar(im,ax=axes[1,0])
        for scene in c.SCENES[1:]:
            rr=[r for r in source_interactions if r.get('scene')==scene and r.get('draw')==-1 and r.get('dimension')=='state']
            axes[1,1].plot(range(len(rr)),[r['accuracy_interaction_pp'] for r in rr],marker='o',label=scene)
            axes[1,1].set(xticks=range(len(rr)),xticklabels=[r['stratum'] for r in rr])
        axes[1,1].axhline(0,color='grey',lw=.7);axes[1,1].set(ylabel='Absolute percentage points',title='Source state interaction; one model seed');axes[1,1].legend(fontsize=8)
        save(fig,'interaction_and_coverage')
    return files


def report(out,result):
    formal=result['formal'];cost=result['cost'];rows=formal['rows'];stage=formal['stage']['C11']
    text=['# P76：不同 TX 包共享传播与环境内判别的四格实验','',
        '四条新尾训均从同一个SAT18 AS_CE20 student/ownEMA分叉，再训练10轮。所有格按RX/day/eq共同抽场景；共享H因子只移植七个传播字段，噪声/相位/sync/tracking/EQ随机流仍逐包独立，RX硬件按session稳定。每步生成一次As64，两mixed前向复用相同IQ与正常dropout；6L/768和原ordinary/raw目标、ordinary SupCon .1保留。ON格只增加As64两层两pass平均损失×.05，无新增NN/head。','',
        '额外As正对为同TX不同RX（anyday），负对为不同TX且同RX/day/eq；同TX同RX、不同TX不同环境均不进入额外分母。无合法正负对锚点仅归零额外损失，全部有标签As CE继续。OFF日志的锚点0表示未评估，离线从labels_L_As另算potential支持。sig_i从不被当作TX。','',
        '主比较C11−C00；C11−C10/C01和C11−C10−C01+C00分别判断单机制增量和交互。P43 NoAs30/As30整配方参考直接复用，其父训练历史不同，不把所有差异归因于本轮机制。','',
        '| 环境 | NoAs % | As % | C00 % | C10 % | C01 % | C11 % | C11−C00 pp | 交互 pp |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v in c.VIEWS:text.append('| '+v+' | '+' | '.join(f'{100*rows[a][v]["accuracy"]:.6f}' for a in ARMS)+f' | {formal["comparisons"]["C11__C00"][v]["accuracy_delta_pp"]:+.6f} | {formal["interaction"][v]["accuracy_interaction_pp"]:+.6f} |')
    effect=formal['comparisons']['C11__C00']
    text+=['',f'C11相对C00：low urban {effect[c.VIEWS[-1]]["accuracy_delta_pp"]:+.6f} pp、mid urban {effect[c.VIEWS[-2]]["accuracy_delta_pp"]:+.6f} pp、urban均值 {effect["urban_accuracy_mean_delta_pp"]:+.6f} pp、clean {effect["clean"]["accuracy_delta_pp"]:+.6f} pp。urban交互 {formal["interaction"]["urban_mean"]["accuracy_interaction_pp"]:+.6f} pp。',
        f'相对原P43 NoAs30：C11 clean {stage["clean_delta_vs_NoAs30_pp"]:+.6f} pp、urban均值 {stage["urban_mean_vs_NoAs30_pp"]:+.6f} pp；clean下降≤0.1绝对百分点：{stage["clean_within_0p1pp"]}。','',
        '只有一个模型seed；两channel draw不是两个seed，不能宣称可重复双提升。全部all7 acc/Macro-F1/最差RX、margin/30、suburban相对自身clean差距、固定P43 NoAs30 clean正确集合保持率、逐RX/day/TX及修复/新增错误见CSV。该集合不是共同AS_CE20父模型的clean正确集合。源域所有状态/lock/质量分层及pooled mid/low交互见source文件。新增噪声SNR不是总SNR；保存元数据和分层关联不能单独识别物理因果。','',
        f'新增4×E10＝40 model-epochs／8880更新，每row累计E30/6660更新。native NN={cost["new_native_NN"]}，source={cost["new_source_NN"]}，formal={cost["new_formal_NN"]}，shared smoke={cost["shared_remote_smoke_NN"]}，合计={cost["new_total_remote_NN"]}，本地NN=0。每row native process142080、As前向槽位284160；group环境与personal构造次数分别从物理日志独立重算，环境构造不是波形生成。父E20成本单列、去重复用；四新尾段训练过程秒数合计={cost["new_training_wall_seconds_sum"]:.3f}，同NN不等于同墙钟，预算结束不等于收敛。','',
        '构造成本：'+json.dumps({k:cost[k] for k in ('training_constructors_recounted','new_training_environment_initializations','new_training_personal_initializations','shared_smoke_constructors')},ensure_ascii=False),'',
        '源判定在formal读取前已经保存，本分析不重算晋级；FAIL仍完成all7，并关闭固定方案，不扫描强度/权重/seed或自动扩E50。已存判定：'+json.dumps(result['stored_source_decision'],ensure_ascii=False),'',
        '未完成／限制：'+json.dumps(result['missing'],ensure_ascii=False),'','实际图表：']
    text+=['- '+p for p in result['figures']]
    text+=['','本离线分析新增NN、IQ/checkpoint读取、训练更新、重推理和重评分均为0。']
    (out/'final_analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def run(project,out,truth=None,code_commit=None,local_nn=0,smoke_nn=3076):
    project=Path(project);decision=read_source_decision(project);out=Path(out)
    c.require(not out.exists(),'Immutable fresh analysis output required');out.mkdir(parents=False)
    started=time.perf_counter();c.write(out/'stored_source_decision.json',decision);specs=row_specs(project)
    missing=[];training=None;curves=[];source=None;source_rows=[];source_interactions=[];pooled=None
    if decision['status'] in c.DECISION_PARTIAL:missing.append(dict(component='source_decision',error='Stored source verdict PARTIAL; not recomputed'))
    for component in ('training','source'):
        try:
            if component=='training':training,curves=training_analysis(project,specs,out)
            else:
                source,source_rows=c.source_analysis(project,specs,out,decision)
                source_interactions=interaction_rows(source_rows,('scene','draw','dimension','stratum'));pooled=pooled_source(source_rows)
                c.csv_write(out/'source_strata_interactions.csv',source_interactions);c.write(out/'source_pooled_midlow.json',pooled)
        except Exception as error:
            failure=dict(component=component,error=repr(error),traceback=traceback.format_exc(),formal_still_required=True)
            missing.append(failure);c.write(out/(component+'_failure.json'),failure)
    truth=Path(truth) if truth else project/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
    formal=c.formal_analysis(project,specs,out,truth)
    formal_strata=interaction_rows(c.read(out/'formal_confusions.json'),('view','dimension','stratum'))
    c.csv_write(out/'formal_strata_interactions.csv',formal_strata)
    c.csv_write(out/'formal_overall_interactions.csv',(dict(view=v,**value) for v,value in formal['interaction'].items()))
    cost=costs(project,specs,local_nn,smoke_nn,training);c.write(out/'costs.json',cost)
    try:figures=plots(out,curves,source_rows,formal,training,source_interactions)
    except Exception as error:
        figures=[];failure=dict(component='plots',error=repr(error),traceback=traceback.format_exc());missing.append(failure);c.write(out/'plot_failure.json',failure)
    result=dict(status='ANALYZED' if not missing else 'FORMAL_ANALYZED_WITH_PARTIAL_COMPONENTS',run_id=RUN,code_commit=code_commit,
        stored_source_decision=decision,training=training,source=source,source_pooled_midlow=pooled,formal=formal,cost=cost,figures=figures,missing=missing,
        model_seed_count=1,crossseed_confirmed=False,reproducible_dual_gain=False,source_decision_recomputed=False,target_feedback=False,goal_complete=False,
        NN=0,IQ_reads=0,checkpoint_reads=0,training_updates=0,wall_seconds=time.perf_counter()-started)
    report(out,result);c.write(out/'summary.json',{k:v for k,v in result.items() if k!='training'});c.write(out/'complete.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--truth',type=Path);parser.add_argument('--code-commit');parser.add_argument('--local-nn',type=int,default=0)
    parser.add_argument('--smoke-nn',type=int,default=3076);parser.add_argument('--checks',action='store_true');args=parser.parse_args()
    if args.checks:
        path=Path(__file__).with_name('cvs_prop_contrast76_analysis_checks_20261019.py')
        spec=importlib.util.spec_from_file_location('_p76_analysis_checks',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.checks();return
    if args.project is None or args.out is None:parser.error('--project and --out required unless --checks')
    result=run(args.project,args.out,args.truth,args.code_commit,args.local_nn,args.smoke_nn)
    print(json.dumps(dict(status=result['status'],out=str(args.out),missing=[r['component'] for r in result['missing']],NN=0,IQ_reads=0,checkpoint_reads=0)),flush=True)


if __name__=='__main__':main()
