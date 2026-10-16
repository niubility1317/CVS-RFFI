"""Full independent source score/cost/sequence recount, no research model import."""
import csv,gzip,json,math,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from experiments.cvs_iq_timing20.harvest import score
OUT=root/'local_artifacts/F_shift_bypass20_20261016/final'
RUN='20261016-phase1-F-crossRX-shiftbypass-sourceRX1-e20-s2026092701-m2-r01'
SEED=2026092701
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
ROWS=tuple(arm+'-RX1-s'+str(SEED) for arm in ('B00','B10'))

def read(p):return json.loads(p.read_text(encoding='utf8'))
def records(p):
    opener=gzip.open if p.suffix=='.gz' else open
    with opener(p,'rt',encoding='utf8') as f:return [json.loads(s) for s in f if s.strip()]
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def finite_numbers(v):
    if isinstance(v,float):assert math.isfinite(v)
    elif isinstance(v,dict):
        for x in v.values():finite_numbers(x)
    elif isinstance(v,list):
        for x in v:finite_numbers(x)

def analyze():
    complete=read(OUT/'completion.json') if len(ROWS)==2 else None
    if complete is not None:assert complete['status']=='ANALYZED' and complete['rows']==2
    scores={};predictions={};execution={};curves=[];sequences=[];native_args=[];shift_draws=[]
    for row in ROWS:
        arm=row[:3];src=OUT/row/'source';summary=read(src/'satmix_result.json');init=read(src/'initialization.json');frozen=read(src/'source_frozen.json');done=read(src/'completion.json')
        assert init['scratch_only'] and not init['checkpoint_sources'] and not init['ancestors'] and not init['target_contact']
        assert done['epoch']==20 and done['optimizer_steps']==4440 and not done['target_access'] and done['head_fit_updates']==0 and frozen['epoch']==20 and frozen['selection']=='final_only'
        assert summary['actual_steps']==4440 and summary['L_forward_packets']==3409920 and summary['supcon_applied']==3330 and summary['supcon_layer_forward_terms']==13320
        assert summary['Ag'] is False and summary['As'] is False and summary['As_training_observations']==0 and summary['ordinary_shift_bypass']==(arm=='B10') and summary['shift_native_L_calls']==4440 and summary['shift_native_L_packets']==568320 and summary['shift_bypass_returned_packets']==(568320 if arm=='B10' else 0) and summary['shift_capture_extra_NN_packets']==0
        scores[arm]={};predictions[arm]={}
        for view in VIEWS:
            p=read(src/'source_readouts'/(view+'.json'));s=score(p['labels'],p['predictions'])
            assert s['count']==1260 and len({tuple(v) for v in p['physical_ids']})==1260 and {v[1] for v in p['physical_ids']}=={1}
            assert math.isclose(s['accuracy'],summary['views'][view]['accuracy'],abs_tol=1e-12) and math.isclose(s['macro_F1'],summary['views'][view]['macro_F1'],abs_tol=1e-12)
            if view!='clean':assert p['labels']==predictions[arm]['clean']['labels'] and p['physical_ids']==predictions[arm]['clean']['physical_ids']
            scores[arm][view]=s;predictions[arm][view]=p
        native=records(src/'metrics_epoch.jsonl');compact=records(src/'epoch_metrics.jsonl')
        with (src/'metrics_epoch.csv').open(encoding='utf8',newline='') as f:table=list(csv.DictReader(f))
        assert [v['epoch'] for v in native]==list(range(1,21)) and len(compact)==len(table)==20
        for a,b,c in zip(native,compact,table):
            for key in ('train_loss_tx_labeled','train_loss_unlabeled','val_tx_acc','lr'):
                assert math.isclose(float(a[key]),float(b[key]),abs_tol=1e-10) and math.isclose(float(a[key]),float(c[key]),abs_tol=1e-10)
        steps=records(src/'step_metrics.jsonl.gz');mix=records(src/'satmix_steps.jsonl.gz');unit=records(src/'unit_steps.jsonl.gz')
        pa=records(src/'shift_steps.jsonl.gz');schedule=read(src/'shift_actual_schedule.json');shift_draws.append(pa)
        assert [v['training_epoch'] for v in schedule]==list(range(1,21)) and [v['configure_epoch'] for v in schedule]==[min(v,13) for v in range(1,21)] and all(v['state']==schedule[12]['state'] for v in schedule[13:])
        expected=list(range(1,4441))
        assert all([v['step'] for v in values]==expected for values in (steps,mix,unit,pa))
        for values in (steps,mix,unit):
            for value in values:finite_numbers(value)
        truth=skips=terms=satellite=0;loss_error=0.;input_error=0.;aggregate={};nonzero=changed=gate_calls=inspection=0
        for native_step,v,u,p in zip(steps,mix,unit,pa):
            epoch=(v['step']-1)//222+1;weight=.1*min(max((epoch-5)/10,0.),1.)
            assert v['epoch']==epoch and v['same_IQ_pairs'] and v['all_L_supervised'] and math.isclose(v['supcon_weight'],weight,abs_tol=1e-12)
            assert v['As_packets']==0 and len(v['supcon_details'])==(4 if epoch>5 else 0)
            assert sum(v['scene_counts'].values())==v['As_packets'] and sum(v['state_counts'].values())==v['As_packets'] and sum(v['lock_counts'].values())==v['As_packets']
            terms+=len(v['supcon_details']);satellite+=v['As_packets']
            finite_numbers(p)
            assert p['count']==128 and p['step']==v['step'] and p['same_native_shift_draw'] and p['RNG_inspection_no_change'] and p['bypass_returned']==(arm=='B10')
            assert len(p['shifts'])==len(p['nonzero_bits'])==len(p['shift_changed_bits'])==len(p['PA_strength'])==len(p['PA_triggered_bits'])==128
            assert p['nonzero_bits']==''.join('1' if z else '0' for z in p['shifts']) and p['nonzero']==p['nonzero_bits'].count('1') and p['shift_changed']==p['shift_changed_bits'].count('1')
            assert set(p['nonzero_bits']+p['shift_changed_bits']+p['PA_triggered_bits'])<=set('01') and all(ch=='0' or nz=='1' for nz,ch in zip(p['nonzero_bits'],p['shift_changed_bits']))
            assert p['shift_gate_calls'] in (0,1) and p['shift_inspection_scalar_draws']==128*p['shift_gate_calls'] and (p['shift_gate_calls'] or all(z==0 for z in p['shifts']))
            assert all(isinstance(z,int) and abs(z)<=p['effective_max_time_shift'] for z in p['shifts']) and p['PA_triggered_bits']==''.join('1' if z>0 else '0' for z in p['PA_strength'])
            assert math.isclose(p['effective_p_time_shift'],schedule[epoch-1]['state']['p_time_shift'],abs_tol=1e-12) and math.isclose(p['effective_p_pa'],schedule[epoch-1]['state']['p_pa'],abs_tol=1e-12) and p['effective_max_time_shift']==schedule[epoch-1]['state']['max_time_shift'] and v['sat_raw_MSE']==0
            expected_MSE=p['returned_raw_MSE_sum']/128
            input_error=max(input_error,abs(v['ordinary_raw_MSE']-expected_MSE));assert math.isclose(v['ordinary_raw_MSE'],expected_MSE,rel_tol=1e-5,abs_tol=1e-6)
            nonzero+=p['nonzero'];changed+=p['shift_changed'];gate_calls+=p['shift_gate_calls'];inspection+=p['shift_inspection_scalar_draws']

            formula=.5*(v['raw_CE']+v['raw_KL'])+.25*(v['ordinary_CE']+v['ordinary_KL'])+.25*(v['s_slot_CE']+v['s_slot_KL'])+weight*v['supcon']
            loss_error=max(loss_error,abs(formula-v['total_L']))
            nm=native_step['metrics'];assert math.isclose(float(nm['satmix/total_L']),v['total_L'],abs_tol=1e-6)
            truth+=int(nm.get('train/pseudo_truth_available',0) or 0);skips+=int(nm.get('train/skipped_nonfinite_loss',0) or nm.get('train/skipped_nonfinite_grad',0) or 0)
            assert u['physical_clip_calls']==v['step']
            sums=aggregate.setdefault(epoch,{})
            for key in ('total_L','raw_CE','ordinary_CE','s_slot_CE','raw_KL','ordinary_KL','s_slot_KL','supcon','ordinary_raw_MSE','sat_raw_MSE'):
                sums[key]=sums.get(key,0.)+v[key]/222
        assert terms==13320 and satellite==0 and truth==skips==0 and loss_error<1e-5 and nonzero==summary['nonzero_shift_packets'] and changed==summary['shift_changed_packets'] and gate_calls==summary['shift_gate_calls'] and inspection==summary['shift_inspection_scalar_draws']
        for v in compact:curves.append(dict(row=row,arm=arm,**v,**aggregate[v['epoch']]))
        coverage=read(src/'satellite_supervision_coverage.json');assert coverage['total']==satellite and sum(v['count'] for v in coverage['cells'])==satellite and not coverage['confidence_gate']
        seq=records(src/'training_sequence.jsonl.gz');sequences.append(seq)
        assert all(p[1] in (3,4,6,8) for v in seq for p in v['ids'])
        observed={r:sum(len(v['ids']) for v in seq if v['role']==r) for r in ('L','U')};unique={r:len({tuple(p) for v in seq if v['role']==r for p in v['ids']}) for r in ('L','U')}
        assert observed=={'L':568320,'U':396368} and unique=={'L':5040,'U':45360}
        args=read(src/'actual_native_args.json');native_args.append(args);assert args['seed']==SEED and args['use_aug'] and args['source_fold_held_rx']==1 and args['label_epochs']==13 and args['pseudo_epochs']==7 and args['aug_enable_pa_normal'] and args['aug_p_dac']==0 and not args['aug_enable_class_signature']
        ordinary=read(src/'ordinary_augmentation_actual.json');assert ordinary['no_second_pair_redraw'] and ordinary['practical_input_is_raw']
        stdout=(OUT/'logs'/(row+'.stdout.log')).read_text(encoding='utf8',errors='replace')
        fatal=[v for v in stdout.splitlines() if any(s in v for s in ('Traceback (most recent call last)','CUDA out of memory','FloatingPointError','SIGKILL'))]
        assert not fatal and 'SMOKE PASS own scratch checkpoint; query_read=false' in stdout
        # EMA is copied immediately after the student's2packet initial smoke.
        # Its inherited eval counter2 is not an actual teacher forward.
        NN_total=sum(summary[k] for k in ('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets'))-2+summary['held_forward_packets']
        execution[arm]=dict(full_steps=4440,full_epochs=20,raw_native_epochs=20,full_stdout_lines=len(stdout.splitlines()),stdout_warnings=[v for v in stdout.splitlines() if 'warning' in v.lower()],native_nonfinite_placeholder_fields=sorted({k for row in native for k,v in row.items() if isinstance(v,float) and not math.isfinite(v)}),all_step_numeric_values_finite_or_explicit_null=True,max_loss_scale_recount_error=loss_error,supcon_layer_forward_terms=terms,As_training_observations=satellite,shift_native_calls=len(pa),shift_native_packets=568320,shift_bypass_returned_packets=summary['shift_bypass_returned_packets'],shift_gate_calls=gate_calls,nonzero_shift_packets=nonzero,shift_changed_packets=changed,shift_inspection_scalar_draws=inspection,shift_input_MSE_recount_max_error=input_error,actual_shift_schedule=schedule,As_source_eval_observations=7560,physical_observations=observed,unique_physical_training=unique,L_batches=sum(v['role']=='L' for v in seq),U_batches=sum(v['role']=='U' for v in seq),teacher_inherited_counter_offset_removed=2,actual_network_forward_packets=NN_total,instrumented_counters={k:summary[k] for k in ('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets')},held_forward_packets=8820,elapsed_training_seconds=summary['elapsed_seconds'],final_L_CE=summary['final_L_CE'],seen_V=summary['seen_V'],ordinary_augmentation=ordinary)
    if len(ROWS)==1:
        value=dict(status='COMPLETED_SOURCE_ROW_ANALYZED',run_id=RUN,model_seed=SEED,source_only=True,arm=ROWS[0][:3],scores=scores,execution=execution,target_access=False,comparison_available=False)
        dump(OUT/('analysis_'+ROWS[0][:3]+'.json'),value)
        print(json.dumps(dict(status=value['status'],arm=value['arm'],execution={k:{kk:vv for kk,vv in v.items() if kk!='ordinary_augmentation'} for k,v in execution.items()}),ensure_ascii=False,indent=2));return value
    assert sequences[0]==sequences[1]
    draw_keys=('shifts','nonzero_bits','shift_changed_bits','shift_gate_calls','effective_p_time_shift','effective_p_pa','effective_max_time_shift','PA_triggered_bits','PA_strength')
    shift_matching={k:all(a[k]==b[k] for a,b in zip(shift_draws[0],shift_draws[1])) for k in draw_keys}
    assert all(shift_matching.values()),shift_matching
    differences=[k for k,v in native_args[0].items() if v!=native_args[1].get(k)]
    assert set(differences)=={'candidate_id','output_dir'}
    paired={};strata=[]
    for view in VIEWS:
        b,s=predictions['B00'][view],predictions['B10'][view]
        assert b['labels']==s['labels'] and b['physical_ids']==s['physical_ids']
        repair=sum(p!=y and q==y for p,q,y in zip(b['predictions'],s['predictions'],b['labels']));broken=sum(p==y and q!=y for p,q,y in zip(b['predictions'],s['predictions'],b['labels']))
        diff=100*(scores['B10'][view]['accuracy']-scores['B00'][view]['accuracy']);assert math.isclose(diff,100*(repair-broken)/1260,abs_tol=1e-10)
        paired[view]=dict(accuracy_gain_pp=diff,macro_F1_gain_pp=100*(scores['B10'][view]['macro_F1']-scores['B00'][view]['macro_F1']),repaired=repair,broken=broken)
        for dimension in ('TX','day'):
            for value in (range(6) if dimension=='TX' else (1,2,3)):
                take=[i for i,(y,p) in enumerate(zip(b['labels'],b['physical_ids'])) if (y if dimension=='TX' else p[2])==value]
                pair={arm:score([predictions[arm][view]['labels'][i] for i in take],[predictions[arm][view]['predictions'][i] for i in take]) for arm in ('B00','B10')}
                strata.append(dict(view=view,dimension=dimension,value=value,count=len(take),B00_accuracy=pair['B00']['accuracy'],B10_accuracy=pair['B10']['accuracy'],gain_pp=100*(pair['B10']['accuracy']-pair['B00']['accuracy'])))
    urban=sum(paired[v]['accuracy_gain_pp'] for v in VIEWS[4:])/3;deltaV=100*(execution['B10']['seen_V']['source_val_accuracy']-execution['B00']['seen_V']['source_val_accuracy'])
    conditions=dict(healthy=all(v['final_L_CE']<=.5 for v in execution.values()),clean=paired['clean']['accuracy_gain_pp']>=1.,clean_macroF1=paired['clean']['macro_F1_gain_pp']>=0.,practical_protection=all(paired[v]['accuracy_gain_pp']>=-.5 for v in VIEWS[1:]),seen_V=deltaV>=-.5)
    qualified=all(conditions.values());assert qualified==complete['qualified'] and conditions==read(OUT/'result.json')['qualification']['conditions']
    gaps={arm:{view:dict(clean_minus_suburban_pp=100*(s['clean']['accuracy']-s[view]['accuracy']),absolute_gap_pp=100*abs(s['clean']['accuracy']-s[view]['accuracy'])) for view in VIEWS[1:4]} for arm,s in scores.items()}
    analysis=dict(status='ANALYZED',run_id=RUN,actual_execution='VERIFIED',source_only=True,model_seed=SEED,qualification=dict(qualified=qualified,conditions=conditions,urban_mean_gain_pp=urban,seen_V_gain_pp=deltaV),scores=scores,paired=paired,suburban_gaps=gaps,execution=execution,all_L_U_sequences_identical=True,both_arms_shift_draws_match=shift_matching,CE_scope='Mean CE reported from frozen default-head forward;hard predictions lack logits,so independent CE recount unavailable',native_argument_differences=differences,new_model_epochs=40,new_optimizer_updates=8880,dispatcher_smoke_NN_packets=2,total_actual_NN_forward_packets=sum(v['actual_network_forward_packets'] for v in execution.values())+2,target_access=False,head_fit_updates=0,joint_goal_complete=False,missing=['Independent source shift-bypass training confirmation','B01/B11','factorial interaction','formal target clean+six practical','cross-seed joint verification'])
    dump(OUT/'analysis_summary.json',analysis)
    for name,values in (('training_curves.csv',curves),('source_strata.csv',strata)):
        fields=sorted({k for v in values for k in v})
        with (OUT/name).open('w',encoding='utf8',newline='') as f:writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(values)
    print(json.dumps({k:v for k,v in analysis.items() if k in ('qualification','paired','suburban_gaps','total_actual_NN_forward_packets')},ensure_ascii=False,indent=2));return analysis

if __name__=='__main__':
    if len(sys.argv)>1:
        assert len(sys.argv)==3 and sys.argv[1]=='--row' and sys.argv[2] in ('B00','B10')
        ROWS=(sys.argv[2]+'-RX1-s'+str(SEED),)
    analyze()
