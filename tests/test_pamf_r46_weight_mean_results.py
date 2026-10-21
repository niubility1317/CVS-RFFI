import copy
import pytest
from experiments.pamf_r46_weight_mean import design as d
from tools.analyze_pamf_r46_weight_mean_results import training_metrics,readout_metadata,completed_metrics,METHOD


def source():
    records=[]
    for epoch in range(1,21):
        w=d.teacher_weight(epoch)
        records.append(dict(epoch=epoch,updates=247*epoch,L_exposures=6300*epoch,U_exposures=56700*epoch,
            teacher_IQ_examples=63000*epoch,source_IQ_examples=63000*epoch,F_inputs=0,teacher_weight=w,microbatch=256,
            source_tensor_backend='direct_float32_buffer',EMA_updates=max(0,epoch-10)*247,late_mean_snapshots=max(0,epoch-15),
            shared_source_training_paths=1,losses=dict(loss=.02+.05*.01+w*.03,iq_nmse=.02,spectral_error=.01,teacher_stage_nmse=.03),
            learning_rate_last=d.learning_rate(247*epoch-1),seconds=70.*epoch,averaging_host_enqueue_seconds=.1*epoch,clip_fraction=.25))
    complete=dict(status='SOURCE_TRAINED',arm=d.SOURCE_ARM,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        source_IQ_examples=1260000,teacher_IQ_examples=1260000,student_context_examples=1260000,shared_source_training_paths=1,
        training_counted_once_across_readouts=True,readouts=list(d.ARMS),parameters=273765,trainable_parameters=273765,
        context_parameters=185615,context_updates=4940,coarse_updates=4940,action_updates=4940,prior_updates=4940,
        context_trainable=True,paired_clean_optimized=True,prior_parameters_changed=True,EMA_updates=2470,late_mean_snapshots=5,
        inherited_component_epochs=80,cumulative_coarse_epochs=100,cumulative_action_epochs=100,cumulative_prior_epochs=100,
        cumulative_component_epoch_sum=300,cumulative_serial_training_passes=140,initialization_IQ_examples=0,
        new_calibration_IQ=0,new_trainable_parameters=0,extra_training_neural_forwards=0,external_student_checkpoint_reads=0,
        TX_labels_read=0,V_reads=0,target_reads=0,training_F_access=False,F_weights_read=0,F_outputs_read=0,
        teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0,R17_training_forward_examples=0,F_training_reference=None,
        seconds=1401.,seconds_including_all_readout_serialization=1402.,averaging_host_enqueue_seconds=2.,peak_allocated_bytes=1000000)
    return records,complete


def info(arm):
    return dict(readout=arm,rule=d.configuration()['readouts'][arm],shared_source_epochs=20,shared_source_updates=4940,
        ema_updates=2470,late_epoch_snapshots=5,unique_parameters_averaged=714,parameter_aliases=770,buffers_averaged=False,
        teacher_changed=False,extra_neural_forwards=0,new_trainable_parameters=0,TX_reads=0,F_reads=0,V_reads=0,target_reads=0)


def summary(role):
    count=1800 if role=='source' else 168000
    def record(matrix,dimension,value,view):
        return dict(confusion=matrix,count=sum(map(sum,matrix)),accuracy=100.,macro_f1=100.,baseline_accuracy=100.,
            repaired=0,damaged=0,net_repaired=0,dimension=dimension,value=value,view=view,readout='restored')
    matrix=[[count//6 if i==j else 0 for j in range(6)] for i in range(6)]
    records=[]
    for view in d.VIEWS:
        records.append(record(matrix,'overall','all',view))
        for tx in range(6):records.append(record([[count//6 if i==j==tx else 0 for j in range(6)] for i in range(6)],'TX',str(tx),view))
        for dim in ('RX','day'):records.append(record(matrix,dim,'0',view))
    # Single-class groups have macroF1=1/6 while overall has macroF1=1.
    for r in records:
        if r['dimension']=='TX':r['macro_f1']=100./6
    return dict(status='ANALYZED',role=role,count=count,F_seed=d.F_SEED,original_default_head=True,F_run=d.F_RUN,
        target_access=role=='target',all_predictions_before_truth=True,training_F_access=False,training_TX_labels_read=0,head_fit_updates=0,
        views={v:dict(restored=dict(waveform=dict(iq_nmse=.01,prefix232_nmse=.01,tail24_nmse=.01,fft_log_power_L1=.01))) for v in d.VIEWS},records=records)


def profile(arm):
    cases={}
    for mode in ('G_only','G_plus_F'):
        for batch,w,n in ((1,16,64),(32,4,16)):
            cases[mode+'_B'+str(batch)]=dict(batch=batch,warmup_per_round=w,iterations_per_round=n,rounds=3,source_V_unique_IQ=64,
                median_ms=1.,p90_ms=2.,p95_ms=3.,throughput_samples_per_second=10.,resident_before_forward_bytes=100.,peak_allocated_bytes=200.,incremental_peak_bytes=100.,round_median_ms=[1.,1.,1.])
    return dict(status='PROFILE_COMPLETE',includes_full_context_3prior_6CG=True,checkpoint_payload_on_GPU=False,optimizer_on_GPU=False,
        source_V_IQ=64,TX_labels_read=0,state_updates=0,physical_GPU=d.GPUS[arm],training_F_access=False,reader_weights_read_for_evaluation_only=True,
        results={name:dict(parameters=size,cases=copy.deepcopy(cases),ConvLinear_MAC_per_IQ=100) for name,size in dict(original_R17=4665073,parent_R46_350521=350521,compact=273765).items()})


def collection():
    records,trained=source()
    init=dict(mode='original_M35_static_prune_shared_source_path',group_count=4,IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,
        target_reads=0,optimizer_updates=0,parent_unchanged=True,exact_function_equivalence=False,fixed_original_maps=True,
        parameter_mean_readouts=list(d.ARMS),external_student_checkpoint_reads=0,new_calibration_IQ=0,new_trainable_parameters=0)
    rows={}
    for arm in d.ARMS:
        row=dict(complete=dict(status='ANALYZED'),failure=None,parameter_readout=info(arm),matched_profile=profile(arm),checkpoint_bytes=1000)
        for role in ('source','target'):
            row[role+'_summary']=summary(role)
            row[role+'_prediction_complete']=dict(status='PREDICTIONS_COMPLETE',role=role,count=1800 if role=='source' else 168000,
                views=list(d.VIEWS),readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=d.F_RUN,F_seed=d.F_SEED,
                seconds=8.,per_view_seconds={v:1. for v in d.VIEWS})
        rows[arm]=row
    return dict(run_id=d.RUN,method_commit=METHOD,failure=None,actual_live_owned_PIDs=[],rows=rows,
        complete=dict(status='ANALYZED',source_rows=list(d.ARMS),target_rows=list(d.ARMS),no_source_score_selection=True,shared_source_training_paths=1,training_counted_once=True),
        shared_source=dict(training_records=records,training_complete=trained,initialization=init,completion=dict(status='SOURCE_TRAINED',source_worker_exited=True,source_training_counted_once=True,readouts=list(d.ARMS))),
        baselines={name:{role:summary(role) for role in ('source','target')} for name in ('R17','R44','R46')})


def test_complete_matrix_recomputes_metrics_and_counts_shared_training_only_once():
    raw=collection();before=copy.deepcopy(raw);result=completed_metrics(raw)
    assert result['shared_source_training']['source_training_paths']==1
    assert result['shared_source_training']['training_complete']['source_IQ_examples']==1260000
    assert len(result['rows'])==3 and all('training' not in r for r in result['rows'].values())
    assert set(result['same_trajectory_comparisons'])=={'late_mean','step_ema'}
    assert all(c['control']=='final' and c['additional_training_examples']==0 for c in result['same_trajectory_comparisons'].values())
    assert raw==before and not result['goal_complete']


@pytest.mark.parametrize('field,bad',[('F_outputs_read',1),('TX_labels_read',1),('V_reads',1),('target_reads',1),('U_exposures',0),
    ('shared_source_training_paths',3),('training_counted_once_across_readouts',False),('EMA_updates',2469),('late_mean_snapshots',4),
    ('extra_training_neural_forwards',1),('seconds_including_all_readout_serialization',1300),('averaging_host_enqueue_seconds',0)])
def test_permissions_full_budget_mean_counts_and_cost(field,bad):
    records,trained=source();trained[field]=bad
    with pytest.raises(ValueError):training_metrics(records,trained)


@pytest.mark.parametrize('field,bad',[('EMA_updates',1),('late_mean_snapshots',1),('F_inputs',1),('shared_source_training_paths',3),('teacher_weight',9),('learning_rate_last',1),('seconds',0),('averaging_host_enqueue_seconds',-1)])
def test_every_epoch_is_checked(field,bad):
    records,trained=source();records[5][field]=bad
    with pytest.raises(ValueError):training_metrics(records,trained)


@pytest.mark.parametrize('damage',['missing_readout','missing_view','truth_early','source_live','readout_retrained','source_tripled','alias_count','window','live_worker','failed','profile_missing','prediction_cost_missing','prediction_time_understates'])
def test_incomplete_incorrect_or_duplicated_training_never_becomes_final(damage):
    value=collection()
    if damage=='missing_readout':value['rows'].pop('step_ema')
    elif damage=='missing_view':value['rows']['step_ema']['target_prediction_complete']['views'].pop()
    elif damage=='truth_early':value['rows']['final']['target_summary']['all_predictions_before_truth']=False
    elif damage=='source_live':value['shared_source']['completion']['source_worker_exited']=False
    elif damage=='readout_retrained':value['rows']['late_mean']['training_complete']={}
    elif damage=='source_tripled':value['complete']['shared_source_training_paths']=3
    elif damage=='alias_count':value['rows']['late_mean']['parameter_readout']['unique_parameters_averaged']=770
    elif damage=='window':value['rows']['late_mean']['parameter_readout']['rule']['epochs']=[15,16,17,18,19]
    elif damage=='live_worker':value['actual_live_owned_PIDs']=[1]
    elif damage=='failed':value['failure']={}
    elif damage=='profile_missing':value['rows']['final']['matched_profile']['results']['compact']['cases'].pop('G_plus_F_B1')
    elif damage=='prediction_cost_missing':value['rows']['late_mean']['source_prediction_complete'].pop('per_view_seconds')
    elif damage=='prediction_time_understates':value['rows']['late_mean']['target_prediction_complete']['seconds']=1.
    with pytest.raises(ValueError):completed_metrics(value)
