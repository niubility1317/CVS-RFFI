import copy
import pytest
from experiments.pamf_r46_delayed_prune import design as d
from tools.analyze_pamf_r46_delayed_prune_results import training_metrics,completed_metrics


def source_records(arm):
    records=[]
    for epoch in range(1,21):
        weight=d.teacher_weight(epoch)
        records.append(dict(epoch=epoch,updates=247*epoch,L_exposures=6300*epoch,U_exposures=56700*epoch,
            teacher_IQ_examples=63000*epoch,source_IQ_examples=63000*epoch,F_inputs=0,teacher_weight=weight,microbatch=256,
            phase=d.phase(epoch,arm),parameters=350521 if d.phase(epoch,arm)=='dense' else 273765,optimizer_resets=int(epoch>d.WARMUP_EPOCHS[arm]),
            source_tensor_backend='direct_float32_buffer',losses=dict(loss=.02+.05*.01+weight*.03,iq_nmse=.02,spectral_error=.01,teacher_stage_nmse=.03),
            seconds=70.*epoch,clip_fraction=.25,learning_rate_last=d.learning_rate(247*epoch-1)))
    complete=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        source_IQ_examples=1260000,teacher_IQ_examples=1260000,student_context_examples=1260000,
        parameters=d.PARAMETERS[arm],trainable_parameters=d.PARAMETERS[arm],context_parameters=d.CONTEXT_PARAMETERS[arm],
        context_updates=4940,coarse_updates=4940,action_updates=4940,prior_updates=4940,context_trainable=True,
        cumulative_coarse_epochs=100,cumulative_action_epochs=100,cumulative_prior_epochs=100,cumulative_component_epoch_sum=300,cumulative_serial_training_passes=140,
        initialization_IQ_examples=0,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,V_reads=0,target_reads=0,
        teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0,paired_clean_optimized=True,
        prior_parameters_changed=True,new_calibration_IQ=0,new_trainable_parameters=0,external_student_checkpoint_reads=0,
        dense_epochs=d.WARMUP_EPOCHS[arm],compact_epochs=20-d.WARMUP_EPOCHS[arm],dense_IQ_examples=63000*d.WARMUP_EPOCHS[arm],compact_IQ_examples=63000*(20-d.WARMUP_EPOCHS[arm]),
        optimizer_resets=1,channel_maps_unchanged=True,global_schedule_restarted=False,phase_seconds=dict(dense=700.,compact=700.))
    return records,complete


@pytest.mark.parametrize('arm',d.ARMS)
def test_reconstruction_checks_every_epoch_and_fixed_source_contract(arm):
    records,complete=source_records(arm);before=copy.deepcopy((records,complete))
    checked=training_metrics(records,complete,arm)
    assert checked['full_log_records']==20 and checked['epoch_seconds_median']==70.
    assert checked['checkpoint_selection']=='fixed final E20/4940'
    assert (records,complete)==before


@pytest.mark.parametrize('field,value',[('F_weights_read',1),('TX_labels_read',1),('target_reads',1),('V_reads',1),
    ('U_exposures',0),('updates',4939),('dense_epochs',6),('optimizer_resets',0),('prior_parameters_changed',False),('new_calibration_IQ',128)])
def test_source_permissions_budget_and_actual_grouping_are_not_reported_as_valid(field,value):
    records,complete=source_records('prune_after5');complete[field]=value
    with pytest.raises(ValueError):training_metrics(records,complete,'prune_after5')


@pytest.mark.parametrize('damage',['missing_epoch','middle_F_access','unclosed_loss','teacher_schedule','nonmonotone_time','wrong_lr','wrong_phase','wrong_dense_size','wrong_reset_count','reset_teacher_at_transition','reset_lr_at_transition'])
def test_full_curve_validation_does_not_depend_only_on_last_epoch(damage):
    records,complete=source_records('prune_after10')
    if damage=='missing_epoch':records.pop(5)
    elif damage=='middle_F_access':records[5]['F_inputs']=1
    elif damage=='unclosed_loss':records[5]['losses']['loss']+=.01
    elif damage=='teacher_schedule':records[5]['teacher_weight']=1.
    elif damage=='nonmonotone_time':records[5]['seconds']=records[4]['seconds']
    elif damage=='wrong_lr':records[5]['learning_rate_last']=1.
    elif damage=='wrong_phase':records[9]['phase']='compact'
    elif damage=='wrong_dense_size':records[5]['parameters']=273765
    elif damage=='wrong_reset_count':records[10]['optimizer_resets']=0
    elif damage=='reset_teacher_at_transition':records[10]['teacher_weight']=1.
    elif damage=='reset_lr_at_transition':records[10]['learning_rate_last']=d.learning_rate(0)
    with pytest.raises(ValueError):training_metrics(records,complete,'prune_after10')


@pytest.mark.parametrize('value',[{},dict(run_id=d.RUN,method_commit='wrong'),dict(run_id=d.RUN,method_commit='4fd058f9bae79687f924e06d057e6a20083ff6dc',failure=None,actual_live_owned_PIDs=[1])])
def test_incomplete_or_live_matrix_never_becomes_analyzed(value):
    with pytest.raises(ValueError):completed_metrics(value)


@pytest.mark.parametrize('bad',[{},dict(dense=0.,compact=1.),dict(dense=float('nan'),compact=1.)])
def test_both_actual_phase_costs_required(bad):
    records,complete=source_records(d.ARMS[0]);complete['phase_seconds']=bad
    with pytest.raises(ValueError):training_metrics(records,complete,d.ARMS[0])
