import copy,json,runpy
from pathlib import Path
import pytest
from experiments.pamf_r46_activation_factor import design as d
from tools.analyze_pamf_r46_activation_factor_results import completed_metrics,METHOD

summary=runpy.run_path(str(Path(__file__).with_name('test_analyze_pamf_r17_task_results.py')))['summary']

def calibration():
    return dict(schema='R46_source_activation_moments_v1',run=d.RUN,config=d.calibration_configuration(),status='SOURCE_CALIBRATED',
        teacher_examples=768,unique_source_positions=256,L_exposures=384,U_exposures=384,F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,
        optimizer_updates=0,parent_unchanged=True,seconds=9.,groups={'prior::synthetic':dict(patch_count=18432,shared_stage_uses=3)})


def initialization(arm):
    return dict(mode=d.MODE[arm],group_count=4,IQ_examples=768,unique_source_positions=256,calibration_L_exposures=384,
        calibration_U_exposures=384,positions_per_conv=8,ridge_fraction=.001,rank_fraction=d.FRACTIONS[arm],TX_labels_read=0,F_reads=0,V_reads=0,
        target_reads=0,optimizer_updates=0,parent_unchanged=True,exact_function_equivalence=False,channel_dimensions_unchanged=True,
        normalization_values_exact=True,activation_calibrated=True,shared_prior_input_moments=True,seconds=.3,
        factor_layers={'prior::synthetic':dict(rank=5,original_kernel_parameters=100,factor_kernel_parameters=70,shared_stage_uses=3,regularized_output_relative_squared_error=.1)})


def fixture():
    rows={}
    for arm,errors in zip(d.ARMS,(1444,1445)):
        records=[]
        for epoch in range(1,21):
            terms=dict(iq_nmse=.1/epoch,spectral_error=.02/epoch,teacher_stage_nmse=.01/epoch,balanced_complex_nmse=.04/epoch,clean_identity_nmse=.05/epoch if d.CLEAN_WEIGHT[arm] else 0.)
            loss=terms['iq_nmse']+.05*terms['spectral_error']+d.teacher_weight(epoch)*terms['teacher_stage_nmse']+d.COMPLEX_WEIGHT*terms['balanced_complex_nmse']+d.CLEAN_WEIGHT[arm]*terms['clean_identity_nmse']
            records.append(dict(epoch=epoch,updates=985*epoch,L_exposures=6300*epoch,U_exposures=56700*epoch,teacher_IQ_examples=63000*epoch,source_IQ_examples=63000*epoch,F_inputs=0,extra_clean_student_inputs=63000*epoch*int(d.CLEAN_WEIGHT[arm]>0),seconds=200.*epoch,clip_fraction=.1,teacher_weight=d.teacher_weight(epoch),learning_rate_last=d.learning_rate(985*epoch-1),losses=dict(loss=loss,**terms)))
        trained=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=19700,L_exposures=126000,U_exposures=1134000,source_IQ_examples=1260000,teacher_IQ_examples=1260000,student_context_examples=1260000,
            parameters=d.PARAMETERS[arm],trainable_parameters=d.PARAMETERS[arm],context_parameters=d.CONTEXT_PARAMETERS[arm],context_trainable=True,context_updates=19700,coarse_updates=19700,action_updates=19700,prior_updates=19700,
            cumulative_coarse_epochs=100,cumulative_action_epochs=100,cumulative_prior_epochs=100,cumulative_component_epoch_sum=300,cumulative_serial_training_passes=140,initialization_IQ_examples=d.CALIBRATION_IQ,calibration_unique_IQ=d.CAL_UNIQUE,calibration_L_exposures=384,calibration_U_exposures=384,calibration_teacher_IQ_examples=768,
            TX_labels_read=0,V_reads=0,target_reads=0,training_F_access=False,F_weights_read=0,F_outputs_read=0,teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0,paired_clean_optimized=True,
            balanced_complex_weight=d.COMPLEX_WEIGHT,clean_identity_weight=d.CLEAN_WEIGHT[arm],extra_clean_student_inputs=1260000*int(d.CLEAN_WEIGHT[arm]>0),conditional_teacher_prior_examples=0)
        profile=dict(status='PROFILE_COMPLETE',includes_full_context_3prior_6CG=True,checkpoint_payload_on_GPU=False,optimizer_on_GPU=False,source_V_IQ=64,TX_labels_read=0,state_updates=0,training_F_access=False,physical_GPU=d.GPUS[arm],results={})
        for name,size in dict(original_R17=4665073,parent_R46_350521=350521,compact=d.PARAMETERS[arm]).items():
            cases={}
            for mode in ('G_only','G_plus_F'):
                for batch in (1,32):cases[mode+'_B'+str(batch)]=dict(batch=batch,rounds=3,iterations_per_round=64 if batch==1 else 16,warmup_per_round=16 if batch==1 else 4,source_V_unique_IQ=64,median_ms=10.,p90_ms=12.,p95_ms=13.,throughput_samples_per_second=100.,resident_before_forward_bytes=50,peak_allocated_bytes=100,incremental_peak_bytes=50)
            profile['results'][name]=dict(parameters=size,cases=cases)
        row=dict(training_records=records,training_complete=trained,initialization=initialization(arm),matched_profile=profile,checkpoint_bytes=100,complete=dict(status='ANALYZED'),failure=None,stdout_findings=[])
        for role in ('source','target'):
            row[role+'_summary']=summary(role,errors if role=='target' else 180)
            row[role+'_prediction_complete']=dict(status='PREDICTIONS_COMPLETE',role=role,count=1800 if role=='source' else 168000,views=list(d.VIEWS),readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=d.F_RUN,F_seed=d.F_SEED)
        rows[arm]=row
    return dict(run_id=d.RUN,method_commit=METHOD,calibration=calibration(),failure=None,actual_live_owned_PIDs=[],complete=dict(status='ANALYZED',source_rows=list(d.ARMS),target_rows=list(d.ARMS)),rows=rows,baselines={name:dict(source=summary(),target=summary('target',errors)) for name,errors in (('R17',200),('R46',100),('R22identity',100))})

def test_same_parent_exact08pp_boundary_and_19700_source_update_accounting():
    result=completed_metrics(fixture())
    assert result['accepted_rows']==[d.ARMS[0]]
    assert result['rows'][d.ARMS[0]]['acceptance']['R46']['maximum_correct_loss']==1344
    assert result['rows'][d.ARMS[1]]['training']['training_complete']['extra_clean_student_inputs']==1260000
    assert result['rows'][d.ARMS[0]]['training']['full_log_records']==20
    assert result['matched_rank500_vs_rank625']['views']['clean']['correct_delta']==-1

@pytest.mark.parametrize('bad',['missingrow','missingview','targettruth','Ffree','epochs','loss','profile','component','conditional','conditionalcost','live'])
def test_incomplete_wrong_cost_or_permissions_are_rejected(bad):
    value=fixture();row=value['rows'][d.ARMS[1]]
    if bad=='missingrow':value['rows'].pop(d.ARMS[0])
    elif bad=='missingview':row['target_prediction_complete']['views'].pop()
    elif bad=='targettruth':row['training_complete']['target_reads']=1
    elif bad=='Ffree':row['training_records'][-1]['F_inputs']=1
    elif bad=='epochs':row['training_records'].pop()
    elif bad=='loss':row['training_records'][-1]['losses']['loss']+=1
    elif bad=='profile':row['matched_profile']['results']['compact']['cases'].pop('G_plus_F_B32')
    elif bad=='component':row['training_complete']['coarse_updates']=4940
    elif bad=='conditional':row['training_complete']['clean_identity_weight']=0.
    elif bad=='conditionalcost':row['training_complete']['extra_clean_student_inputs']=0
    else:value['actual_live_owned_PIDs']=[123]
    with pytest.raises(ValueError):completed_metrics(value)

def test_report_renders_every_scene_cost_and_group_without_IO(tmp_path,monkeypatch):
    from tools import complete_pamf_r46_activation_factor as delivery
    local=tmp_path/'local';workspace=tmp_path/'workspace';folder=local/'automation_reports/CV-SincNet'/d.RUN;folder.mkdir(parents=True)
    (folder/'report.md').write_text('状态：VERIFIED / RUNNING。\n',encoding='utf-8');(folder/'experiment.json').write_text('{"execution":{}}',encoding='utf-8');(folder/'events.jsonl').write_text('',encoding='utf-8')
    output=local/'artifacts';output.mkdir();artifacts=dict(status='VERIFIED',models={a:dict(components=dict(coarse=1,action=2,prior=3),conv_linear_MACs_per_IQ=100,bytes=100) for a in d.ARMS})
    monkeypatch.setattr(delivery,'WORK',workspace);result=completed_metrics(fixture());delivery.report(local,output,result,artifacts,record_events=False)
    text=(folder/'report.md').read_text(encoding='utf-8')
    assert '状态：VERIFIED / ANALYZED。' in text
    assert all('|'+a+'|'+v+'|' in text for a in d.ARMS for v in d.VIEWS)
    assert all(mode in text for mode in ('G_only_B1','G_only_B32','G_plus_F_B1','G_plus_F_B32'))
    assert len((folder/'row_metrics.csv').read_text(encoding='utf-8').splitlines())==393
    saved=json.loads((folder/'results.json').read_text(encoding='utf-8'));assert saved['accepted_rows']==[d.ARMS[0]] and not saved['target_feedback']


@pytest.mark.parametrize('bad',['calibration_exposure','identity_exposure','complex_weight'])
def test_historical_control_and_new_objective_accounting_required(bad):
    value=fixture();row=value['rows'][d.ARMS[1]]
    if bad=='calibration_exposure':value['calibration']['teacher_examples']=0
    elif bad=='identity_exposure':row['training_records'][-1]['extra_clean_student_inputs']=0
    else:row['training_complete']['balanced_complex_weight']=0
    with pytest.raises(ValueError):completed_metrics(value)


def test_missing_prospective_control_is_nonblocking_and_variable_rank_cost_is_preserved():
    value=fixture();value['baselines'].pop('R22identity');result=completed_metrics(value)
    assert result['rows'][d.ARMS[1]]['parameters']==236355
    assert result['rows'][d.ARMS[1]]['profile']['results']['compact']['parameters']==236355
    assert result['rows'][d.ARMS[0]]['target_vs_R22identity'] is None
    assert result['calibration']['teacher_examples']==768


@pytest.mark.parametrize('bad',['F_read','shared_prior_uses','rank','calibration_training_count'])
def test_calibration_and_initialization_cost_or_permissions_cannot_be_dropped(bad):
    value=fixture()
    if bad=='F_read':value['calibration']['F_reads']=1
    elif bad=='shared_prior_uses':value['calibration']['groups']['prior::synthetic']['shared_stage_uses']=1
    elif bad=='rank':value['rows'][d.ARMS[1]]['initialization']['rank_fraction']=.625
    else:value['rows'][d.ARMS[0]]['training_complete']['calibration_L_exposures']=0
    with pytest.raises(ValueError):completed_metrics(value)
