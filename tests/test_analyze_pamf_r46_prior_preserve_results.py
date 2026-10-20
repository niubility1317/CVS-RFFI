import copy,json,runpy
from pathlib import Path
import pytest
from experiments.pamf_r46_prior_preserve import design as d
from tools.analyze_pamf_r46_prior_preserve_results import completed_metrics,METHOD

summary=runpy.run_path(str(Path(__file__).with_name('test_analyze_pamf_r17_task_results.py')))['summary']

def fixture():
    rows={}
    for arm,errors in zip(d.ARMS,(1444,1445)):
        records=[]
        for epoch in range(1,21):
            terms=dict(iq_nmse=.1/epoch,spectral_error=.02/epoch,teacher_stage_nmse=.01/epoch,conditional_prior_nmse=.05/epoch if d.CONDITIONAL_WEIGHT[arm] else 0.)
            loss=terms['iq_nmse']+.05*terms['spectral_error']+d.teacher_weight(epoch)*terms['teacher_stage_nmse']+d.CONDITIONAL_WEIGHT[arm]*terms['conditional_prior_nmse']
            records.append(dict(epoch=epoch,updates=247*epoch,L_exposures=6300*epoch,U_exposures=56700*epoch,teacher_IQ_examples=63000*epoch,source_IQ_examples=63000*epoch,F_inputs=0,seconds=200.*epoch,clip_fraction=.1,teacher_weight=d.teacher_weight(epoch),learning_rate_last=d.learning_rate(247*epoch-1),losses=dict(loss=loss,**terms)))
        trained=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,source_IQ_examples=1260000,teacher_IQ_examples=1260000,student_context_examples=0,shared_context_examples=1260000,
            parameters=d.PARAMETERS[arm],trainable_parameters=d.PRIOR_PARAMETERS[arm],context_parameters=237927,context_trainable=False,context_updates=0,coarse_updates=0,action_updates=0,prior_updates=4940,
            cumulative_coarse_epochs=80,cumulative_action_epochs=80,cumulative_prior_epochs=100,cumulative_component_epoch_sum=260,cumulative_serial_training_passes=140,initialization_IQ_examples=0,
            TX_labels_read=0,V_reads=0,target_reads=0,training_F_access=False,F_weights_read=0,F_outputs_read=0,teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0,paired_clean_optimized=True,
            conditional_weight=d.CONDITIONAL_WEIGHT[arm],conditional_teacher_prior_examples=3780000*int(d.CONDITIONAL_WEIGHT[arm]>0))
        profile=dict(status='PROFILE_COMPLETE',includes_full_context_3prior_6CG=True,checkpoint_payload_on_GPU=False,optimizer_on_GPU=False,source_V_IQ=64,TX_labels_read=0,state_updates=0,training_F_access=False,physical_GPU=d.GPUS[arm],results={})
        for name,size in dict(original_R17=4665073,parent_R46_350521=350521,compact=d.PARAMETERS[arm]).items():
            cases={}
            for mode in ('G_only','G_plus_F'):
                for batch in (1,32):cases[mode+'_B'+str(batch)]=dict(batch=batch,rounds=3,iterations_per_round=64 if batch==1 else 16,warmup_per_round=16 if batch==1 else 4,source_V_unique_IQ=64,median_ms=10.,p90_ms=12.,p95_ms=13.,throughput_samples_per_second=100.,resident_before_forward_bytes=50,peak_allocated_bytes=100,incremental_peak_bytes=50)
            profile['results'][name]=dict(parameters=size,cases=cases)
        row=dict(training_records=records,training_complete=trained,matched_profile=profile,checkpoint_bytes=100,complete=dict(status='ANALYZED'),failure=None,stdout_findings=[])
        for role in ('source','target'):
            row[role+'_summary']=summary(role,errors if role=='target' else 180)
            row[role+'_prediction_complete']=dict(status='PREDICTIONS_COMPLETE',role=role,count=1800 if role=='source' else 168000,views=list(d.VIEWS),readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=d.F_RUN,F_seed=d.F_SEED)
        rows[arm]=row
    return dict(run_id=d.RUN,method_commit=METHOD,failure=None,actual_live_owned_PIDs=[],complete=dict(status='ANALYZED',source_rows=list(d.ARMS),target_rows=list(d.ARMS)),rows=rows,baselines={name:dict(source=summary(),target=summary('target',errors)) for name,errors in (('R17',200),('R46',100))})

def test_same_parent_exact08pp_boundary_and_4940_source_update_accounting():
    result=completed_metrics(fixture())
    assert result['accepted_rows']==['prior625_cond']
    assert result['rows']['prior625_cond']['acceptance']['R46']['maximum_correct_loss']==1344
    assert result['rows']['prior750_cond']['training']['training_complete']['conditional_teacher_prior_examples']==3780000
    assert result['rows']['prior625_cond']['training']['full_log_records']==20
    assert result['matched_rank750_vs_rank625']['views']['clean']['correct_delta']==-1

@pytest.mark.parametrize('bad',['missingrow','missingview','targettruth','Ffree','epochs','loss','profile','component','conditional','conditionalcost','live'])
def test_incomplete_wrong_cost_or_permissions_are_rejected(bad):
    value=fixture();row=value['rows']['prior750_cond']
    if bad=='missingrow':value['rows'].pop('prior625_cond')
    elif bad=='missingview':row['target_prediction_complete']['views'].pop()
    elif bad=='targettruth':row['training_complete']['target_reads']=1
    elif bad=='Ffree':row['training_records'][-1]['F_inputs']=1
    elif bad=='epochs':row['training_records'].pop()
    elif bad=='loss':row['training_records'][-1]['losses']['loss']+=1
    elif bad=='profile':row['matched_profile']['results']['compact']['cases'].pop('G_plus_F_B32')
    elif bad=='component':row['training_complete']['coarse_updates']=1
    elif bad=='conditional':row['training_complete']['conditional_weight']=0.
    elif bad=='conditionalcost':row['training_complete']['conditional_teacher_prior_examples']=0
    else:value['actual_live_owned_PIDs']=[123]
    with pytest.raises(ValueError):completed_metrics(value)

def test_report_renders_every_scene_cost_and_group_without_IO(tmp_path,monkeypatch):
    from tools import complete_pamf_r46_prior_preserve as delivery
    local=tmp_path/'local';workspace=tmp_path/'workspace';folder=local/'automation_reports/CV-SincNet'/d.RUN;folder.mkdir(parents=True)
    (folder/'report.md').write_text('状态：VERIFIED / RUNNING。\n',encoding='utf-8');(folder/'experiment.json').write_text('{"execution":{}}',encoding='utf-8');(folder/'events.jsonl').write_text('',encoding='utf-8')
    output=local/'artifacts';output.mkdir();artifacts=dict(status='VERIFIED',models={a:dict(components=dict(coarse=1,action=2,prior=3),conv_linear_MACs_per_IQ=100,bytes=100) for a in d.ARMS})
    monkeypatch.setattr(delivery,'WORK',workspace);result=completed_metrics(fixture());delivery.report(local,output,result,artifacts,record_events=False)
    text=(folder/'report.md').read_text(encoding='utf-8')
    assert '状态：VERIFIED / ANALYZED。' in text
    assert all('|'+a+'|'+v+'|' in text for a in d.ARMS for v in d.VIEWS)
    assert all(mode in text for mode in ('G_only_B1','G_only_B32','G_plus_F_B1','G_plus_F_B32'))
    assert len((folder/'row_metrics.csv').read_text(encoding='utf-8').splitlines())==393
    saved=json.loads((folder/'results.json').read_text(encoding='utf-8'));assert saved['accepted_rows']==['prior625_cond'] and not saved['target_feedback']
