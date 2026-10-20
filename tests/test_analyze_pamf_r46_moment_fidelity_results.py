import copy,importlib,json
import pytest


def tools():
    return importlib.import_module('tools.analyze_pamf_r46_moment_fidelity_results')


def fixture():
    m=tools();d=m.d
    old=importlib.import_module('tests.test_analyze_pamf_r46_complex_fidelity_results').fixture()
    old.update(run_id=d.RUN,method_commit=m.METHOD)
    old['complete'].update(source_rows=list(d.ARMS),target_rows=list(d.ARMS))
    old['baselines']['R22identity']=copy.deepcopy(old['baselines']['R18stage'])
    old['rows']=dict(zip(d.ARMS,old['rows'].values()))
    for arm,row in old['rows'].items():
        for record in row['training_records']:
            epoch=record['epoch'];loss=record['losses'];loss['clean_identity_nmse']=.05/epoch
            base=loss['iq_nmse']+.05*loss['spectral_error']+d.teacher_weight(epoch)*loss['teacher_stage_nmse']+.02*loss['balanced_complex_nmse']+.25*loss['clean_identity_nmse']
            cm,tm=.003/epoch,.001/epoch;selected=cm if arm=='moment_clean' else tm
            loss.update(base_loss=base,clean_moment_mse=cm,teacher_moment_mse=tm,moment_objective=selected,weighted_moment_objective=selected,loss=base+selected)
            record.update(extra_clean_student_inputs=63000*epoch,moment_reference=d.MOMENT_REFERENCE[arm],moment_weight=1.,extra_moment_G_inputs=0,moment_descriptor_inputs=189000*epoch)
        row['training_complete'].update(arm=arm,extra_clean_student_inputs=1260000,clean_identity_weight=.25,student_total_IQ_examples=2520000,
            moment_weight=1.,moment_reference=d.MOMENT_REFERENCE[arm],extra_moment_G_inputs=0,moment_descriptor_inputs=3780000)
    return old


def test_both_moment_references_full_losses_counts_and_formal7():
    m=tools();result=m.completed_metrics(fixture())
    assert result['accepted_rows']==['moment_clean']
    assert result['matched_teacher_vs_clean']['views']['clean']['correct_delta']==-1
    for row in result['rows'].values():
        assert row['training']['full_log_records']==20 and len(row['target']['views'])==7
        assert len(row['training']['losses'])==11
        assert row['training']['training_complete']['student_total_IQ_examples']==2520000
    assert result['matched_forward_exposure']['extra_moment_G_inputs']==0


@pytest.mark.parametrize('bad',['moment_weight','moment_reference','weighted_loss','selected_loss','descriptor_count','extraG',
    'student_count','teacher_count','identity_count','Ffree','targettruth','missingview','missingrow','live','historical_reference','epochs','nonfinite','base_loss'])
def test_inconsistent_moment_or_formal_evidence_rejected(bad):
    m=tools();value=fixture();row=value['rows']['moment_teacher'];last=row['training_records'][-1]
    if bad=='moment_weight':last['moment_weight']=.5
    elif bad=='moment_reference':last['moment_reference']='paired_clean'
    elif bad=='weighted_loss':last['losses']['weighted_moment_objective']+=.1
    elif bad=='selected_loss':last['losses']['moment_objective']=last['losses']['clean_moment_mse']
    elif bad=='descriptor_count':last['moment_descriptor_inputs']-=1
    elif bad=='extraG':row['training_complete']['extra_moment_G_inputs']=2
    elif bad=='student_count':row['training_complete']['student_total_IQ_examples']-=1
    elif bad=='teacher_count':last['teacher_IQ_examples']+=1
    elif bad=='identity_count':last['extra_clean_student_inputs']=0
    elif bad=='Ffree':last['F_inputs']=1
    elif bad=='targettruth':row['training_complete']['target_reads']=1
    elif bad=='missingview':row['target_prediction_complete']['views'].pop()
    elif bad=='missingrow':value['rows'].pop('moment_clean')
    elif bad=='live':value['actual_live_owned_PIDs']=[123]
    elif bad=='historical_reference':value['baselines'].pop('R22identity')
    elif bad=='epochs':row['training_records'].pop()
    elif bad=='nonfinite':last['losses']['clean_moment_mse']=float('nan')
    else:last['losses']['base_loss']+=1
    with pytest.raises(ValueError):m.completed_metrics(value)


def test_public_report_contains_both_targets_all_scenes_and_true_forward_cost(tmp_path,monkeypatch):
    m=tools();delivery=importlib.import_module('tools.complete_pamf_r46_moment_fidelity');d=m.d
    local=tmp_path/'local';workspace=tmp_path/'workspace';folder=local/'automation_reports/CV-SincNet'/d.RUN;folder.mkdir(parents=True)
    (folder/'report.md').write_text('状态：QUEUED。\n',encoding='utf-8');(folder/'experiment.json').write_text('{"execution":{}}',encoding='utf-8');(folder/'events.jsonl').write_text('',encoding='utf-8')
    output=local/'artifacts';output.mkdir();artifacts=dict(status='VERIFIED',models={a:dict(components=dict(coarse=1,action=2,prior=3),conv_linear_MACs_per_IQ=100,bytes=100) for a in d.ARMS})
    monkeypatch.setattr(delivery,'WORK',workspace);result=m.completed_metrics(fixture());delivery.report(local,output,result,artifacts,record_events=False)
    body=(folder/'report.md').read_text(encoding='utf-8')
    assert all('|'+a+'|'+v+'|' in body for a in d.ARMS for v in d.VIEWS)
    assert all(x in body for x in ('1260000','2520000','3780000','paired_clean','original_350521_restored_IQ'))
    assert len((folder/'row_metrics.csv').read_text(encoding='utf-8').splitlines())==393
    assert json.loads((folder/'results.json').read_text(encoding='utf-8'))['matched_teacher_vs_clean']
