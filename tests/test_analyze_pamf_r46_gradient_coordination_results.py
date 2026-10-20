import copy,importlib,json
import pytest


def tools():
    return importlib.import_module('tools.analyze_pamf_r46_gradient_coordination_results')


def fixture():
    m=tools();d=m.d
    old=importlib.import_module('tests.test_analyze_pamf_r46_complex_fidelity_results').fixture()
    old.update(run_id=d.RUN,method_commit=m.METHOD)
    old['complete'].update(source_rows=list(d.ARMS),target_rows=list(d.ARMS))
    old['baselines']['R22identity']=copy.deepcopy(old['baselines']['R18stage'])
    old['baselines']['R30teacher']=copy.deepcopy(old['baselines']['R18stage'])
    old['rows']=dict(zip(d.ARMS,old['rows'].values()))
    for arm,row in old['rows'].items():
        for record in row['training_records']:
            epoch=record['epoch'];loss=record['losses'];loss['clean_identity_nmse']=.05/epoch
            base=loss['iq_nmse']+.05*loss['spectral_error']+d.teacher_weight(epoch)*loss['teacher_stage_nmse']+.02*loss['balanced_complex_nmse']+.25*loss['clean_identity_nmse']
            cm,tm=.003/epoch,.001/epoch;selected=cm if arm=='moment_clean' else tm
            loss.update(base_loss=base,clean_moment_mse=cm,teacher_moment_mse=tm,moment_objective=selected,weighted_moment_objective=selected,loss=base+selected)
            record.update(gradient_policy=d.GRADIENT_POLICY[arm],gradient_evaluations=1970*epoch,gradient_statistics={g:dict(updates=985,conflict_updates=10,projected_updates=0 if d.GRADIENT_POLICY[arm]=='sum' else 10,means=dict(base_norm=1.,moment_norm=.1,raw_norm=1.1,applied_norm=1.1,raw_base_dot=1.,raw_moment_dot=.1,applied_base_dot=1.,applied_moment_dot=.1),base_cos_valid=985,base_cos_mean=.5,moment_cos_valid=0,moment_cos_mean=None) for g in d.GRADIENT_GROUPS})
            record.update(extra_clean_student_inputs=63000*epoch,moment_reference=d.MOMENT_REFERENCE[arm],moment_weight=1.,extra_moment_G_inputs=0,moment_descriptor_inputs=189000*epoch)
        row['training_complete'].update(gradient_policy=d.GRADIENT_POLICY[arm],gradient_evaluations=39400,gradient_groups=list(d.GRADIENT_GROUPS),gradient_norm_floor=d.NORM_FLOOR)
        row['training_complete'].update(arm=arm,extra_clean_student_inputs=1260000,clean_identity_weight=.25,student_total_IQ_examples=2520000,
            moment_weight=1.,moment_reference=d.MOMENT_REFERENCE[arm],extra_moment_G_inputs=0,moment_descriptor_inputs=3780000)
    return old


def test_both_moment_references_full_losses_counts_and_formal7():
    m=tools();result=m.completed_metrics(fixture())
    assert result['accepted_rows']==['moment_teacher_grad_sum']
    assert result['matched_symproj_vs_sum']['views']['clean']['correct_delta']==-1
    for row in result['rows'].values():
        assert row['training']['full_log_records']==20 and len(row['target']['views'])==7
        assert len(row['training']['losses'])==11
        assert row['training']['training_complete']['student_total_IQ_examples']==2520000
    assert result['matched_forward_exposure']['extra_moment_G_inputs']==0


@pytest.mark.parametrize('bad',['moment_weight','moment_reference','weighted_loss','selected_loss','descriptor_count','extraG',
    'student_count','teacher_count','identity_count','Ffree','targettruth','missingview','missingrow','live','historical_reference','epochs','nonfinite','base_loss'])
def test_inconsistent_moment_or_formal_evidence_rejected(bad):
    m=tools();value=fixture();row=value['rows']['moment_teacher_symproj'];last=row['training_records'][-1]
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
    elif bad=='missingrow':value['rows'].pop('moment_teacher_grad_sum')
    elif bad=='live':value['actual_live_owned_PIDs']=[123]
    elif bad=='historical_reference':value['baselines'].pop('R22identity')
    elif bad=='epochs':row['training_records'].pop()
    elif bad=='nonfinite':last['losses']['clean_moment_mse']=float('nan')
    else:last['losses']['base_loss']+=1
    with pytest.raises(ValueError):m.completed_metrics(value)


def test_public_report_contains_both_targets_all_scenes_and_true_forward_cost(tmp_path,monkeypatch):
    m=tools();delivery=importlib.import_module('tools.complete_pamf_r46_gradient_coordination');d=m.d
    local=tmp_path/'local';workspace=tmp_path/'workspace';folder=local/'automation_reports/CV-SincNet'/d.RUN;folder.mkdir(parents=True)
    (folder/'report.md').write_text('状态：QUEUED。\n',encoding='utf-8');(folder/'experiment.json').write_text('{"execution":{}}',encoding='utf-8');(folder/'events.jsonl').write_text('',encoding='utf-8')
    output=local/'artifacts';output.mkdir();artifacts=dict(status='VERIFIED',models={a:dict(components=dict(coarse=1,action=2,prior=3),conv_linear_MACs_per_IQ=100,bytes=100) for a in d.ARMS})
    monkeypatch.setattr(delivery,'WORK',workspace);result=m.completed_metrics(fixture());delivery.report(local,output,result,artifacts,record_events=False)
    body=(folder/'report.md').read_text(encoding='utf-8')
    assert all('|'+a+'|'+v+'|' in body for a in d.ARMS for v in d.VIEWS)
    assert all(x in body for x in ('1260000','2520000','3780000','39400','original_350521_restored_IQ','moment_teacher_symproj','prior'))
    assert len((folder/'row_metrics.csv').read_text(encoding='utf-8').splitlines())==393
    assert json.loads((folder/'results.json').read_text(encoding='utf-8'))['matched_symproj_vs_sum']


@pytest.mark.parametrize('bad',['policy','gradientcount','missinggroup','updates','projection','nonfinite','negative_norm','nullcos','invalidcos','finalcount','finalgroups','history'])
def test_invalid_gradient_telemetry_or_full_reference_rejected(bad):
    m=tools();value=fixture();row=value['rows'][m.d.ARMS[1]];last=row['training_records'][-1];g=last['gradient_statistics']['coarse']
    if bad=='policy':last['gradient_policy']='sum'
    elif bad=='gradientcount':last['gradient_evaluations']-=2
    elif bad=='missinggroup':last['gradient_statistics'].pop('prior')
    elif bad=='updates':g['updates']=984
    elif bad=='projection':g['projected_updates']=0
    elif bad=='nonfinite':g['means']['applied_base_dot']=float('nan')
    elif bad=='negative_norm':g['means']['raw_norm']=-1
    elif bad=='nullcos':g['moment_cos_mean']=0
    elif bad=='invalidcos':g['base_cos_mean']=2
    elif bad=='finalcount':row['training_complete']['gradient_evaluations']=19700
    elif bad=='finalgroups':row['training_complete']['gradient_groups'].pop()
    else:value['baselines'].pop('R30teacher')
    with pytest.raises(ValueError):m.completed_metrics(value)


def test_whole_epoch_gradient_counts_and_null_cosines_remain_explicit():
    m=tools();result=m.completed_metrics(fixture())
    for arm,row in result['rows'].items():
        for group,v in row['training']['gradient_statistics'].items():
            assert v['updates']==19700 and v['conflict_updates']==200
            assert v['projected_updates']==(0 if arm==m.d.ARMS[0] else 200)
            assert v['base_cos_valid']==19700 and v['base_cos_mean']==.5
            assert v['moment_cos_valid']==0 and v['moment_cos_mean'] is None
        assert row['target_vs_R30teacher']
    assert result['matched_forward_exposure']['gradient_evaluations_per_row']==39400
