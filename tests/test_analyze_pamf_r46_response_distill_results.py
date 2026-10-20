import copy,importlib,json
import pytest

def tools():
    try:return importlib.import_module('tools.analyze_pamf_r46_response_distill_results')
    except ModuleNotFoundError:pytest.fail('R26 complete-result exposure accounting is not implemented')

def fixture():
    m=tools();d=m.d
    old=importlib.import_module('tests.test_analyze_pamf_r46_complex_fidelity_results').fixture()
    old.update(run_id=d.RUN,method_commit=m.METHOD)
    old['complete'].update(source_rows=list(d.ARMS),target_rows=list(d.ARMS))
    old['baselines']['R22identity']=copy.deepcopy(old['baselines']['R18stage'])
    old['rows']=dict(zip(d.ARMS,old['rows'].values()))
    for arm,row in old['rows'].items():
        for record in row['training_records']:
            epoch=record['epoch'];n=985*epoch;count=16*n;record['losses']['clean_identity_nmse']=.05/epoch
            loss=record['losses'];base=loss['iq_nmse']+.05*loss['spectral_error']+d.teacher_weight(epoch)*loss['teacher_stage_nmse']+.02*loss['balanced_complex_nmse']+.25*loss['clean_identity_nmse']
            loss.update(base_loss=base,probe_objective=.2/epoch,loss=base+.01*.2/epoch)
            counts={p+':'+f'{e:.3f}':0 for p,e in d.PROBE_CYCLE}
            for i in range(n):
                p,e=d.PROBE_CYCLE[i%4];counts[p+':'+f'{e:.3f}']+=16
            record.update(teacher_IQ_examples=63000*epoch+2*count,extra_clean_student_inputs=63000*epoch,
                selected_probe_examples=count,selected_probe_L=count//10,selected_probe_U=count-count//10,
                probe_cell_examples=counts,extra_probe_teacher_inputs=2*count,extra_probe_student_inputs=2*count,
                student_total_IQ_examples=126000*epoch+2*count,probe_weight=.01,probe_objective=arm,probe_seed=d.PROBE_SEED)
        row['training_complete'].update(arm=arm,teacher_IQ_examples=1890400,extra_clean_student_inputs=1260000,
            clean_identity_weight=.25,selected_probe_examples=315200,selected_probe_L=31520,selected_probe_U=283680,
            probe_cell_examples=counts,extra_probe_teacher_inputs=630400,extra_probe_student_inputs=630400,
            student_total_IQ_examples=3150400,probe_weight=.01,probe_objective=arm,probe_seed=d.PROBE_SEED)
    return old


def test_probe_totals_weighted_objective_and_all7_results_are_preserved():
    m=tools();value=m.completed_metrics(fixture())
    assert value['accepted_rows']==['pair_aug_control']
    assert value['matched_response_vs_pair_aug']['views']['clean']['correct_delta']==-1
    for arm,row in value['rows'].items():
        assert row['training']['training_complete']['teacher_IQ_examples']==1890400
        assert row['training']['training_complete']['student_total_IQ_examples']==3150400
        assert row['training']['full_log_records']==20 and len(row['target']['views'])==7
        assert set(row['training']['losses'])=={'loss','base_loss','probe_objective','iq_nmse','spectral_error','teacher_stage_nmse','balanced_complex_nmse','clean_identity_nmse'}


@pytest.mark.parametrize('bad',['teacher_count','student_count','probe_count','cell_count','probe_role','probe_seed','probe_weight','probe_objective','weighted_loss','Ffree','targettruth','missingview','missingrow','live','historical_reference','matched_role'])
def test_incomplete_or_inconsistent_probe_or_formal_evidence_rejected(bad):
    m=tools();v=fixture();row=v['rows']['odd_response'];last=row['training_records'][-1]
    if bad=='teacher_count':last['teacher_IQ_examples']-=1
    elif bad=='student_count':row['training_complete']['student_total_IQ_examples']-=1
    elif bad=='probe_count':last['selected_probe_examples']-=1
    elif bad=='cell_count':last['probe_cell_examples']['random:0.005']-=16
    elif bad=='probe_role':last['selected_probe_U']-=1
    elif bad=='probe_seed':last['probe_seed']+=1
    elif bad=='probe_weight':row['training_complete']['probe_weight']=.02
    elif bad=='probe_objective':last['probe_objective']='pair_aug_control'
    elif bad=='weighted_loss':last['losses']['loss']+=1
    elif bad=='Ffree':last['F_inputs']=1
    elif bad=='targettruth':row['training_complete']['target_reads']=1
    elif bad=='missingview':row['target_prediction_complete']['views'].pop()
    elif bad=='missingrow':v['rows'].pop('pair_aug_control')
    elif bad=='live':v['actual_live_owned_PIDs']=[123]
    elif bad=='historical_reference':v['baselines'].pop('R22identity')
    else:
        for record in row['training_records']:
            record['selected_probe_L']+=1;record['selected_probe_U']-=1
        row['training_complete']['selected_probe_L']+=1;row['training_complete']['selected_probe_U']-=1
    with pytest.raises(ValueError):m.completed_metrics(v)


def test_public_report_includes_all_views_and_actual_probe_cost(tmp_path,monkeypatch):
    m=tools();delivery=importlib.import_module('tools.complete_pamf_r46_response_distill');d=m.d
    local=tmp_path/'local';workspace=tmp_path/'workspace';folder=local/'automation_reports/CV-SincNet'/d.RUN;folder.mkdir(parents=True)
    (folder/'report.md').write_text('状态：QUEUED。\n',encoding='utf-8');(folder/'experiment.json').write_text('{"execution":{}}',encoding='utf-8');(folder/'events.jsonl').write_text('',encoding='utf-8')
    output=local/'artifacts';output.mkdir();artifacts=dict(status='VERIFIED',models={a:dict(components=dict(coarse=1,action=2,prior=3),conv_linear_MACs_per_IQ=100,bytes=100) for a in d.ARMS})
    monkeypatch.setattr(delivery,'WORK',workspace);result=m.completed_metrics(fixture());delivery.report(local,output,result,artifacts,record_events=False)
    text=(folder/'report.md').read_text(encoding='utf-8')
    assert all('|'+a+'|'+v+'|' in text for a in d.ARMS for v in d.VIEWS)
    assert '1890400' in text and '3150400' in text and '630400' in text
    assert len((folder/'row_metrics.csv').read_text(encoding='utf-8').splitlines())==393
    assert json.loads((folder/'results.json').read_text(encoding='utf-8'))['matched_response_vs_pair_aug']
