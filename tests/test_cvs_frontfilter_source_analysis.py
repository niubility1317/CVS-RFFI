"""Complete synthetic source artifacts only; no dataset, target or SSH access."""
import copy
import csv
import io
import json
import statistics as st

import pytest

from experiments.cvs_frontfilter_identity import analyze, collect
from experiments.cvs_frontfilter_identity.dispatch import CANDIDATES, SEEDS
from experiments.cvs_frontfilter_identity.model import VARIANTS, filter_contract


def diagnostic(variant, packets=28):
    dynamic = variant == VARIANTS[1]
    front = dict(active=True, variant=variant, dynamic=dynamic,
        fixed_coefficient_bound_only=True, dynamic_map_invertible_claim=False,
        records=[dict(block='frontfilter', packets=packets, relative_input_change_mean=0.,
            relative_input_change_max=0., input_norm_ratio_min=1., input_norm_ratio_max=1.,
            input_norm_ratio_eligible_packets=packets, coefficient_l1_mean=0., coefficient_l1_max=0.,
            kernel_l1_mean=0., kernel_l1_max=0., basis_l1_max=.5, bound_active_fraction=0.,
            basis_bound_active_fraction=0., coefficient_packet_variance=0.,
            coefficient_l1_by_packet=[0.]*packets, kernel_l1_by_packet=[0.]*packets,
            coefficients=[[[0.]*4, [0.]*4] for _ in range(packets)],
            fixed_coefficient_delta_operator_bound_max=0., basis_gradient_norm=0.,
            context_gradient_norm=.01 if dynamic else None, exit_gradient_norm=.01)])
    return dict(alignment_strength=0., frontfilter=front,
        diagnostic_input_scopes=dict(coordinate_cfo_and_coherence='original_input_x',
            normalization_adaptive_input_neural_residual='actual_G_x(x), exactly once per diagnostic forward',
            core_weights_and_gradients='parameters and last CE gradients, no input-dependent estimate'),
        adaptive_input=dict(active=True, raw_parameters=[0.,0.], coefficients=[0.,0.],
            records=[dict(block='behavior.0.conv', packets=packets, complex_terms=12,
                actual_envelope_lag=4, actual_phase_lag=4, input_formula_max_abs_error=0.,
                input_abs_max=3., degree3_relative_input_change_mean=.1, degree5_relative_input_change_mean=.2)]),
        normalization=dict(blocks=6, records=[dict(block=p+'.'+str(i), eligible_packets=packets,
            relative_energy_fraction_max_error=1e-7, relative_energy_fraction_variance_mean=.02)
            for p in ('time','behavior') for i in range(3)]),
        neural_residual=dict(active=True, records=[dict(block=p+'.2.neural.0', packets=packets,
            relative_output_change_mean=.1, projection_norm=1.) for p in ('time','behavior')]))


def artifacts(epochs, steps):
    compact = [{k:v for k,v in e.items() if not isinstance(v,dict)} for e in epochs]
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream,fieldnames=list(compact[0]))
    writer.writeheader(); writer.writerows(compact)
    return (epochs, steps, stream.getvalue(), 'RESOLVED_CONFIG {}\n'+'\n'.join('EPOCH '+json.dumps(e) for e in epochs)), '\n'.join(json.dumps(r) for r in compact)


def make_logs(variant):
    npar = filter_contract(variant)['total_parameters']
    flags = dict(ce_weight=1.,augmentation_active=False,domain_backbone_active=False,pseudo_labels_active=False,
        extra_losses_active=False,frontfilter_active=True,gradient_used_parameters=npar,
        alignment_gradient_used_parameters=0,alignment_gradient_norm=None,cudnn_allow_tf32=False,
        mixture_gradient_used_parameters=2,mix_gradient3=.001,mix_gradient5=-.002,mix_gradient_norm=.00223606797749979,
        frontfilter_gradient_norm=.01,frontfilter_gradient_used_parameters=npar-220987)
    steps=[]; epochs=[]
    for epoch in range(1,201):
        for batch in range(50):
            steps.append(dict(epoch=epoch,step=(epoch-1)*50+batch+1,clean_ce=.1,total_loss=.1,
                learning_rate=.0002,gradient_norm=2.,source_samples=128 if batch<49 else 28,satellite_samples=0,
                mix_raw3_before=0.,mix_raw5_before=0.,mix_raw3_after=0.,mix_raw5_after=0.,
                mix_coefficient3_before=0.,mix_coefficient5_before=0.,mix_coefficient3_after=0.,mix_coefficient5_after=0.,
                alignment_strength_before=0.,alignment_strength_after=0.,**flags))
        epochs.append(dict(epoch=epoch,optimizer_steps=50,optimizer_steps_total=epoch*50,source_sample_exposure=6300,
            satellite_sample_exposure=0,clean_ce=.1,total_loss=.1,learning_rate=.0002,gradient_norm=2.,source_val_ce=.2,
            source_val_count=27000,source_val_accuracy=.9,source_val_rx_accuracy={r:.9 for r in ('1','3','4','6','8')},
            source_val_worst_rx=.9,mix_raw3=0.,mix_raw5=0.,mix_coefficient3=0.,mix_coefficient5=0.,
            alignment_strength=0.,frontfilter_diagnostics=diagnostic(variant),elapsed_seconds=1.,
            peak_cuda_allocated_bytes=1234,**flags))
    return epochs, steps, npar, variant


@pytest.fixture(scope='module',params=VARIANTS)
def logs(request):
    return make_logs(request.param)


def audit(logs):
    epochs,steps,npar,variant=logs
    args,compact=artifacts(epochs,steps)
    return collect.audit_logs(*args,npar,0.,compact,expected_phase_lag=4,
                              expected_frontfilter_contract=filter_contract(variant))


def test_complete_10000_step_and_200_epoch_reconciliation(logs):
    result=audit(logs)
    assert result['steps']==10000 and result['epochs']==200
    assert result['all_step_flags_match'] and result['step_epoch_csv_stdout_reconciled']


@pytest.mark.parametrize('kind', ['missing_step','extra_loss','precision','missing_precision','wrong_exposure',
    'wrong_average','wrong_parameter_count','gradient_nan','gradient_mean','inactive','wrong_variant',
    'wrong_scope','nonfinite_input','coefficients_shape','bound_fraction','old_gate_jump','source_ce'])
def test_complete_log_corruption_rejected(logs,kind):
    epochs,steps,npar,variant=copy.deepcopy(logs)
    diag=epochs[73]['frontfilter_diagnostics']; rec=diag['frontfilter']['records'][0]
    if kind=='missing_step':steps.pop(17)
    elif kind=='extra_loss':steps[500]['extra_losses_active']=True
    elif kind=='precision':steps[500]['cudnn_allow_tf32']=True
    elif kind=='missing_precision':steps[500].pop('cudnn_allow_tf32')
    elif kind=='wrong_exposure':steps[49]['source_samples']=128
    elif kind=='wrong_average':steps[500]['clean_ce']=steps[500]['total_loss']=.2
    elif kind=='wrong_parameter_count':steps[500]['frontfilter_gradient_used_parameters']=1
    elif kind=='gradient_nan':steps[500]['frontfilter_gradient_norm']=float('nan')
    elif kind=='gradient_mean':epochs[73]['frontfilter_gradient_norm']=.5
    elif kind=='inactive':diag['frontfilter']['active']=False
    elif kind=='wrong_variant':diag['frontfilter']['variant']='unregistered'
    elif kind=='wrong_scope':diag['diagnostic_input_scopes']['coordinate_cfo_and_coherence']='filtered_input'
    elif kind=='nonfinite_input':rec['relative_input_change_mean']=float('inf')
    elif kind=='coefficients_shape':rec['coefficients'][0][0].pop()
    elif kind=='bound_fraction':rec['bound_active_fraction']=1.1
    elif kind=='old_gate_jump':steps[500]['mix_raw3_before']=.1
    elif kind=='source_ce':epochs[73]['source_val_ce']=float('nan')
    with pytest.raises(ValueError):audit((epochs,steps,npar,variant))


@pytest.mark.parametrize('kind',['stdout','compact','runtime_error'])
def test_independent_text_and_compact_records_are_checked(logs,kind):
    epochs,steps,npar,variant=logs; args,compact=artifacts(epochs,steps)
    if kind=='stdout':args=(*args[:-1],args[-1].replace('"gradient_norm": 2.0','"gradient_norm": 3.0',1))
    elif kind=='compact':compact=compact.replace('"clean_ce": 0.1','"clean_ce": 0.2',1)
    else:args=(*args[:-1],args[-1]+'\nCUDA out of memory')
    with pytest.raises(ValueError):
        collect.audit_logs(*args,npar,0.,compact,expected_phase_lag=4,expected_frontfilter_contract=filter_contract(variant))


@pytest.mark.parametrize('variant',VARIANTS)
def test_public_scope_30_and_missing_gradient_are_distinguished(variant):
    d=diagnostic(variant,30)['frontfilter']
    for key in ('basis_gradient_norm','context_gradient_norm','exit_gradient_norm'):d['records'][0][key]=None
    assert collect.validate_frontfilter_diagnostics(d,30,filter_contract(variant))['packets']==30
    with pytest.raises(ValueError):collect.validate_frontfilter_diagnostics(d,28,filter_contract(variant))
    d['records'][0].update(input_norm_ratio_eligible_packets=0,input_norm_ratio_min=None,input_norm_ratio_max=None)
    assert collect.validate_frontfilter_diagnostics(d,30,filter_contract(variant))['input_norm_ratio_min'] is None


@pytest.fixture
def telemetry_rows():
    rows=[]
    for v,variant in enumerate(CANDIDATES):
        for s,seed in enumerate(sorted(SEEDS)):
            epochs=[]
            for epoch in range(1,201):
                item=dict(epoch=epoch,clean_ce=.8-.002*epoch+.01*s+.002*v,
                    source_val_ce=.9-.001*epoch+.03*s+.005*v,
                    source_val_accuracy=.6+.001*epoch+.005*v+.001*s,
                    source_val_worst_rx=.5+.001*epoch+.003*v+.002*s)
                if variant==VARIANTS[1]:
                    item['source_val_accuracy']+=.004*s-.008
                    item['source_val_worst_rx']-=.002*s
                if variant in VARIANTS:
                    item['frontfilter_gradient_norm']=0 if epoch==1 else .01*epoch+.1*s
                    item['frontfilter_diagnostics']=diagnostic(variant)
                    # A valid bounded measurement that changes at every epoch and seed.
                    r=item['frontfilter_diagnostics']['frontfilter']['records'][0]
                    r.update(coefficient_l1_mean=.8,coefficient_l1_max=.8,kernel_l1_mean=.8,kernel_l1_max=.8,
                        basis_l1_max=1.,coefficient_l1_by_packet=[.8]*28,kernel_l1_by_packet=[.8]*28,
                        coefficients=[[[.8,0.,0.,0.],[0.]*4] for _ in range(28)],
                        fixed_coefficient_delta_operator_bound_max=.2)
                    r['relative_input_change_mean']=.0005*epoch+.01*s
                    r['relative_input_change_max']=.0005*epoch+.01*s
                epochs.append(item)
            rows.append(dict(resolved=dict(variant=variant,model_seed=seed),epochs=epochs,log_audit=dict(steps=10000)))
    return rows


def test_ce_gap_uses_all_curves_and_paired_seed_dispersion(telemetry_rows):
    result=analyze.summarize_source_telemetry(telemetry_rows)
    assert len(result['source_learning_by_seed'])==16 and len(result['source_curve_summary'])==800
    for row in result['source_learning_summary']:
        originals=[r for r in telemetry_rows if r['resolved']['variant']==row['variant']]
        gaps=[r['epochs'][-1]['source_val_ce']-r['epochs'][-1]['clean_ce'] for r in originals]
        assert row['CE_gap_mean']==pytest.approx(st.mean(gaps))
        assert row['CE_gap_SD']==pytest.approx(st.stdev(gaps))
        assert row['train_CE_late_delta_mean']==pytest.approx(-.05)
        assert row['V_CE_late_delta_mean']==pytest.approx(-.025)
        assert row['CE_gap_late_delta_mean']==pytest.approx(.025)


def test_static_dynamic_pairing_preserves_e200_and_input(telemetry_rows):
    for row in telemetry_rows:row['epochs'][0].update(source_val_accuracy=.999,source_val_worst_rx=.999)
    telemetry_rows.reverse(); original=copy.deepcopy(telemetry_rows)
    result=analyze.summarize_source_telemetry(telemetry_rows)
    assert telemetry_rows==original
    for s,row in enumerate(result['frontfilter_pairwise_by_seed']):
        assert row['model_seed']==sorted(SEEDS)[s] and row['epoch']==200
        assert row['V_delta_pp']==pytest.approx(-.3+.4*s)
        assert row['worst_RX_delta_pp']==pytest.approx(.3-.2*s)
        assert row['score_delta_pp']==pytest.approx(.1*s)
    val=next(r for r in result['frontfilter_pairwise_summary'] if r['metric']=='V')
    assert val['paired_SD_pp']==pytest.approx(st.stdev([-.3,.1,.5,.9]))
    assert val['positive_seeds']==3
    assert result['scope']['epoch_reselection'] is False and result['scope']['target_results_read'] is False


def test_all_gradient_and_filter_epochs_including_null_context(telemetry_rows):
    result=analyze.summarize_source_telemetry(telemetry_rows)
    assert len(result['frontfilter_gradient_epochs'])==len(result['frontfilter_output_epochs'])==1600
    assert len(result['frontfilter_output_curve_summary'])==400
    assert len(result['frontfilter_last_epoch'])==8 and len(result['frontfilter_last_epoch_summary'])==2
    first=result['frontfilter_gradient_by_seed'][0]
    assert first['audited_steps']==10000 and first['zero_epoch_means']==1
    assert first['full_run_epoch_mean']==pytest.approx(st.mean([0]+[.01*i for i in range(2,201)]))
    assert first['E151_200_epoch_mean']==pytest.approx(1.755)
    final=result['frontfilter_last_epoch_summary'][0]
    assert final['relative_input_change_mean_mean']==pytest.approx(.115)
    assert final['context_gradient_norm_mean'] is None and final['context_gradient_norm_measured_seeds']==0
    new=next(r for r in telemetry_rows if r['resolved']['variant'] in VARIANTS)
    new['epochs'][7]['frontfilter_gradient_norm']+=10
    changed=analyze.summarize_source_telemetry(telemetry_rows)['frontfilter_gradient_by_seed'][0]
    assert changed['full_run_epoch_mean']-first['full_run_epoch_mean']==pytest.approx(.05)
    assert changed['E151_200_epoch_mean']==first['E151_200_epoch_mean']
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('kind',['missing_seed','duplicate_seed','missing_epoch','duplicate_epoch','nan_ce',
    'inf_gradient','missing_filter','wrong_batch','wrong_variant'])
def test_incomplete_or_nonfinite_telemetry_cannot_report_completion(telemetry_rows,kind):
    new=next(r for r in telemetry_rows if r['resolved']['variant'] in VARIANTS)
    d=new['epochs'][0]['frontfilter_diagnostics']['frontfilter']
    if kind=='missing_seed':telemetry_rows.pop()
    elif kind=='duplicate_seed':telemetry_rows[-1]=copy.deepcopy(telemetry_rows[-2])
    elif kind=='missing_epoch':telemetry_rows[0]['epochs'].pop(30)
    elif kind=='duplicate_epoch':telemetry_rows[0]['epochs'][30]['epoch']=30
    elif kind=='nan_ce':telemetry_rows[0]['epochs'][30]['clean_ce']=float('nan')
    elif kind=='inf_gradient':new['epochs'][7]['frontfilter_gradient_norm']=float('inf')
    elif kind=='missing_filter':d['records']=[]
    elif kind=='wrong_batch':d['records'][0]['packets']=128
    elif kind=='wrong_variant':d['variant']=VARIANTS[1]
    with pytest.raises(ValueError):analyze.summarize_source_telemetry(telemetry_rows)


def test_report_states_frontfilter_measurement_boundaries(telemetry_rows):
    text=analyze.telemetry_report(analyze.summarize_source_telemetry(telemetry_rows))
    for phrase in ('不是误差率差','50个batch均值的等权平均','27000个验证样本','不重新选择epoch',
                   '28个样本不能代表全源分布','不是参数完全匹配','固定系数','原输入x','不能用来评价滤波后的补偿效果',
                   'static没有context','未计算step极值','frontfilter_dynamic−frontfilter_static'):
        assert phrase in text
    assert all(phrase not in text for phrase in ('attention','readout','crosspath','关系分支','投影范数'))


def test_analyze_validates_before_any_report_write(tmp_path,monkeypatch):
    evidence=tmp_path/'automation_reports/CV-SincNet'/analyze.RUN/'evidence';evidence.mkdir(parents=True)
    source=evidence/'source_research_complete.json';source.write_text('{}',encoding='utf-8')
    def reject(_):raise ValueError('synthetic incomplete source evidence')
    monkeypatch.setattr(analyze,'validate_completed',reject)
    with pytest.raises(ValueError,match='incomplete source'):analyze.analyze(tmp_path)
    assert list(evidence.iterdir())==[source] and not (evidence.parent/'report.md').exists()


def source_diagnostics():
    groups=[]; front=[]
    for tx in range(6):
        for rx in (1,3,4,6,8):
            for day in (1,2,3):
                base=dict(tx=tx,receiver=rx,day=day,count=300)
                groups.append(dict(base,accuracy=.9,alignment_strength=0.,residual_valid_fraction=1.,
                    residual_formula_eligible_count=300,residual_formula_max_error_hz=0.,
                    relative_cfo_hz_mean=0.,relative_cfo_hz_min=0.,relative_cfo_hz_max=0.,coherence_mean=.5,
                    fallback_fraction=0.,residual_estimated_cfo_hz_mean=0.,nominal_residual_cfo_hz_mean=0.,
                    unit_embedding_mean=[0.]*160,unit_embedding_trace_variance=1.))
                front.append(dict(base,relative_input_change_mean=0.,relative_input_change_max=0.,
                    input_norm_ratio_min=1.,input_norm_ratio_max=1.,input_norm_ratio_eligible_count=300,
                    coefficient_l1_mean=0.,coefficient_l1_max=0.,kernel_l1_mean=0.,kernel_l1_max=0.,
                    coefficient_mean=[0.]*8,coefficient_trace_variance=0.))
    return dict(role='V',count=27000,groups=groups,frontfilter_groups=front,target_access=False,
        used_for_training=False,used_for_selection=False,
        frontfilter_scope='All 27000 source V packets; actual single frontend forward; observational only',
        coordinate_scope='Original input x; inherited nominal CFO diagnostic does not measure frontend compensation')


@pytest.fixture(scope='module')
def completed():
    from experiments.cvs_frontfilter_identity.dispatch import select_source_candidate
    from experiments.cvs_reference_identity.physics import TX_ROWS,RX_ROWS
    controls=[dict(variant=v,seed=s,accuracy=.9,worst_rx=.9,parameters=220987,macs=100)
              for v in CANDIDATES[:2] for s in sorted(SEEDS)]
    rows=[]
    for variant in VARIANTS:
        epochs,_,npar,_=make_logs(variant)
        contract=filter_contract(variant)
        for seed in sorted(SEEDS):
            record=dict(variant=variant,seed=seed,accuracy=.9,worst_rx=.9,parameters=npar,macs=200)
            phys=diagnostic(variant,30)
            phys.update(status='FROZEN_SYNTHETIC_DIAGNOSTIC_COMPLETE',synthetic_only=True,
                target_access=False,training_augmentation=False,formal_data_access=False,model_updated=False,
                hardware_parameter_recovery=False,whole_affine_phase_invariant_claim=False,
                arbitrary_channel_rx_invariant_claim=False,dynamic_map_invertible_claim=False,exact_wisig_equalizer=False,
                phase_audit=[dict(theta_radians=t,received_cfo_hz=0.,whole_logit_max_abs_error=0.) for t in (.37,-1.2,2.9)],
                phase_numeric_tolerance=1e-3,phase_tolerance_scope='Report only; not ranking or stopping',phase_tolerance_pass=True,
                records=[dict(tx=t['name'],rx=r['name'],unit_embedding_distance_same_tx_identity_rx=.1,
                    source_classifier_logits=[0.]*6) for t in TX_ROWS for r in RX_ROWS])
            initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],
                target_access=False,target_contact=False,model_seed=seed,physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
            final={k:v for k,v in epochs[-1].items() if k.startswith('source_val_')}
            rows.append(dict(row_id=variant+'-s'+str(seed),resolved=dict(variant=variant,model_seed=seed,
                frontfilter=contract,frontfilter_actual=contract,frontfilter_active=True,
                total_parameters=npar,trainable_parameters=npar),
                epochs=copy.deepcopy(epochs),source_record=record,initialization=initial,
                log_audit=dict(steps=10000,epochs=200,csv_epochs=200,roles_match=True,full_source_record_reverified=True,
                    all_step_flags_match=True,all_step_metrics_finite=True,step_epoch_csv_stdout_reconciled=True,errors=[]),
                completion=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,
                    target_evaluated=False,final_source_metrics=final),
                profile=dict(total_parameters=npar,trainable_parameters=npar,gradient_used_parameters=npar,
                    conv_linear_macs_per_sample=200,all_identity_paths_shared_energy_normalization=False,
                    base_complex_paths_shared_energy_normalization=True,filter_contract=contract,
                    resident_state_bytes=npar*4,inference_batch1_ms=1.,training_batch128_ms=10.,
                    hardware='synthetic CPU fixture',torch_version='synthetic'),
                source_diagnostics=source_diagnostics(),physical_diagnostics=phys))
    return dict(ready=True,rows=rows,source_controls=controls,
                source_selection=select_source_candidate(controls+[r['source_record'] for r in rows]))


def test_complete_16_record_freeze_and_phase_tolerance_is_not_selection_gate(completed):
    data=copy.deepcopy(completed)
    expected=collect.validate_completed(data)
    assert expected['new_candidate_selected'] is False
    data['rows'][0]['physical_diagnostics']['phase_audit'][0]['whole_logit_max_abs_error']=.01
    data['rows'][0]['physical_diagnostics']['phase_tolerance_pass']=False
    assert collect.validate_completed(data)==expected


@pytest.mark.parametrize('kind',['missing_model','missing_control','inheritance','target_contact','target_evaluation',
    'missing_epoch','steps','parameters','actual_contract','ranking_record','stored_selection','physics_target',
    'physics_nonfinite','physics_missing_pair','geometry_missing_cell'])
def test_completed_artifact_protocol_corruption_rejected(completed,kind):
    data=copy.deepcopy(completed);row=data['rows'][0]
    if kind=='missing_model':data['rows'].pop()
    elif kind=='missing_control':data['source_controls'].pop()
    elif kind=='inheritance':row['initialization']['ancestors']=['unapproved.pt']
    elif kind=='target_contact':row['initialization']['target_contact']=True
    elif kind=='target_evaluation':row['completion']['target_evaluated']=True
    elif kind=='missing_epoch':row['epochs'].pop(5)
    elif kind=='steps':row['completion']['steps']=9999
    elif kind=='parameters':row['profile']['trainable_parameters']-=1
    elif kind=='actual_contract':row['resolved']['frontfilter_actual']=filter_contract(VARIANTS[1])
    elif kind=='ranking_record':row['source_record']['accuracy']=.99
    elif kind=='stored_selection':data['source_selection']['selected_variant']=VARIANTS[0]
    elif kind=='physics_target':row['physical_diagnostics']['target_access']=True
    elif kind=='physics_nonfinite':row['physical_diagnostics']['records'][0]['source_classifier_logits'][0]=float('nan')
    elif kind=='physics_missing_pair':row['physical_diagnostics']['records'][-1]=copy.deepcopy(row['physical_diagnostics']['records'][0])
    elif kind=='geometry_missing_cell':row['source_diagnostics']['groups'].pop()
    with pytest.raises(ValueError):collect.validate_completed(data)


@pytest.mark.parametrize('kind',['missing_cell','duplicate_cell','count','nonfinite','coefficient_dimension',
    'static_variance','ratio_bound','missing_ratio','kernel_bound','coordinate_scope'])
def test_full_v_filter_cells_reject_invalid_source_evidence(kind):
    d=source_diagnostics();r=d['frontfilter_groups'][0]
    if kind=='missing_cell':d['frontfilter_groups'].pop()
    elif kind=='duplicate_cell':d['frontfilter_groups'][-1]=copy.deepcopy(r)
    elif kind=='count':r['count']=299
    elif kind=='nonfinite':r['relative_input_change_mean']=float('nan')
    elif kind=='coefficient_dimension':r['coefficient_mean'].pop()
    elif kind=='static_variance':r['coefficient_trace_variance']=.01
    elif kind=='ratio_bound':r['input_norm_ratio_max']=1.4
    elif kind=='missing_ratio':r['input_norm_ratio_min']=None
    elif kind=='kernel_bound':r['kernel_l1_max']=1.1
    elif kind=='coordinate_scope':d['coordinate_scope']='filtered CFO'
    with pytest.raises(ValueError):collect.validate_frontfilter_groups(d,filter_contract(VARIANTS[0]))


def test_full_v_aggregation_reconstructs_variance_and_keeps_zero_input_null(completed):
    rows=copy.deepcopy(completed['rows'])
    for row in rows:
        for i,g in enumerate(row['source_diagnostics']['frontfilter_groups']):
            g.update(input_norm_ratio_min=None,input_norm_ratio_max=None,input_norm_ratio_eligible_count=0)
            if row['resolved']['variant']==VARIANTS[1]:
                g.update(coefficient_mean=[.2 if i%2 else -.2]+[0.]*7,coefficient_trace_variance=.01,
                    coefficient_l1_mean=.3,coefficient_l1_max=.4)
    result=analyze.summarize_source_filter_cells(rows)
    assert len(result['source_frontfilter_cells'])==720
    for row in result['source_frontfilter_by_seed']:
        assert row['count']==27000 and row['cells']==90
        assert row['input_norm_ratio_min'] is None and row['input_norm_ratio_eligible_count']==0
        expected=.05 if row['variant']==VARIANTS[1] else 0.
        assert row['coefficient_trace_variance']==pytest.approx(expected)
    assert result['source_frontfilter_summary'][1]['coefficient_trace_variance_mean']==pytest.approx(.05)


def test_static_filter_cannot_change_coefficients_between_full_v_cells():
    d=source_diagnostics();r=d['frontfilter_groups'][0]
    r.update(coefficient_mean=[.1]+[0.]*7,coefficient_l1_mean=.1,coefficient_l1_max=.1)
    with pytest.raises(ValueError,match='between source cells'):
        collect.validate_frontfilter_groups(d,filter_contract(VARIANTS[0]))


def test_collect_and_analyze_synthetic_artifact_roundtrip(tmp_path,monkeypatch,completed):
    import ast
    from experiments.cvs_frontfilter_identity import publish
    scripts=[]
    def fake_ssh(wrapper):
        compile(wrapper,'collector-wrapper','exec')
        call=ast.parse(wrapper).body[1].value
        command=ast.literal_eval(call.args[0]);compile(command[2],'collector-payload','exec')
        scripts.append(command[2])
        return json.dumps(completed).encode('utf-8')
    monkeypatch.setattr(publish,'ssh',fake_ssh)
    assert collect.collect(tmp_path,tmp_path/'output') is True
    assert len(scripts)==1 and 'FRONTFILTER_AUDITOR' not in scripts[0]
    for variant,run in ((analyze.CONTROL,analyze.CONTROL_RUN),(analyze.ANCHOR_CONTROL,analyze.ANCHOR_CONTROL_RUN)):
        controls=[]
        for source in completed['rows'][:4]:
            row=copy.deepcopy(source);row['resolved']['variant']=variant
            for e in row['epochs']:e.pop('frontfilter_diagnostics');e.pop('frontfilter_gradient_norm')
            controls.append(row)
        p=tmp_path/'automation_reports/CV-SincNet'/run/'evidence';p.mkdir(parents=True)
        (p/'source_research_complete.json').write_text(json.dumps(dict(rows=controls)),encoding='utf-8')
    analyze.analyze(tmp_path)
    evidence=tmp_path/'automation_reports/CV-SincNet'/analyze.RUN/'evidence'
    validation=json.loads((evidence/'source_analysis_validation.json').read_text(encoding='utf-8'))
    assert validation['source_epochs']==validation['control_epochs']==1600
    assert validation['source_frontfilter_cells']==720 and validation['frontfilter_output_records']==1600
    with (evidence/'source_curves.csv').open(encoding='utf-8',newline='') as f:assert len(list(csv.DictReader(f)))==3200
    with (evidence/'source_cells.csv').open(encoding='utf-8',newline='') as f:assert len(list(csv.DictReader(f)))==1440
    assert (evidence/'source_curves.png').is_file() and (evidence/'source_curves.pdf').is_file()
