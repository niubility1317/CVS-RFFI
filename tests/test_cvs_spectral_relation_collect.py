"""Complete synthetic E200 and V artifacts; no source data, target or SSH access."""
import base64
import copy
import csv
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.cvs_spectral_relation_identity import collect
from experiments.cvs_spectral_relation_identity.dispatch import control_rows, select_source_candidate, SEEDS
from experiments.cvs_spectral_relation_identity.model import VARIANTS, relation_contract, build
from experiments.cvs_spectral_relation_identity.prepare import PROJECT, RUN
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY


def diagnostic(variant, packets=28):
    branch=dict(active=True,variant=variant,scope='Synthetic all provided packets',records=[dict(
        block='frequency.spectral_relation',packets=packets,frames=7,frequency_bins=64,output_dimension=160,
        per_frequency=variant==VARIANTS[1],relative_output_change_mean=.05,relative_output_change_max=.05,
        floor_fraction=0.,floor_fraction_by_frequency=[0.]*64,relation_norm_mean=1.,relation_norm_max=1.,
        relation_trace_mean=1.,relation_trace_max=1.,relation_norm_by_frequency=[1.]*64,
        relation_trace_by_frequency=[1.]*64,mix_norm=1.,projection_norm=.1,
        mix_gradient_norm=.001,projection_gradient_norm=.002,encoder_gradient_norm=.003)])
    return dict(alignment_strength=0.,spectral_relation=branch,
        adaptive_input=dict(active=True,raw_parameters=[0.,0.],coefficients=[0.,0.],
            records=[dict(block='behavior.0.conv',packets=packets,complex_terms=12,actual_envelope_lag=4,
                actual_phase_lag=4,input_formula_max_abs_error=0.,input_abs_max=3.,
                degree3_relative_input_change_mean=.1,degree5_relative_input_change_mean=.2)]),
        normalization=dict(blocks=6,records=[dict(block=p+'.'+str(i),eligible_packets=packets,
            relative_energy_fraction_max_error=1e-7,relative_energy_fraction_variance_mean=.02)
            for p in ('time','behavior') for i in range(3)]),
        neural_residual=dict(active=True,records=[dict(block=p+'.2.neural.0',packets=packets,
            relative_output_change_mean=.1,projection_norm=1.) for p in ('time','behavior')]))


def make_logs(variant):
    flags=dict(ce_weight=1.,augmentation_active=False,domain_backbone_active=False,pseudo_labels_active=False,
        extra_losses_active=False,spectral_relation_active=True,gradient_used_parameters=247731,
        alignment_gradient_used_parameters=0,alignment_gradient_norm=None,cudnn_allow_tf32=False,
        mixture_gradient_used_parameters=2,mix_gradient3=.001,mix_gradient5=-.002,mix_gradient_norm=math.sqrt(.000005),
        spectral_relation_gradient_norm=.01,spectral_relation_gradient_used_parameters=26744)
    steps=[];epochs=[]
    for epoch in range(1,201):
        lr=1e-6+(.0002-1e-6)*(1+math.cos(math.pi*(epoch-1)/200))/2
        for batch in range(50):
            steps.append(dict(epoch=epoch,step=(epoch-1)*50+batch+1,clean_ce=.1,total_loss=.1,
                learning_rate=lr,gradient_norm=2.,source_samples=128 if batch<49 else 28,satellite_samples=0,
                mix_raw3_before=0.,mix_raw5_before=0.,mix_raw3_after=0.,mix_raw5_after=0.,
                mix_coefficient3_before=0.,mix_coefficient5_before=0.,mix_coefficient3_after=0.,mix_coefficient5_after=0.,
                alignment_strength_before=0.,alignment_strength_after=0.,**flags))
        diag=diagnostic(variant)
        epochs.append(dict(epoch=epoch,optimizer_steps=50,optimizer_steps_total=epoch*50,source_sample_exposure=6300,
            satellite_sample_exposure=0,clean_ce=.1,total_loss=.1,learning_rate=lr,gradient_norm=2.,source_val_ce=.2,
            source_val_count=27000,source_val_accuracy=.9,source_val_rx_accuracy={r:.9 for r in ('1','3','4','6','8')},
            source_val_worst_rx=.9,mix_raw3=0.,mix_raw5=0.,mix_coefficient3=0.,mix_coefficient5=0.,
            alignment_strength=0.,spectral_relation_diagnostics=diag,elapsed_seconds=1.,peak_cuda_allocated_bytes=1234,
            energy_diagnostic_scope='last source batch of epoch (28 packets); not complete source V',
            **{'spectral_relation_'+key:diag['spectral_relation']['records'][0][key] for key in collect.COMPACT_RELATION_KEYS},**flags))
    return epochs,steps


def artifacts(epochs,steps,resolved=None,physics=None):
    compact=[{k:v for k,v in epoch.items() if not isinstance(v,dict)} for epoch in epochs]
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(compact[0]))
    writer.writeheader();writer.writerows(compact)
    stdout='RESOLVED_CONFIG '+json.dumps(resolved or {})+'\n'+'\n'.join('EPOCH '+json.dumps(e) for e in epochs)
    if physics is not None:stdout+='\nFROZEN_PHYSICS '+json.dumps(physics)+'\n'
    return stream.getvalue(),stdout,'\n'.join(json.dumps(row) for row in compact)


def audit(variant,epochs,steps):
    csv_text,stdout,compact=artifacts(epochs,steps)
    return collect.audit_logs(epochs,steps,csv_text,stdout,247731,0.,compact,4,relation_contract(variant))


@pytest.fixture(scope='module')
def logs():
    return {variant:make_logs(variant) for variant in VARIANTS}


@pytest.mark.parametrize('variant',VARIANTS)
def test_complete_10000_step_audit(variant,logs):
    result=audit(variant,*logs[variant])
    assert result['steps']==10000 and result['step_epoch_csv_stdout_reconciled']


@pytest.mark.parametrize('kind',['missing_step','extra_loss','precision','exposure','gradient_parameters',
    'gradient_nan','gradient_mean','gate_jump','schedule','source_ce','inactive','scope','floor_mean',
    'trace','frequency_count','compact_scalar','missing_gradient'])
def test_complete_log_corruptions_rejected(logs,kind):
    variant=VARIANTS[0];epochs,steps=copy.deepcopy(logs[variant]);row=epochs[73]
    rec=row['spectral_relation_diagnostics']['spectral_relation']['records'][0]
    if kind=='missing_step':steps.pop(17)
    elif kind=='extra_loss':steps[500]['extra_losses_active']=True
    elif kind=='precision':steps[500].pop('cudnn_allow_tf32')
    elif kind=='exposure':steps[49]['source_samples']=128
    elif kind=='gradient_parameters':steps[500]['spectral_relation_gradient_used_parameters']=1
    elif kind=='gradient_nan':steps[500]['spectral_relation_gradient_norm']=float('nan')
    elif kind=='gradient_mean':row['spectral_relation_gradient_norm']=.5
    elif kind=='gate_jump':steps[500]['mix_raw3_before']=.1
    elif kind=='schedule':
        row['learning_rate']=.0002
        for s in steps[73*50:74*50]:s['learning_rate']=.0002
    elif kind=='source_ce':row['source_val_ce']=float('nan')
    elif kind=='inactive':row['spectral_relation_diagnostics']['spectral_relation']['active']=False
    elif kind=='scope':row['energy_diagnostic_scope']='Complete source V'
    elif kind=='floor_mean':rec['floor_fraction']=.5
    elif kind=='trace':rec['relation_trace_max']=2.
    elif kind=='frequency_count':rec['relation_norm_by_frequency'].pop()
    elif kind=='compact_scalar':row['spectral_relation_projection_norm']=9.
    else:rec['mix_gradient_norm']=None
    with pytest.raises(ValueError):audit(variant,epochs,steps)


@pytest.mark.parametrize('kind',['stdout','compact','csv','runtime_error'])
def test_full_output_formats_are_independently_reconciled(logs,kind):
    epochs,steps=logs[VARIANTS[0]];csv_text,stdout,compact=artifacts(epochs,steps)
    if kind=='stdout':stdout=stdout.replace('"gradient_norm": 2.0','"gradient_norm": 3.0',1)
    elif kind=='compact':compact=compact.replace('"clean_ce": 0.1','"clean_ce": 0.2',1)
    elif kind=='csv':csv_text=csv_text.replace('False','True',1)
    else:stdout+='\nCUDA out of memory'
    with pytest.raises(ValueError):
        collect.audit_logs(epochs,steps,csv_text,stdout,247731,0.,compact,4,relation_contract(VARIANTS[0]))


def scalar_fixture():
    cells=[(t,r,d) for t in range(6) for r in (1,3,4,6,8) for d in (1,2,3)]
    entries=[(*cell,j) for cell in cells for j in range(300)]
    arrays=dict(ids=np.asarray(['/'.join(map(str,row)) for row in entries]),
        tx=np.asarray([row[0] for row in entries]),receiver=np.asarray([row[1] for row in entries]),
        day=np.asarray([row[2] for row in entries]))
    constants=dict(branch_output_norm=.1,base_frequency_norm=2.,relative_output_change=.05,
        relation_frobenius_mean=1.,relation_frobenius_max=1.,relation_trace_mean=1.,relation_trace_max=1.,
        floor_fraction=0.,projected_energy_mean=1.,projected_energy_min=1.)
    arrays.update({key:np.full(27000,value,np.float64) for key,value in constants.items()})
    groups=[];geometry=[]
    for tx,rx,day in cells:
        row=dict(tx=tx,receiver=rx,day=day,count=300)
        row.update({key+'_'+suffix:value for key,value in constants.items() for suffix in ('mean','min','max')})
        groups.append(row)
        geometry.append(dict(tx=tx,receiver=rx,day=day,count=300,accuracy=.9,unit_embedding_mean=[0.]*160,
            unit_embedding_trace_variance=1.,relative_cfo_hz_mean=0.,relative_cfo_hz_min=0.,relative_cfo_hz_max=0.,
            alignment_strength=0.,residual_estimated_cfo_hz_mean=0.,nominal_residual_cfo_hz_mean=0.,
            residual_formula_max_error_hz=0.,residual_formula_eligible_count=300,residual_valid_fraction=1.,
            coherence_mean=1.,fallback_fraction=0.))
    diag=dict(role='V',count=27000,groups=geometry,spectral_relation_groups=groups,
        spectral_relation_scalars=dict(schema='spectral_relation_source_scalars_v1',path='',count=27000,
            fields=list(arrays),dtype='float64 storage of FP32 forward measurements',scope='complete source V',
            base_frequency_norm_scope='before relation',projected_energy_scope='before floor',
            floor_fraction_scope='absolute and relative floor',target_access=False,used_for_training=False,used_for_selection=False),
        mean_between_tx_centroid_squared_distance=0.,mean_within_tx_rx_centroid_squared_distance=0.,
        target_access=False,used_for_training=False,used_for_selection=False)
    return arrays,diag


def encoded_arrays(arrays):
    buf=io.BytesIO();np.savez_compressed(buf,**arrays)
    return base64.b64encode(buf.getvalue()).decode('ascii')


@pytest.fixture(scope='module')
def scalars():return scalar_fixture()


def test_independent_recount_of_all_27000_packets_and_90_cells(scalars):
    arrays,diag=scalars
    result,metadata=collect.recount_source_scalars(arrays,arrays['ids'].tolist(),diag,relation_contract(VARIANTS[0]))
    assert result['packets']==27000 and result['cells']==90 and len(metadata)==27000
    assert result['all_cell_mean_min_max_recomputed']


@pytest.mark.parametrize('kind',['ids','duplicate','metadata','shape','dtype','nan','negative','floor',
    'relative_norm','trace','bound','summary_mean','summary_min','cell','schema','target'])
def test_bad_packet_or_cell_evidence_is_rejected(scalars,kind):
    arrays,diag=copy.deepcopy(scalars);ids=scalars[0]['ids'].tolist()
    if kind=='ids':arrays['ids'][0]='other'
    elif kind=='duplicate':arrays['ids'][0]=arrays['ids'][1]
    elif kind=='metadata':arrays['receiver'][0]=0
    elif kind=='shape':arrays['branch_output_norm']=np.zeros((27000,160))
    elif kind=='dtype':arrays['floor_fraction']=arrays['floor_fraction'].astype(np.float32)
    elif kind=='nan':arrays['branch_output_norm'][0]=np.nan
    elif kind=='negative':arrays['branch_output_norm'][0]=-1
    elif kind=='floor':arrays['floor_fraction'][0]=2.
    elif kind=='relative_norm':arrays['relative_output_change'][0]=.3
    elif kind=='trace':arrays['relation_trace_mean'][0]=2.
    elif kind=='bound':arrays['relation_trace_max'][0]=65.
    elif kind=='summary_mean':diag['spectral_relation_groups'][0]['branch_output_norm_mean']=.2
    elif kind=='summary_min':diag['spectral_relation_groups'][0]['floor_fraction_min']=.1
    elif kind=='cell':diag['spectral_relation_groups'][0]['count']=299
    elif kind=='schema':diag['spectral_relation_scalars']['schema']='other'
    else:diag['spectral_relation_scalars']['used_for_training']=True
    with pytest.raises(ValueError):collect.recount_source_scalars(arrays,ids,diag,relation_contract(VARIANTS[0]))


@pytest.fixture(scope='module')
def physics():
    from experiments.cvs_spectral_relation_identity.physics import frozen_synthetic_diagnostics
    torch.set_num_threads(2);torch.manual_seed(9)
    return {variant:frozen_synthetic_diagnostics(build(variant)) for variant in VARIANTS}


@pytest.mark.parametrize('variant',VARIANTS)
def test_actual_public_physics_schema_and_domain_qualification(variant,physics):
    result=collect.validate_public_physics(physics[variant],relation_contract(variant))
    assert result['eligible_frequencies']+result['excluded_frequencies']==384
    assert result['outcome_used_for_selection'] is False


@pytest.mark.parametrize('kind',['count','missing_eligible','target','fir','receiver','floor','normalization'])
def test_incomplete_public_evidence_rejected(physics,kind):
    value=copy.deepcopy(physics[VARIANTS[1]])
    if kind=='count':value['ideal_gain']['varying_gain_excluded_frequency_count']+=1
    elif kind=='missing_eligible':value['ideal_gain']['varying_frequency_gain_eligible_max_error']=None
    elif kind=='target':value['target_access']=True
    elif kind=='fir':value['finite_fir_sensitivity'].pop()
    elif kind=='receiver':value['physical_parameters']['rx_iq_gains']=[1.,1.]
    elif kind=='floor':value['spectral_relation']['records'][0]['floor_fraction']=2.
    else:value['ideal_gain']['per_frequency_normalization']=False
    with pytest.raises(ValueError):collect.validate_public_physics(value,relation_contract(VARIANTS[1]))


def test_public_outcomes_do_not_create_a_new_selection_gate(physics):
    value=copy.deepcopy(physics[VARIANTS[1]])
    value['whole_model_phase']['ordinary_max_abs_error']=100.
    value['ideal_gain']['common_complex_gain_max_error']=20.
    value['finite_fir_sensitivity'][0]['relation_relative_distance']=50.
    collect.validate_public_physics(value,relation_contract(VARIANTS[1]))


@pytest.fixture(scope='module')
def complete(logs,scalars,physics):
    arrays,source_diag=scalars;rows=[]
    controls=[dict(variant=r['variant'],seed=r['model_seed'],source_output=r['source_output'],accuracy=.91,
        worst_rx=.91,parameters=220987 if r['variant']=='neural_residual_shallow' else 265275,
        macs=100.,provenance='Synthetic independently checked frozen control') for r in control_rows()]
    directory=Path(collect.__file__).parent/'configs'
    for variant in VARIANTS:
        for seed in sorted(SEEDS):
            rid=variant+'-s'+str(seed);folder=PROJECT+'/runs/'+RUN+'/'+rid+'/source'
            contract=relation_contract(variant)
            cfg=json.loads((directory/(rid+'.json')).read_text(encoding='utf-8'))
            cfg.update(commit=collect.SOURCE_COMMIT,spectral_relation_actual=contract,spectral_relation_active=True,
                precision='float32',backend_flags=FULL_FP32_POLICY.copy(),source_counts={'L_s':6300,'U_s':56700,'V':27000},
                steps_per_epoch=50,U_s_use='unused',gradient_clipping=None,optimizer='AdamW+CosineAnnealingLR',loader_seed=seed,
                classifier_scale=30.,total_parameters=247731,trainable_parameters=247731)
            epochs,steps=copy.deepcopy(logs[variant]);log_audit=audit(variant,epochs,steps)
            log_audit.update(roles_match=True,full_source_record_reverified=True,resolved_stdout_matched=True)
            final={k:epochs[-1][k] for k in ('source_val_count','source_val_accuracy','source_val_ce','source_val_rx_accuracy','source_val_worst_rx')}
            diag=copy.deepcopy(source_diag);diag['spectral_relation_scalars']['path']=folder+'/source_spectral_relation_scalars.npz'
            rows.append(dict(row_id=rid,resolved=cfg,epochs=epochs,log_audit=log_audit,
                initialization=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],
                    target_access=False,target_contact=False,model_seed=seed,physical_roles='EXACT_MATCH',selection='fixed_last_epoch'),
                completion=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,
                    backend_flags=FULL_FP32_POLICY.copy(),checkpoint=folder+'/last.pt',final_source_metrics=final),
                profile=dict(total_parameters=247731,trainable_parameters=247731,gradient_used_parameters=247731,
                    conv_linear_macs_per_sample=100.,spectral_relation_contract=contract,
                    resident_state_bytes=1000000,inference_batch1_ms=1.,training_batch128_ms=10.,
                    hardware='synthetic CPU',torch_version='synthetic',
                    all_identity_paths_shared_energy_normalization=False,base_complex_paths_shared_energy_normalization=True),
                source_record=dict(variant=variant,seed=seed,accuracy=.9,worst_rx=.9,parameters=247731,macs=100.,
                    source_output=folder,provenance='Synthetic independently checked scratch source'),
                source_diagnostics=diag,physical_diagnostics=copy.deepcopy(physics[variant]),scalar_npz_base64=encoded_arrays(arrays)))
    inputs=controls+[row['source_record'] for row in rows];selection=select_source_candidate(inputs)
    state=dict(commit=collect.SOURCE_COMMIT,target_access=False,run_id=RUN,status='SOURCE_RESEARCH_COMPLETE_BASELINE_RETAINED')
    return dict(ready=True,run_id=RUN,pipeline=state,source_controls=controls,rows=rows,
        source_selection_inputs=inputs,source_selection=selection,source_v_ids=arrays['ids'].tolist())


def test_complete_eight_rows_80000_steps_and_sixteen_record_selection(complete):
    result=collect.validate_completed(copy.deepcopy(complete))
    assert result==complete['source_selection']
    assert sum(r['log_audit']['steps'] for r in complete['rows'])==80000


@pytest.mark.parametrize('kind',['missing_row','source_commit','init','e199','audit','profile','metrics','geometry',
    'control_path','selection','inputs','npz_path','metadata_pair','fp32'])
def test_incomplete_or_mismatched_completion_cannot_freeze(complete,kind):
    data=copy.deepcopy(complete);row=data['rows'][0]
    if kind=='missing_row':data['rows'].pop()
    elif kind=='source_commit':row['resolved']['commit']='a'*40
    elif kind=='init':row['initialization']['ancestors']=['old.pt']
    elif kind=='e199':row['completion']['epoch']=199
    elif kind=='audit':row['log_audit']['steps']=9999
    elif kind=='profile':row['profile']['gradient_used_parameters']=220987
    elif kind=='metrics':row['completion']['final_source_metrics']['source_val_ce']=.3
    elif kind=='geometry':row['source_diagnostics']['groups'][0]['accuracy']=.5
    elif kind=='control_path':data['source_controls'][0]['source_output']='/other'
    elif kind=='selection':data['source_selection']['selected_variant']=VARIANTS[0]
    elif kind=='inputs':data['source_selection_inputs'][-1]['accuracy']=.7
    elif kind=='npz_path':row['source_diagnostics']['spectral_relation_scalars']['path']='/other'
    elif kind=='fp32':row['resolved']['backend_flags']['cuda_matmul_allow_tf32']=True
    else:
        victim=data['rows'][-1]
        with np.load(io.BytesIO(base64.b64decode(victim['scalar_npz_base64'])),allow_pickle=False) as npz:
            arrays={key:npz[key] for key in npz.files}
        arrays['ids'][[0,300]]=arrays['ids'][[300,0]]
        victim['scalar_npz_base64']=encoded_arrays(arrays)
    with pytest.raises(ValueError):collect.validate_completed(data)


def test_large_evidence_stays_local_and_saved_npz_revalidates(complete,tmp_path):
    data=copy.deepcopy(complete);output=tmp_path/'local_artifacts/synthetic'
    result=collect.save_evidence(data,tmp_path,output)
    assert result['status']=='VERIFIED' and result['source_scalar_packets']==216000
    persisted=json.loads((output/'source_research_complete.json').read_text(encoding='utf-8'))
    assert collect.validate_completed(persisted)==data['source_selection']
    report=tmp_path/'automation_reports/CV-SincNet'/RUN/'evidence'
    assert sorted(p.name for p in report.iterdir())==['source_completion_validation.json','source_evidence_manifest.json','source_selection.json']
    assert len(list((output/'source_scalar_npz').glob('*.npz')))==8
    assert all('scalar_npz_base64' not in row for row in persisted['rows'])
    with pytest.raises(ValueError):collect.save_evidence(copy.deepcopy(complete),tmp_path,tmp_path/'report')
    path=output/'do_not_overwrite.bin';path.write_bytes(b'old')
    with pytest.raises(FileExistsError):collect.write_once_or_identical(path,b'new')
    assert path.read_bytes()==b'old'


def test_remote_injected_auditors_compile_and_no_fit_or_dataset_access():
    script=collect.remote_script();compile(script,'spectral-source-recount','exec')
    assert collect.SOURCE_COMMIT in script and 'source_spectral_relation_scalars.npz' in script
    assert 'torch.load(' not in script and 'ManySig.pkl' not in script
    assert 'build_contract_split(' not in script and 'optimizer.step(' not in script
    # The injected suite is genuinely standalone in the frozen release, not an
    # import of collect.py, which was implemented after that release.
    assert 'from experiments.cvs_spectral_relation_identity.collect' not in script


def test_saved_collector_evidence_runs_complete_analysis_without_schema_adapters(complete,tmp_path):
    from experiments.cvs_spectral_relation_identity import analyze
    output=tmp_path/'local_artifacts/synthetic-analysis'
    collect.save_evidence(copy.deepcopy(complete),tmp_path,output)
    validation=analyze.analyze(tmp_path,output/'source_research_complete.json')
    assert validation['status']=='VERIFIED' and validation['source_relation_cells']==720
    assert validation['source_epochs']==1600 and validation['target_results_read'] is False
    assert validation['selection']==complete['source_selection']
    evidence=tmp_path/'automation_reports/CV-SincNet'/RUN/'evidence'
    assert (evidence/'source_final.csv').exists()
    assert (evidence/'source_curves.png').exists() and (evidence/'source_curves.pdf').exists()


def test_collect_not_ready_does_not_create_outputs(tmp_path,monkeypatch):
    from experiments.cvs_spectral_relation_identity import publish
    commands=[]
    def fake_ssh(command):
        commands.append(command);return b'{"ready": false, "status": "SOURCE_TRAINING"}'
    monkeypatch.setattr(publish,'ssh',fake_ssh)
    assert collect.collect(tmp_path,tmp_path/'local_artifacts/synthetic') is False
    assert not (tmp_path/'local_artifacts').exists()
    assert len(commands)==1 and '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python' in commands[0]
    with pytest.raises(ValueError):collect.collect(tmp_path,tmp_path/'local_artifacts/synthetic','wrong-run')
