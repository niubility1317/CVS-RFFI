"""Source-only architecture protocol and complete measured-log reconciliation."""
import copy,csv,io,json,math
import pytest
from experiments.cvs_response_fusion_identity.model import VARIANTS,fusion_contract as response_contract
from experiments.cvs_response_fusion_identity.source import validate_config
from experiments.cvs_response_fusion_identity.dispatch import CANDIDATES,SEEDS,select_source_candidate
from experiments.cvs_response_fusion_identity.collect import audit_logs
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY

def config(variant):
    return dict(method='cvs_response_fusion_identity',variant=variant,epochs=200,batch_size=128,lr=.0002,
                lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,
                extra_losses=[],selection='fixed_last_epoch',split_seed=392005,
                response=response_contract(variant),numerical_policy=FULL_FP32_POLICY.copy())

@pytest.mark.parametrize('variant',VARIANTS)
def test_original_ce_training_contract(variant):
    c=config(variant)
    assert validate_config(c)==c


def test_baseline_source_record_without_readout_uses_its_own_validator(tmp_path):
    from experiments.cvs_response_fusion_identity import dispatch
    from experiments.cvs_neural_residual_identity.model import neural_contract

    seed=2026092701;variant='neural_residual_shallow';method='cvs_neural_residual_identity'
    resolved=config(VARIANTS[0]);resolved.pop('response')
    architecture=neural_contract(variant)
    resolved.update(method=method,variant=variant,model_seed=seed,neural=architecture,
        neural_actual=architecture,neural_residual_active=True,classifier_scale=30.,
        total_parameters=220987,steps_per_epoch=50,source_counts={'L_s':6300,'U_s':56700,'V':27000},
        U_s_use='unused',target_access=False,precision='float32',gradient_clipping=None,
        optimizer='AdamW+CosineAnnealingLR',loader_seed=seed,output_root=str(tmp_path),
        source_contract=dispatch.PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json',
        dataset=dispatch.PROJECT+'/Dataset_WigSig/ManySig.pkl',backend_flags=FULL_FP32_POLICY.copy())
    initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],
        target_access=False,target_contact=False,model_seed=seed,physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
    contract=dict(physical_roles='EXACT_MATCH',dataset_path=resolved['dataset'],
        classes=['14-10','14-7','20-15','20-19','6-15','8-20'],equalized=1,out_len=256,normalize=True)
    metrics=dict(source_val_count=27000,source_val_accuracy=.9,source_val_worst_rx=.9,
        source_val_rx_accuracy={rx:.9 for rx in ('1','3','4','6','8')})
    complete=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,
        final_source_metrics=metrics,backend_flags=FULL_FP32_POLICY.copy())
    profile=dict(total_parameters=220987,gradient_used_parameters=220987,conv_linear_macs_per_sample=100)
    for name,value in [('completion.json',complete),('resolved_config.json',resolved),
                       ('initialization.json',initial),('source_contract.json',contract),('resource_profile.json',profile)]:
        (tmp_path/name).write_text(json.dumps(value),encoding='utf-8')
    assert 'response' not in resolved
    record=dispatch.read_source_record(dict(variant=variant,model_seed=seed,source_output=str(tmp_path)),contract,method)
    assert record['variant']==variant and record['parameters']==220987
    assert record['accuracy']==.9 and record['source_output']==str(tmp_path)


@pytest.mark.parametrize('variant',VARIANTS)
def test_actual_response_metadata_and_flags_are_validated(tmp_path,variant):
    from experiments.cvs_response_fusion_identity import dispatch
    seed=2026092701;c=config(variant);architecture=c['response']
    c.update(model_seed=seed,response_actual=copy.deepcopy(architecture),classifier_scale=30.,
        total_parameters=architecture['total_parameters'],trainable_parameters=architecture['total_trainable_parameters'],
        steps_per_epoch=50,source_counts={'L_s':6300,'U_s':56700,'V':27000},U_s_use='unused',target_access=False,
        precision='float32',gradient_clipping=None,optimizer='AdamW+CosineAnnealingLR',loader_seed=seed,
        output_root=str(tmp_path),source_contract=dispatch.PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json',
        dataset=dispatch.PROJECT+'/Dataset_WigSig/ManySig.pkl',backend_flags=FULL_FP32_POLICY.copy(),
        **{key:architecture[key] for key in ('response_active','response_compensation_active','response_order_active')})
    contract=dict(physical_roles='EXACT_MATCH',dataset_path=c['dataset'],classes=['14-10','14-7','20-15','20-19','6-15','8-20'],equalized=1,out_len=256,normalize=True)
    initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],target_access=False,target_contact=False,model_seed=seed,physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
    metrics=dict(source_val_count=27000,source_val_accuracy=.9,source_val_worst_rx=.9,source_val_rx_accuracy={rx:.9 for rx in ('1','3','4','6','8')})
    done=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,final_source_metrics=metrics,backend_flags=FULL_FP32_POLICY.copy())
    profile=dict(total_parameters=architecture['total_parameters'],gradient_used_parameters=architecture['total_trainable_parameters'],conv_linear_macs_per_sample=100)
    for name,value in [('completion.json',done),('resolved_config.json',c),('initialization.json',initial),('source_contract.json',contract),('resource_profile.json',profile)]:
        (tmp_path/name).write_text(json.dumps(value),encoding='utf-8')
    row=dict(variant=variant,model_seed=seed,source_output=str(tmp_path))
    assert dispatch.read_source_record(row,contract,'cvs_response_fusion_identity')['parameters']==architecture['total_parameters']
    for flag in ('response_active','response_compensation_active','response_order_active'):
        invalid=copy.deepcopy(c);invalid[flag]=not invalid[flag]
        (tmp_path/'resolved_config.json').write_text(json.dumps(invalid),encoding='utf-8')
        with pytest.raises(ValueError,match='architecture'):
            dispatch.read_source_record(row,contract,'cvs_response_fusion_identity')


@pytest.mark.parametrize('variant',VARIANTS)
def test_measured_model_diagnostics_match_source_collector(variant):
    import torch
    from experiments.cvs_response_fusion_identity.model import build
    from experiments.cvs_response_fusion_identity.collect import validate_response_diagnostics
    torch.set_num_threads(2);torch.manual_seed(20261003)
    model=build(variant).train();x=torch.randn(28,2,256)
    torch.nn.functional.cross_entropy(model(x),torch.arange(28)%6).backward()
    measured=model.diagnostics(x)
    validate_response_diagnostics(measured['channel_response'],response_contract(variant),28)
    assert sum(p.numel() for p in model.parameters() if p.grad is not None)==response_contract(variant)['total_trainable_parameters']


def test_registered8_rows_and_generated_remote_templates(tmp_path,monkeypatch):
    import ast
    from pathlib import Path
    from experiments.cvs_response_fusion_identity import prepare,dispatch,publish,collect
    root=prepare.ROOT
    spec=json.loads((root/'experiments/cvs_response_fusion_identity/configs/launch_spec.json').read_text(encoding='utf-8'))
    prefix=prepare.PROJECT+'/releases/'+prepare.RELEASE+'/'
    original=Path.read_text
    def mapped(path,*args,**kwargs):
        name=str(path).replace('\\','/')
        return original(root/name[len(prefix):] if name.startswith(prefix) else path,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'read_text',mapped)
        assert dispatch.validate_spec(spec)==spec
    assert len(spec['rows'])==8 and len(spec['source_controls'])==20
    assert len({r['source_output'] for r in spec['rows']})==8
    assert not ({r['source_output'] for r in spec['rows']}&{r['source_output'] for r in spec['source_controls']})
    compile(publish.REMOTE.replace('c=CONFIG','c='+repr({'project':'/synthetic'}),1),'publish','exec')
    compile(publish.INSPECT.replace('PROJECT',repr(prepare.PROJECT)).replace('RELEASE',repr(prepare.RELEASE)).replace('RUN',repr(prepare.RUN)),'inspect','exec')
    scripts=[]
    monkeypatch.setattr(publish,'ssh',lambda script:scripts.append(script) or json.dumps(dict(ready=False,status='SYNTHETIC_NOT_READY')))
    assert collect.collect(root,tmp_path) is False
    wrapper=ast.parse(scripts[0]);command=ast.literal_eval(wrapper.body[1].value.args[0])
    compile(command[-1],'assembled-collector','exec')


def test_complete8_source_rows_validate_with_real_frozen_physics():
    import torch
    from experiments.cvs_response_fusion_identity.model import build
    from experiments.cvs_response_fusion_identity.physics import frozen_synthetic_diagnostics
    from experiments.cvs_response_fusion_identity.collect import validate_completed
    from experiments.cvs_response_fusion_identity.dispatch import CONTROL,CHANNEL_CONTROL
    torch.set_num_threads(2)
    controls=[dict(variant=v,seed=seed,accuracy=.9,worst_rx=.9,parameters=220987,macs=100) for v in CANDIDATES if v not in VARIANTS for seed in SEEDS]
    rows=[]
    for variant in VARIANTS:
        c=response_contract(variant);model=build(variant).train();x=torch.randn(28,2,256)
        torch.nn.functional.cross_entropy(model(x),torch.arange(28)%6).backward()
        diagnostic=model.diagnostics(x);physical=frozen_synthetic_diagnostics(model)
        groups=[dict(tx=tx,receiver=rx,day=day,count=300,alignment_strength=0.,
            residual_valid_fraction=1.,residual_formula_eligible_count=300,residual_formula_max_error_hz=0.,
            relative_cfo_hz_mean=0.,relative_cfo_hz_min=0.,relative_cfo_hz_max=0.,coherence_mean=1.,fallback_fraction=0.,
            residual_estimated_cfo_hz_mean=0.,nominal_residual_cfo_hz_mean=0.)
            for tx in range(6) for rx in (1,3,4,6,8) for day in (1,2,3)]
        source_diag=dict(role='V',count=27000,groups=groups,target_access=False,used_for_training=False,used_for_selection=False)
        profile=dict(total_parameters=c['total_parameters'],trainable_parameters=c['total_trainable_parameters'],
            gradient_used_parameters=c['total_trainable_parameters'],conv_linear_macs_per_sample=100,
            all_identity_paths_shared_energy_normalization=False,base_identity_paths_shared_energy_normalization=True,
            response_contract=c)
        metrics=dict(source_val_count=27000,source_val_accuracy=.9,source_val_worst_rx=.9)
        epochs=[dict(epoch=epoch,alignment_strength=0.,response_diagnostics=diagnostic,**metrics) for epoch in range(1,201)]
        for seed in SEEDS:
            rows.append(dict(resolved=dict(variant=variant,model_seed=seed,response=c,response_actual=c),
                completion=dict(final_source_metrics=metrics),epochs=epochs,profile=profile,
                source_diagnostics=source_diag,physical_diagnostics=physical,
                source_record=dict(variant=variant,seed=seed,accuracy=.9,worst_rx=.9,parameters=c['total_parameters'],macs=100)))
    selection=select_source_candidate(controls+[row['source_record'] for row in rows])
    complete=dict(rows=rows,source_controls=controls,source_selection=selection)
    assert validate_completed(complete)==selection
    broken=copy.deepcopy(complete);broken['rows'][0]['profile']['all_identity_paths_shared_energy_normalization']=True
    with pytest.raises(ValueError,match='resource accounting'):
        validate_completed(broken)


@pytest.mark.parametrize('key,value',[
    ('checkpoint','old.pt'),('resume','old.pt'),('teacher','old.pt'),('initial_checkpoint','old.pt'),
    ('target_inputs','query.npz'),('target_truth','truth.json'),('p1_truth','truth.json'),('p1_capsule','capsule'),
    ('epochs',199),('batch_size',256),('lr',.001),('lr_min',0.),('weight_decay',0.),('drop_last',True),
    ('augmentation',True),('domain_backbone',True),('extra_losses',['aux']),('selection','best_epoch'),('split_seed',1)])
def test_source_contract_rejects_changed_training_or_inheritance(key,value):
    c=config(VARIANTS[0]);c[key]=value
    with pytest.raises(ValueError):validate_config(c)

@pytest.mark.parametrize('corruption',['architecture','precision','unknown_variant'])
def test_invalid_architecture_or_numerics(corruption):
    c=config(VARIANTS[0])
    if corruption=='architecture':c['response']['total_parameters']+=1
    elif corruption=='precision':c['numerical_policy']['cudnn_allow_tf32']=True
    else:c['variant']='unregistered'
    with pytest.raises(ValueError):validate_config(c)

def records(winner):
    return [dict(variant=v,seed=s,accuracy=.99 if v==winner else .98,
                 worst_rx=.98 if v==winner else .96,parameters=310243 if v==winner else 1,macs=200 if v==winner else 1)
            for v in CANDIDATES for s in SEEDS]

@pytest.mark.parametrize('winner',CANDIDATES)
def test_fixed_source_ranking_beats_cost_and_handles_control(winner):
    chosen=select_source_candidate(records(winner))
    assert chosen['selected_variant']==winner
    assert chosen['new_candidate_selected']==(winner in VARIANTS)
    assert chosen['target_access'] is False and chosen['target_score_used'] is False

@pytest.mark.parametrize('kind',['missing','duplicate_seed','extra'])
def test_source_selection_rejects_incomplete_matrix(kind):
    rows=records(VARIANTS[0])
    if kind=='missing':rows.pop()
    elif kind=='duplicate_seed':rows[-1]['seed']=rows[-2]['seed']
    else:rows.append(copy.deepcopy(rows[-1]))
    with pytest.raises(ValueError):select_source_candidate(rows)

def artifacts(epochs,steps):
    compact=[{k:v for k,v in e.items() if not isinstance(v,dict)} for e in epochs]
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)
    return (epochs,steps,stream.getvalue(),'RESOLVED_CONFIG {}\n'+'\n'.join('EPOCH '+json.dumps(e) for e in epochs)), '\n'.join(json.dumps(r) for r in compact)

@pytest.fixture(scope='module',params=VARIANTS)
def logs(request):
    contract=response_contract(request.param);npar=contract['total_trainable_parameters']
    flags=dict(ce_weight=1.,augmentation_active=False,domain_backbone_active=False,pseudo_labels_active=False,
               extra_losses_active=False,response_active=True,response_compensation_active=contract['response_compensation_active'],response_order_active=contract['response_order_active'],gradient_used_parameters=npar,
               alignment_gradient_used_parameters=0,alignment_gradient_norm=None,cudnn_allow_tf32=False,
               mixture_gradient_used_parameters=2,mix_gradient3=.001,mix_gradient5=-.002,mix_gradient_norm=.00223606797749979,
               response_gradient_norm=.01,response_gradient_used_parameters=contract['new_trainable_parameters'])
    lift=dict(active=True,raw_parameters=[0.,0.],coefficients=[0.,0.],records=[dict(block='behavior.0.conv',packets=28,
        complex_terms=12,actual_envelope_lag=4,actual_phase_lag=4,input_formula_max_abs_error=0.,input_abs_max=3.,
        degree3_relative_input_change_mean=.1,degree5_relative_input_change_mean=.2)])
    response=dict(active=True,variant=request.param,packets=28,
        compensation_input_relative=.1,response_feature_relative=.1,response_token_norm=.1,
        response_pooled_norm=.1,response_projection_relative=.1,attention_entropy=1.,
        compensation_exit_gradient_norm=.01,value_gradient_norm=.01,projection_gradient_norm=.01,
        response_order_relative=.1 if contract['response_order_active'] else None,
        attention_gradient_norm=.01 if contract['response_attention_active'] else None)
    diag=dict(alignment_strength=0.,adaptive_input=lift,normalization=dict(blocks=6,records=[dict(block=p+'.'+str(i),
        eligible_packets=28,relative_energy_fraction_max_error=1e-7,relative_energy_fraction_variance_mean=.02)
        for p in ('time','behavior') for i in range(3)]),channel_response=response,response_fusion=dict(active=True,variant=request.param,packets=28,raw_response_relative=.1,used_response_norm=.1,used_response_norm_max=.2,base_feature_norm=1.,centered_response_logit_norm=.2,final_joint_normalization=contract['fusion_final_joint_normalization']))
    steps=[];epochs=[]
    for epoch in range(1,201):
        for batch in range(50):
            steps.append(dict(epoch=epoch,step=(epoch-1)*50+batch+1,clean_ce=.1,total_loss=.1,
                learning_rate=.0002,gradient_norm=2.,source_samples=128 if batch<49 else 28,satellite_samples=0,
                mix_raw3_before=0.,mix_raw5_before=0.,mix_raw3_after=0.,mix_raw5_after=0.,
                mix_coefficient3_before=0.,mix_coefficient5_before=0.,mix_coefficient3_after=0.,mix_coefficient5_after=0.,
                alignment_strength_before=0.,alignment_strength_after=0.,**flags))
        epochs.append(dict(epoch=epoch,optimizer_steps=50,optimizer_steps_total=epoch*50,source_sample_exposure=6300,
            satellite_sample_exposure=0,clean_ce=.1,total_loss=.1,learning_rate=.0002,gradient_norm=2.,
            source_val_count=27000,source_val_accuracy=.9,source_val_rx_accuracy={r:.9 for r in ('1','3','4','6','8')},
            source_val_worst_rx=.9,mix_raw3=0.,mix_raw5=0.,mix_coefficient3=0.,mix_coefficient5=0.,
            alignment_strength=0.,response_diagnostics=copy.deepcopy(diag),elapsed_seconds=1.,peak_cuda_allocated_bytes=1234,**flags))
    return epochs,steps,contract

def test_all10000_steps_and200epochs_match(logs):
    epochs,steps,contract=logs;npar=contract['total_trainable_parameters'];data,compact=artifacts(epochs,steps)
    assert audit_logs(*data,npar,0.,compact,expected_phase_lag=4,expected_response_contract=contract)['steps']==10000

@pytest.mark.parametrize('kind',['missing_step','extra_loss','precision','wrong_exposure','wrong_average','stdout',
    'response_gradient','response_flags','response_nan','response_packets','response_NA','old_gate_jump','compact'])
def test_measured_execution_corruption_rejected(logs,kind):
    epochs,steps,contract=copy.deepcopy(logs);npar=contract['total_trainable_parameters']
    diag=epochs[73]['response_diagnostics']['channel_response']
    if kind=='missing_step':steps.pop(17)
    elif kind=='extra_loss':steps[500]['extra_losses_active']=True
    elif kind=='precision':steps[500]['cudnn_allow_tf32']=True
    elif kind=='wrong_exposure':steps[49]['source_samples']=128
    elif kind=='wrong_average':steps[500]['clean_ce']=steps[500]['total_loss']=.2
    elif kind=='response_gradient':steps[500]['response_gradient_used_parameters']=1
    elif kind=='response_flags':diag['active']=False
    elif kind=='response_nan':diag['value_gradient_norm']=float('nan')
    elif kind=='response_packets':diag['packets']=29
    elif kind=='response_NA':diag['compensation_input_relative']=None
    elif kind=='old_gate_jump':steps[500]['mix_raw3_before']=.1
    data,compact=artifacts(epochs,steps)
    if kind=='stdout':data=(*data[:-1],data[-1].replace('"gradient_norm": 2.0','"gradient_norm": 3.0',1))
    elif kind=='compact':compact=compact.replace('"clean_ce": 0.1','"clean_ce": 0.2',1)
    with pytest.raises(ValueError):audit_logs(*data,npar,0.,compact,expected_phase_lag=4,expected_response_contract=contract)


def test_fusion_telemetry_cannot_report_unbounded_anchor_or_missing_used_response():
    import torch
    from experiments.cvs_response_fusion_identity.model import build
    from experiments.cvs_response_fusion_identity.collect import validate_fusion_diagnostics
    model=build('response_anchor_mean');c=model.contract();d=model.fusion_diagnostics(torch.randn(3,2,256))
    validate_fusion_diagnostics(d,c,3)
    for broken in (dict(d,used_response_norm_max=1.01),dict(d,final_joint_normalization=True),dict(d,centered_response_logit_norm=float('nan'))):
        with pytest.raises(ValueError):validate_fusion_diagnostics(broken,c,3)
