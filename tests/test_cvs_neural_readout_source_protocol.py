"""Source-only architecture protocol and complete measured-log reconciliation."""
import copy,csv,io,json,math
import pytest
from experiments.cvs_neural_readout_identity.model import VARIANTS,readout_contract
from experiments.cvs_neural_readout_identity.source import validate_config
from experiments.cvs_neural_readout_identity.dispatch import CANDIDATES,SEEDS,select_source_candidate
from experiments.cvs_neural_readout_identity.collect import audit_logs
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY

def config(variant):
    return dict(method='cvs_neural_readout_identity',variant=variant,epochs=200,batch_size=128,lr=.0002,
                lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,
                extra_losses=[],selection='fixed_last_epoch',split_seed=392005,
                readout=readout_contract(variant),numerical_policy=FULL_FP32_POLICY.copy())

@pytest.mark.parametrize('variant',VARIANTS)
def test_original_ce_training_contract(variant):
    c=config(variant)
    assert validate_config(c)==c


def test_baseline_source_record_without_readout_uses_its_own_validator(tmp_path):
    from experiments.cvs_neural_readout_identity import dispatch
    from experiments.cvs_neural_residual_identity.model import neural_contract

    seed=2026092701;variant='neural_residual_shallow';method='cvs_neural_residual_identity'
    resolved=config(VARIANTS[0]);resolved.pop('readout')
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
    assert 'readout' not in resolved
    record=dispatch.read_source_record(dict(variant=variant,model_seed=seed,source_output=str(tmp_path)),contract,method)
    assert record['variant']==variant and record['parameters']==220987
    assert record['accuracy']==.9 and record['source_output']==str(tmp_path)


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
    if corruption=='architecture':c['readout']['attention_heads']=3
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

@pytest.fixture(scope='module',params=[306147,310243])
def logs(request):
    npar=request.param
    flags=dict(ce_weight=1.,augmentation_active=False,domain_backbone_active=False,pseudo_labels_active=False,
               extra_losses_active=False,learned_readout_active=True,gradient_used_parameters=npar,
               alignment_gradient_used_parameters=0,alignment_gradient_norm=None,cudnn_allow_tf32=False,
               mixture_gradient_used_parameters=2,mix_gradient3=.001,mix_gradient5=-.002,mix_gradient_norm=.00223606797749979,
               readout_gradient_norm=.01,readout_gradient_used_parameters=npar-220987)
    lift=dict(active=True,raw_parameters=[0.,0.],coefficients=[0.,0.],records=[dict(block='behavior.0.conv',packets=28,
        complex_terms=12,actual_envelope_lag=4,actual_phase_lag=4,input_formula_max_abs_error=0.,input_abs_max=3.,
        degree3_relative_input_change_mean=.1,degree5_relative_input_change_mean=.2)])
    diag=dict(alignment_strength=0.,adaptive_input=lift,normalization=dict(blocks=6,records=[dict(block=p+'.'+str(i),
        eligible_packets=28,relative_energy_fraction_max_error=1e-7,relative_energy_fraction_variance_mean=.02)
        for p in ('time','behavior') for i in range(3)]),learned_readout=dict(active=True,records=[dict(block=p+'.readout',
        packets=28,relative_output_change_mean=.1,projection_norm=1.,attention_entropy_mean=math.log(64),
        attention_effective_tokens_mean=64.) for p in ('time','behavior')]))
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
            alignment_strength=0.,readout_diagnostics=copy.deepcopy(diag),elapsed_seconds=1.,peak_cuda_allocated_bytes=1234,**flags))
    return epochs,steps,npar

def test_all10000_steps_and200epochs_match(logs):
    epochs,steps,npar=logs;data,compact=artifacts(epochs,steps)
    assert audit_logs(*data,npar,0.,compact,expected_phase_lag=4)['steps']==10000

@pytest.mark.parametrize('kind',['missing_step','extra_loss','precision','wrong_exposure','wrong_average','stdout',
    'readout_gradient','readout_paths','attention_nan','attention_tokens','projection','old_gate_jump','compact'])
def test_measured_execution_corruption_rejected(logs,kind):
    epochs,steps,npar=copy.deepcopy(logs)
    diag=epochs[73]['readout_diagnostics']['learned_readout']
    if kind=='missing_step':steps.pop(17)
    elif kind=='extra_loss':steps[500]['extra_losses_active']=True
    elif kind=='precision':steps[500]['cudnn_allow_tf32']=True
    elif kind=='wrong_exposure':steps[49]['source_samples']=128
    elif kind=='wrong_average':steps[500]['clean_ce']=steps[500]['total_loss']=.2
    elif kind=='readout_gradient':steps[500]['readout_gradient_used_parameters']=1
    elif kind=='readout_paths':diag['records'][0]['block']='frequency.readout'
    elif kind=='attention_nan':diag['records'][0]['attention_entropy_mean']=float('nan')
    elif kind=='attention_tokens':diag['records'][0]['attention_effective_tokens_mean']=65
    elif kind=='projection':diag['records'][0]['projection_norm']=-1
    elif kind=='old_gate_jump':steps[500]['mix_raw3_before']=.1
    data,compact=artifacts(epochs,steps)
    if kind=='stdout':data=(*data[:-1],data[-1].replace('"gradient_norm": 2.0','"gradient_norm": 3.0',1))
    elif kind=='compact':compact=compact.replace('"clean_ce": 0.1','"clean_ce": 0.2',1)
    with pytest.raises(ValueError):audit_logs(*data,npar,0.,compact,expected_phase_lag=4)
