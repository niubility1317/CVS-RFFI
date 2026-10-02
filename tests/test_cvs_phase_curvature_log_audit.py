"""Actual scalar execution and full E200 log consistency, no formal data."""
import copy,csv,io,json
import pytest
from experiments.cvs_phase_curvature_identity.collect import audit_logs as source_audit
from experiments.cvs_phase_curvature_identity.telemetry import memory_step_metrics,memory_epoch_metrics
from experiments.cvs_phase_curvature_identity.model import PATHS
def audit_logs(*args,**kwargs):
    kwargs.setdefault('expected_curvature_delays',[1,4,5])
    return source_audit(*args,**kwargs)


def artifacts(epochs,steps):
    compact=[{k:v for k,v in e.items() if not isinstance(v,dict)} for e in epochs]
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(compact[0]));w.writeheader();w.writerows(compact)
    return epochs,steps,stream.getvalue(),'RESOLVED_CONFIG {}\n'+'\n'.join('EPOCH '+json.dumps(e) for e in epochs)

@pytest.fixture(scope='module')
def logs(request):
    learned=False;npar=202561
    flags=dict(ce_weight=1.,augmentation_active=False,domain_backbone_active=False,
        pseudo_labels_active=False,extra_losses_active=False,feature_curvature_active=True,
        gradient_used_parameters=npar,alignment_gradient_used_parameters=int(learned),
        alignment_gradient_norm=.3 if learned else None,cudnn_allow_tf32=False,mixture_gradient_used_parameters=2,mix_gradient3=.001,mix_gradient5=-.002,mix_gradient_norm=.00223606797749979)
    steps=[];epochs=[];alpha=0.
    lift=dict(active=True,raw_parameters=[0.,0.],coefficients=[0.,0.],records=[dict(block='behavior.0.conv',packets=28,complex_terms=12,actual_envelope_lag=4,actual_phase_lag=4,
        input_formula_max_abs_error=0.,input_abs_max=3.,degree3_relative_input_change_mean=.1,degree5_relative_input_change_mean=.2)])
    for epoch in range(1,201):
        for batch in range(50):
            before=alpha
            if learned:alpha+=.000001
            steps.append(dict(epoch=epoch,step=(epoch-1)*50+batch+1,clean_ce=.1,total_loss=.1,
                learning_rate=.0002,gradient_norm=2.,source_samples=128 if batch<49 else 28,
                satellite_samples=0,mix_raw3_before=0.,mix_raw5_before=0.,mix_raw3_after=0.,mix_raw5_after=0.,mix_coefficient3_before=0.,mix_coefficient5_before=0.,mix_coefficient3_after=0.,mix_coefficient5_after=0.,alignment_strength_before=before,alignment_strength_after=alpha,**flags))
        epochs.append(dict(epoch=epoch,optimizer_steps=50,optimizer_steps_total=epoch*50,
            source_sample_exposure=6300,satellite_sample_exposure=0,clean_ce=.1,total_loss=.1,
            learning_rate=.0002,gradient_norm=2.,source_val_count=27000,source_val_accuracy=.9,
            source_val_rx_accuracy={r:.9 for r in ('1','3','4','6','8')},source_val_worst_rx=.9,
            mix_raw3=0.,mix_raw5=0.,mix_coefficient3=0.,mix_coefficient5=0.,alignment_strength=alpha,curvature_diagnostics=dict(alignment_strength=alpha,adaptive_input=lift,normalization=dict(blocks=6,records=[dict(block=p+'.'+str(i),eligible_packets=28,relative_energy_fraction_max_error=1e-7,relative_energy_fraction_variance_mean=.02) for p in ('time','behavior') for i in range(3)])),
            elapsed_seconds=1.,peak_cuda_allocated_bytes=1234,**flags))
    for step in steps:step.update(memory_step_metrics([(0.,0.)]*6,[(0.,0.)]*6,[.001]*6))
    for epoch in epochs:
        epoch.update(memory_epoch_metrics([(0.,0.)]*6,[.05]*6,[.05]*6,50))
        epoch['curvature_diagnostics']['feature_curvature']=dict(active=True,records=[dict(
            block=path,packets=28,complex_channels=16,grid_length=64,actual_delays=[1,4,5],raw_parameter=0.,coefficient=0.,
            input_formula_max_abs_error=0.,prefix_correction_max_abs_error=0.,delta_complex_abs_max=3.,
            actual_relative_output_change_mean=0.,eligible_grid_positions=59) for path in PATHS])
    return artifacts(epochs,steps),npar,alpha

def test_all10000_steps_and_fixed_or_learned_telemetry(logs):
    data,npar,alpha=logs;assert audit_logs(*data,npar,alpha)['steps']==10000

def test_compact_jsonl_matches_all_measured_epochs(logs):
    data,npar,alpha=logs
    compact=[{k:v for k,v in e.items() if not isinstance(v,dict)} for e in data[0]]
    content='\n'.join(json.dumps(r) for r in compact)
    assert audit_logs(*data,npar,alpha,content)['epochs']==200
    compact[13]['clean_ce']=999
    with pytest.raises(ValueError):audit_logs(*data,npar,alpha,'\n'.join(json.dumps(r) for r in compact))

@pytest.mark.parametrize('kind',['missing_step','extra_loss','wrong_exposure','precision','missing_scalar_grad','alpha_jump','alpha_range','epoch_alpha','wrong_average','stdout','missing_normalization','wrong_normalization_count','nonfinite_normalization'])
def test_corrupt_execution_rejected(logs,kind):
    (epochs,steps,_,_),npar,alpha=copy.deepcopy(logs)
    if kind=='missing_step':steps.pop(17)
    elif kind=='extra_loss':steps[9000]['extra_losses_active']=True
    elif kind=='wrong_exposure':steps[49]['source_samples']=128
    elif kind=='precision':steps[9000]['cudnn_allow_tf32']=True
    elif kind=='missing_scalar_grad':steps[9000]['alignment_gradient_norm']=None if npar==202554 else 0.
    elif kind=='alpha_jump':steps[9000]['alignment_strength_before']+=.01
    elif kind=='alpha_range':steps[9000]['alignment_strength_after']=1.1
    elif kind=='epoch_alpha':epochs[180]['alignment_strength']+=.01
    elif kind=='wrong_average':steps[9000]['clean_ce']=.2;steps[9000]['total_loss']=.2
    elif kind=='missing_normalization':epochs[180]['curvature_diagnostics']['normalization']['records'].pop()
    elif kind=='wrong_normalization_count':epochs[180]['curvature_diagnostics']['normalization']['records'][0]['eligible_packets']=29
    elif kind=='nonfinite_normalization':epochs[180]['curvature_diagnostics']['normalization']['records'][0]['relative_energy_fraction_max_error']=float('nan')
    data=artifacts(epochs,steps)
    if kind=='stdout':data=(*data[:-1],data[-1].replace('"gradient_norm": 2.0','"gradient_norm": 3.0',1))
    with pytest.raises(ValueError):audit_logs(*data,npar,alpha)

@pytest.mark.parametrize('kind',['missing','wrong_lag','wrong_terms','nonfinite'])
def test_adaptive_input_execution_measurements_rejected(logs,kind):
    (epochs,steps,_,_),npar,alpha=copy.deepcopy(logs)
    lift=epochs[73]['curvature_diagnostics']['adaptive_input']
    if kind=='missing':lift['records']=[]
    elif kind=='wrong_lag':lift['records'][0]['actual_phase_lag']=1
    elif kind=='wrong_terms':lift['records'][0]['complex_terms']=28
    else:lift['records'][0]['degree3_relative_input_change_mean']=float('nan')
    with pytest.raises(ValueError):audit_logs(*artifacts(epochs,steps),npar,alpha,expected_phase_lag=4)

@pytest.mark.parametrize('kind',['missing_gradient','raw_jump','coefficient_formula','input_gate','nonfinite_gradient'])
def test_corrupt_adaptive_gate_execution_rejected(logs,kind):
    (epochs,steps,_,_),npar,alpha=copy.deepcopy(logs)
    if kind=='missing_gradient':steps[55]['mixture_gradient_used_parameters']=0
    elif kind=='raw_jump':steps[55]['mix_raw3_before']=.1
    elif kind=='coefficient_formula':steps[55]['mix_coefficient3_after']=.1
    elif kind=='input_gate':epochs[5]['curvature_diagnostics']['adaptive_input']['coefficients']=[.1,0.]
    else:steps[55]['mix_gradient3']=float('nan')
    with pytest.raises(ValueError):audit_logs(*artifacts(epochs,steps),npar,alpha)

@pytest.mark.parametrize('kind',['missing_gradient','raw_jump','tanh','nonfinite_gradient','epoch_average','missing_block','wrong_delay','formula','prefix','bound','actual_gate','eligibility'])
def test_corrupt_six_feature_execution_rejected(logs,kind):
    (epochs,steps,_,_),npar,alpha=copy.deepcopy(logs)
    rec=epochs[5]['curvature_diagnostics']['feature_curvature']['records'][0]
    if kind=='missing_gradient':steps[55]['memory_gradient_used_parameters']=5
    elif kind=='raw_jump':steps[55]['memory_time_0_raw_before']=.1
    elif kind=='tanh':steps[55]['memory_time_0_coefficient_after']=.1
    elif kind=='nonfinite_gradient':steps[55]['memory_time_0_gradient']=float('nan')
    elif kind=='epoch_average':epochs[5]['memory_time_0_gradient_abs_mean']=.1
    elif kind=='missing_block':epochs[5]['curvature_diagnostics']['feature_curvature']['records'].pop()
    elif kind=='wrong_delay':rec['actual_delays']=[2,4,6]
    elif kind=='formula':rec['input_formula_max_abs_error']=.1
    elif kind=='prefix':rec['prefix_correction_max_abs_error']=.1
    elif kind=='bound':rec['delta_complex_abs_max']=4.1
    elif kind=='actual_gate':rec['raw_parameter']=.1
    elif kind=='eligibility':rec['eligible_grid_positions']=58
    with pytest.raises(ValueError):audit_logs(*artifacts(epochs,steps),npar,alpha)
