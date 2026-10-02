"""Actual scalar execution and full E200 log consistency, no formal data."""
import copy,csv,io,json
import pytest
from experiments.cvs_coupled_identity.collect import audit_logs

def artifacts(epochs,steps):
    compact=[{k:v for k,v in e.items() if not isinstance(v,dict)} for e in epochs]
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(compact[0]));w.writeheader();w.writerows(compact)
    return epochs,steps,stream.getvalue(),'RESOLVED_CONFIG {}\n'+'\n'.join('EPOCH '+json.dumps(e) for e in epochs)

@pytest.fixture(scope='module')
def logs(request):
    learned=False;npar=202553
    flags=dict(ce_weight=1.,augmentation_active=False,domain_backbone_active=False,
        pseudo_labels_active=False,extra_losses_active=False,coupled_active=True,
        gradient_used_parameters=npar,alignment_gradient_used_parameters=int(learned),
        alignment_gradient_norm=.3 if learned else None,cudnn_allow_tf32=False)
    steps=[];epochs=[];alpha=0.
    lift=dict(active=True,records=[dict(block='behavior.0.conv',packets=28,complex_terms=12,actual_envelope_lag=1,
        input_formula_max_abs_error=0.,input_abs_max=3.,degree3_relative_input_change_mean=.1,degree5_relative_input_change_mean=.2)])
    for epoch in range(1,201):
        for batch in range(50):
            before=alpha
            if learned:alpha+=.000001
            steps.append(dict(epoch=epoch,step=(epoch-1)*50+batch+1,clean_ce=.1,total_loss=.1,
                learning_rate=.0002,gradient_norm=2.,source_samples=128 if batch<49 else 28,
                satellite_samples=0,alignment_strength_before=before,alignment_strength_after=alpha,**flags))
        epochs.append(dict(epoch=epoch,optimizer_steps=50,optimizer_steps_total=epoch*50,
            source_sample_exposure=6300,satellite_sample_exposure=0,clean_ce=.1,total_loss=.1,
            learning_rate=.0002,gradient_norm=2.,source_val_count=27000,source_val_accuracy=.9,
            source_val_rx_accuracy={r:.9 for r in ('1','3','4','6','8')},source_val_worst_rx=.9,
            alignment_strength=alpha,coupled_diagnostics=dict(alignment_strength=alpha,coupled_input=lift,normalization=dict(blocks=6,records=[dict(block=p+'.'+str(i),eligible_packets=28,relative_energy_fraction_max_error=1e-7,relative_energy_fraction_variance_mean=.02) for p in ('time','behavior') for i in range(3)])),
            elapsed_seconds=1.,peak_cuda_allocated_bytes=1234,**flags))
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
    elif kind=='missing_normalization':epochs[180]['coupled_diagnostics']['normalization']['records'].pop()
    elif kind=='wrong_normalization_count':epochs[180]['coupled_diagnostics']['normalization']['records'][0]['eligible_packets']=29
    elif kind=='nonfinite_normalization':epochs[180]['coupled_diagnostics']['normalization']['records'][0]['relative_energy_fraction_max_error']=float('nan')
    data=artifacts(epochs,steps)
    if kind=='stdout':data=(*data[:-1],data[-1].replace('"gradient_norm": 2.0','"gradient_norm": 3.0',1))
    with pytest.raises(ValueError):audit_logs(*data,npar,alpha)

@pytest.mark.parametrize('kind',['missing','wrong_lag','wrong_terms','nonfinite'])
def test_coupled_input_execution_measurements_rejected(logs,kind):
    (epochs,steps,_,_),npar,alpha=copy.deepcopy(logs)
    lift=epochs[73]['coupled_diagnostics']['coupled_input']
    if kind=='missing':lift['records']=[]
    elif kind=='wrong_lag':lift['records'][0]['actual_envelope_lag']=4
    elif kind=='wrong_terms':lift['records'][0]['complex_terms']=28
    else:lift['records'][0]['degree3_relative_input_change_mean']=float('nan')
    with pytest.raises(ValueError):audit_logs(*artifacts(epochs,steps),npar,alpha,expected_envelope_lag=1)
