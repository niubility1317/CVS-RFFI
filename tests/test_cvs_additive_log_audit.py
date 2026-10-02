"""Full-budget reconciliation rejects corruption without touching formal data."""
import copy
import csv
import io
import json
import pytest
from experiments.cvs_additive_identity.collect import audit_logs


@pytest.fixture(scope='module')
def complete_logs():
    flags=dict(ce_weight=1.,augmentation_active=False,domain_backbone_active=False,
               pseudo_labels_active=False,extra_losses_active=False,coordinate_active=True,
               gradient_used_parameters=203193,conditioner_gradient_used_parameters=640,conditioner_gradient_norm=.3,cudnn_allow_tf32=False)
    steps=[];epochs=[]
    for epoch in range(1,201):
        for batch in range(50):
            steps.append(dict(epoch=epoch,step=(epoch-1)*50+batch+1,clean_ce=.1,total_loss=.1,
                learning_rate=.0002,gradient_norm=2.,source_samples=128 if batch<49 else 28,
                satellite_samples=0,**flags))
        epochs.append(dict(epoch=epoch,optimizer_steps=50,optimizer_steps_total=epoch*50,
            source_sample_exposure=6300,satellite_sample_exposure=0,clean_ce=.1,total_loss=.1,
            learning_rate=.0002,gradient_norm=2.,source_val_count=27000,source_val_accuracy=.9,
            source_val_rx_accuracy={r:.9 for r in ('1','3','4','6','8')},source_val_worst_rx=.9,
            coordinate_diagnostics=dict(time_gate_a_abs_mean=.1,time_gradient_norm=.3),
            elapsed_seconds=1.,peak_cuda_allocated_bytes=1234,**flags))
    compact=[{k:v for k,v in e.items() if not isinstance(v,dict)} for e in epochs]
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(compact[0]))
    writer.writeheader();writer.writerows(compact)
    text='RESOLVED_CONFIG {}\n'+'\n'.join('EPOCH '+json.dumps(e) for e in epochs)
    return epochs,steps,stream.getvalue(),text


def test_complete_full_budget_and_startup_warning(complete_logs):
    epochs,steps,csv_text,stdout=complete_logs
    warning='A module that was compiled using NumPy 1.x\nTraceback (most recent call last)\n'
    audit=audit_logs(epochs,steps,csv_text,warning+stdout,203193)
    assert audit['steps']==10000 and audit['epochs']==200
    assert audit['step_epoch_csv_stdout_reconciled'] and audit['startup_traceback_warning']


@pytest.mark.parametrize('kind',['missing_step','duplicate_step','wrong_exposure','extra_loss','inactive_synchronized','changed_gradient','wrong_epoch','wrong_average','wrong_csv','wrong_stdout','runtime_error'])
def test_corrupted_complete_artifacts_rejected(complete_logs,kind):
    epochs,steps,csv_text,stdout=copy.deepcopy(complete_logs)
    if kind=='missing_step':steps.pop(17)
    if kind=='duplicate_step':steps[17]['step']=steps[16]['step']
    if kind=='wrong_exposure':steps[49]['source_samples']=128
    if kind=='extra_loss':steps[9000]['extra_losses_active']=True
    if kind=='inactive_synchronized':steps[9000]['coordinate_active']=False
    if kind=='changed_gradient':steps[9000]['gradient_used_parameters']=164865
    if kind=='wrong_epoch':steps[9000]['epoch']=42
    if kind=='wrong_average':steps[9000]['clean_ce']=.2;steps[9000]['total_loss']=.2
    if kind=='wrong_csv':csv_text=csv_text.replace('203193','164865',1)
    if kind=='wrong_stdout':stdout=stdout.replace('"gradient_norm": 2.0','"gradient_norm": 3.0',1)
    if kind=='runtime_error':stdout+='\nTraceback (most recent call last)\n'
    with pytest.raises(ValueError):audit_logs(epochs,steps,csv_text,stdout,203193)


def test_full_fp32_all10000steps_require_actual_flag(complete_logs):
    epochs,steps,_,_=copy.deepcopy(complete_logs)
    for row in [*epochs,*steps]:row['cudnn_allow_tf32']=False
    compact=[{k:v for k,v in e.items() if not isinstance(v,dict)} for e in epochs]
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)
    stdout='RESOLVED_CONFIG {}\n'+'\n'.join('EPOCH '+json.dumps(e) for e in epochs)
    audit_logs(epochs,steps,stream.getvalue(),stdout,203193)
    steps[9274]['cudnn_allow_tf32']=True
    with pytest.raises(ValueError,match='Full FP32'):audit_logs(epochs,steps,stream.getvalue(),stdout,203193)
    steps[9274].pop('cudnn_allow_tf32')
    with pytest.raises(ValueError,match='Full FP32'):audit_logs(epochs,steps,stream.getvalue(),stdout,203193)


def test_complete_gauge_parameter_count_is_different_and_checked(complete_logs):
    epochs,steps,csv_text,stdout=copy.deepcopy(complete_logs)
    for rec in [*epochs,*steps]:rec['gradient_used_parameters']=164865
    csv_text=csv_text.replace('203193','164865');stdout=stdout.replace('203193','164865')
    audit_logs(epochs,steps,csv_text,stdout,164865)
    with pytest.raises(ValueError):audit_logs(epochs,steps,csv_text,stdout,203193)


def test_omitting_precision_from_every_record_is_not_a_legacy_exemption(complete_logs):
    epochs,steps,_,_=copy.deepcopy(complete_logs)
    for rec in [*epochs,*steps]:rec.pop('cudnn_allow_tf32')
    compact=[{k:v for k,v in e.items() if not isinstance(v,dict)} for e in epochs]
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)
    stdout='RESOLVED_CONFIG {}\n'+'\n'.join('EPOCH '+json.dumps(e) for e in epochs)
    with pytest.raises(ValueError,match='Full FP32'):audit_logs(epochs,steps,stream.getvalue(),stdout,203193)
