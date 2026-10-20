import copy
import numpy as np
import pytest

from tools.analyze_pamf_r17_compact_results import classification_record,summary_metrics,compare,training_metrics
from experiments.pamf_r17_compact import design as d


def record(matrix,**fields):
    matrix=np.asarray(matrix,np.int64);n=int(matrix.sum());correct=int(matrix.trace())
    precision=np.divide(matrix.diagonal(),matrix.sum(0),out=np.zeros(6),where=matrix.sum(0)>0)
    recall=np.divide(matrix.diagonal(),matrix.sum(1),out=np.zeros(6),where=matrix.sum(1)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(6),where=precision+recall>0)
    return dict(confusion=matrix.tolist(),count=n,accuracy=100*correct/n,macro_f1=100*float(f1.mean()),
                baseline_accuracy=100.,repaired=0,damaged=n-correct,net_repaired=correct-n,readout='restored',**fields)


def summary(role='source',errors=180):
    n=1800 if role=='source' else 168000;records=[]
    for view in d.VIEWS:
        matrix=np.eye(6,dtype=np.int64)*(n//6);matrix[0,0]-=errors;matrix[0,1]+=errors
        records.append(record(matrix,view=view,dimension='overall',value='all'))
        for label in range(6):
            part=np.zeros_like(matrix);part[label]=matrix[label]
            records.append(record(part,view=view,dimension='TX',value=str(label)))
        for dim,groups in (('RX',5),('day',2)):
            for index in range(groups):
                part=matrix//groups+(matrix%groups>index)
                records.append(record(part,view=view,dimension=dim,value=str(index)))
    return dict(status='ANALYZED',role=role,count=n,F_seed=2026092701,original_default_head=True,
                all_predictions_before_truth=True,training_F_access=False,training_TX_labels_read=0,head_fit_updates=0,
                records=records,views={v:dict(restored=dict(waveform=dict(iq_nmse=.02,prefix232_nmse=.02,tail24_nmse=.02,fft_log_power_L1=.1))) for v in d.VIEWS})


def test_recomputes_every_partition_and_detects_one_target_error_regression():
    baseline=summary_metrics(summary('target',errors=100),'target')
    candidate=summary_metrics(summary('target',errors=101),'target')
    result=compare(candidate,baseline)
    assert not result['accuracy_nonregression']
    assert all(v['correct_delta']==-1 and v['accuracy_delta_pp']==-100/168000 for v in result['views'].values())
    assert baseline['independently_recomputed_records']==7*(1+6+5+2)
    assert compare(baseline,baseline)['accuracy_nonregression']


@pytest.mark.parametrize('key,bad',[('accuracy',99.),('macro_f1',99.),('net_repaired',0),('count',1)])
def test_inconsistent_aggregate_arithmetic_rejected(key,bad):
    value=summary()['records'][0];value[key]=bad
    with pytest.raises(ValueError):classification_record(value)


def test_incomplete_or_mismatched_physical_partitions_rejected():
    value=summary();value['records']=[r for r in value['records'] if not(r['view']=='clean' and r['dimension']=='RX' and r['value']=='0')]
    with pytest.raises(ValueError):summary_metrics(value,'source')
    with pytest.raises(ValueError):summary_metrics(summary('target'),'source')


def test_training_analysis_requires_all_fixed_records_and_does_not_select_lowest_loss():
    rows=[dict(epoch=e,updates=247*e,L_exposures=6300*e,U_exposures=56700*e,teacher_IQ_examples=63000*e,
               seconds=100.*e,clip_fraction=0.,losses=dict(loss=.1+.01/e,iq_nmse=.1,teacher_stage_nmse=.001/e)) for e in range(1,21)]
    complete=dict(status='SOURCE_TRAINED',arm='shared_mean',epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        teacher_IQ_examples=1260000,inherited_prior_epochs=20,cumulative_prior_epochs=40,cumulative_serial_training_passes=80,
        parameters=d.PARAMETERS['shared_mean'],trainable_parameters=d.PARAMETERS['shared_mean']-2157547,
        teacher_parameter_updates=0,frozen_context_updates=0,TX_labels_read=0,F_outputs_read=0,F_weights_read=0,V_reads=0,target_reads=0)
    result=training_metrics(rows,complete,'shared_mean')
    assert result['full_log_records']==20 and result['checkpoint_selection'].startswith('fixedfinalE20')
    with pytest.raises(ValueError):training_metrics(rows[:-1],complete,'shared_mean')
    invalid=copy.deepcopy(rows);invalid[-1]['U_exposures']-=1
    with pytest.raises(ValueError):training_metrics(invalid,complete,'shared_mean')
