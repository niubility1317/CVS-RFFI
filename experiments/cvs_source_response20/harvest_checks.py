"""Check analysis boundaries and arithmetic without touching IQ or truth."""
import copy
import json
from .train_harvest import ready,source_qualification,state_script


def fixture():
    rows={}
    for arm,acc in [('EMP',.988),('ROT',.987)]:
        rows[arm]=dict(status='SOURCE_DECISION_FROZEN',target_metrics_read=False,before_own_target_prediction=True,
            source_metrics=dict(source_val_accuracy=acc,source_val_worst_rx=.97),
            control=dict(source_metrics=dict(source_val_accuracy=.98,source_val_worst_rx=.97)),
            checks=dict(clean_gain=True,worst_RX_health=True),passed=True)
    return dict(status='SOURCE_QUALIFICATION_FROZEN',target_scores_read=False,rows=rows,passed=True,EMP_minus_ROT_source_clean_pp=.1)


def main():
    assert not ready(dict(complete=None))
    q=fixture();source_qualification(q)
    state=dict(complete=dict(status='ARTIFACTS_COMPLETE',actual_model_count=2,seed_count=1,truth_last=True),failure=None,qualification=q)
    assert ready(state)
    negatives=[]
    for field,value in [('target_scores_read',True),('passed',False),('EMP_minus_ROT_source_clean_pp',2.)]:
        changed=copy.deepcopy(q);changed[field]=value;negatives.append(changed)
    changed=copy.deepcopy(q);changed['rows']['EMP']['before_own_target_prediction']=False;negatives.append(changed)
    changed=copy.deepcopy(q);changed['rows']['EMP']['source_metrics']['source_val_accuracy']=.981;negatives.append(changed)
    for changed in negatives:
        try:source_qualification(changed)
        except ValueError:pass
        else:raise AssertionError('Invalid source boundary or qualification accepted')
    compile(state_script('/safe/fixed/run'),'generated_read_only_status','exec')
    print(json.dumps(dict(status='PASS',incomplete_not_ready=True,complete_source_gate=True,negative_cases=len(negatives),
        real_IQ_reads=0,checkpoint_loads=0,truth_reads=0,model_forwards=0)))


if __name__=='__main__':main()
