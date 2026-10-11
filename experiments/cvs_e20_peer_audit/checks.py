"""Bounded P0-only synthetic pairing, calibration, and permission checks."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch
from .audit import analyze,select_p0,SEEDS,_metrics


def fixture():
    predictions=[[0,0,2,2,4,0],[1,1,2,3,0,0],[0,1,2,3,0,0]]
    confidence=[[.99,.99,.7,.7,.99,.99],[.99]*6,[.99]*6]
    result={}
    for seed,preds,conf in zip(SEEDS,predictions,confidence):
        probabilities=torch.tensor([[(c if k==p else (1-c)/5) for k in range(6)] for p,c in zip(preds,conf)],dtype=torch.float64)
        result[seed]={role:dict(role=role,ids=[f'{role}-{i}' for i in range(6)],probabilities=probabilities.clone(),
            y=torch.arange(6),rx=torch.tensor([1,1,3,3,4,4]),day=torch.tensor([0,1,0,1,0,1])) for role in ('L','V')}
    return result


def run():
    torch.set_num_threads(2);inputs=fixture()
    class ParentOnly(dict):
        def __getitem__(self,key):
            if key=='endpoints':raise AssertionError('Endpoint variants were inspected')
            return super().__getitem__(key)
    selected=select_p0({role:ParentOnly(parent=inputs[SEEDS[0]][role],endpoints=object()) for role in ('L','V')})
    assert selected['L'] is inputs[SEEDS[0]]['L']
    before=deepcopy(inputs);rng=torch.random.get_rng_state().clone();result=analyze(inputs)
    json.dumps(result,allow_nan=False)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(inputs[s][r]['probabilities'],before[s][r]['probabilities']) for s in SEEDS for r in ('L','V'))
    loo=result['roles']['V']['students'][str(SEEDS[0])]['LOO2']
    p=loo['paired']['overall'];overall=p['overall']
    assert overall['fixed']==2 and overall['broken']==1 and overall['persistent_error']==1
    assert overall['high_confidence_original_errors_fixed']==1
    assert {k:v['summary']['count'] for k,v in p['student_initial_cohorts'].items()}==dict(high_correct=2,high_error=2,low_correct=1,low_error=1)
    assert p['selections']['peer_consensus']['count']==5
    assert p['selections']['teacher_q095']['count']==5
    assert p['selections']['peer_consensus_and_changed']['count']==3
    assert p['selections']['peer_consensus_and_changed']['teacher_accuracy_when_changed']==2/3
    assert p['selections']['peer_consensus_and_changed']['selection_risk']==1/3
    overlap=result['roles']['V']['error_overlap']['overall']
    assert overlap['all_three_wrong']==1 and overlap['joint_oracle_accuracy_upper_bound']==5/6
    assert result['roles']['V']['ALL3']['teacher_count']==1
    assert loo['teacher_seeds']==list(SEEDS[1:])
    bad_cases=[]
    def reject(name,mutate):
        bad=deepcopy(inputs);mutate(bad)
        try:analyze(bad)
        except (ValueError,KeyError,TypeError):bad_cases.append(name)
        else:raise AssertionError('Invalid input accepted: '+name)
    reject('PID order',lambda x:x[SEEDS[1]]['V']['ids'].reverse())
    reject('duplicate PID',lambda x:x[SEEDS[0]]['L']['ids'].__setitem__(1,'L-0'))
    reject('role contamination',lambda x:x[SEEDS[0]]['L'].__setitem__('role','U_s'))
    reject('extra source role',lambda x:x[SEEDS[0]].__setitem__('U',{}))
    reject('L V overlap',lambda x:x[SEEDS[0]]['V'].__setitem__('ids',list(x[SEEDS[0]]['L']['ids'])))
    reject('truth mismatch',lambda x:x[SEEDS[1]]['V']['y'].__setitem__(0,1))
    reject('RX mismatch',lambda x:x[SEEDS[1]]['V']['rx'].__setitem__(0,8))
    reject('day mismatch',lambda x:x[SEEDS[1]]['V']['day'].__setitem__(0,2))
    reject('NaN probability',lambda x:x[SEEDS[0]]['L']['probabilities'].__setitem__((0,0),float('nan')))
    reject('negative probability',lambda x:x[SEEDS[0]]['L']['probabilities'].__setitem__((0,0),-.1))
    reject('not simplex',lambda x:x[SEEDS[0]]['L']['probabilities'].__setitem__((0,0),.8))
    reject('wrong shape',lambda x:x[SEEDS[0]]['L'].__setitem__('probabilities',torch.ones(6,5)/5))
    reject('missing seed',lambda x:x.pop(SEEDS[-1]))
    # Exact-zero probabilities have finite NLL, no fitting or renormalization.
    q=torch.eye(6,dtype=torch.float64);m=_metrics(q,torch.tensor([1,1,2,3,4,5]))
    assert m['NLL']>0 and sum(v['count'] for v in m['ECE15_bins'])==6 and m['ECE15_bins'][-1]['count']==6
    empty=fixture()
    for s in SEEDS:
        for role in ('L','V'):empty[s][role]['probabilities']=q.clone()
    e=analyze(empty)['roles']['L']['students'][str(SEEDS[0])]['LOO2']['paired']['overall']
    assert e['student_initial_cohorts']['high_error']['summary']['count']==0
    assert e['student_initial_cohorts']['high_error']['summary']['teacher']['accuracy'] is None
    assert e['selections']['teacher_changed']['teacher_accuracy_when_changed'] is None
    json.dumps(e,allow_nan=False)
    return dict(status='PASS',source_only=True,target_access=False,endpoint_poison_not_accessed=True,
        all_three_seed_P0_pairing=True,LOO_excludes_student=True,ALL3_single_teacher=True,
        repair_break_and_high_confidence_error_counts=True,consensus_both_peers_q095=True,
        accepted_disagreement_risk=True,pairwise_overlap_and_oracle=True,ECE15_and_zero_probability_NLL=True,
        empty_cohorts_JSON_null=True,input_and_RNG_unchanged=True,rejected_cases=bad_cases)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(result))
