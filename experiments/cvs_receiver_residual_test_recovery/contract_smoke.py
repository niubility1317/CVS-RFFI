"""Regression: legacy contract has no classes, class order still cannot drift."""
import copy
import json
from experiments.cvs_receiver_residual_test_recovery.contract import validate_source_contract


def main():
    expected=dict(role_ids=dict(L_s=['L'],U_s=['U'],V=['V']),source_rxs=[1,3,4,6,8],source_days=[1,2,3],
        ratios=[.07,.63,.30],split_seed=392005,num_classes=6,checkpoint_init='scratch_only',target_access_before_freeze=False)
    classes=['14-10','14-7','20-15','20-19','6-15','8-20'];dataset='/source/ManySig.pkl'
    actual=dict(expected,classes=classes,dataset_path=dataset,physical_roles='EXACT_MATCH',equalized=1,out_len=256,normalize=True)
    try:
        for k in ('role_ids','source_rxs','source_days','ratios','split_seed','classes'):
            if actual[k]!=expected[k]:raise ValueError(k)
    except KeyError as e:assert e.args==('classes',)
    else:raise AssertionError('Legacy failure not reproduced')
    assert validate_source_contract(actual,expected,classes,dataset)['status']=='VERIFIED'
    mutations=[dict(classes=list(reversed(classes))),dict(num_classes=5),dict(equalized=0),dict(normalize=False),
        dict(role_ids=dict(L_s=['target'],U_s=['U'],V=['V'])),dict(dataset_path='/other'),dict(source_rxs=[0,1]),
        dict(target_access_before_freeze=True),dict(checkpoint_init='unverified')]
    for mutation in mutations:
        bad=copy.deepcopy(actual);bad.update(mutation)
        try:validate_source_contract(bad,expected,classes,dataset)
        except ValueError:pass
        else:raise AssertionError('Mismatch accepted: '+repr(mutation))
    incomplete=dict(expected);del incomplete['role_ids']
    try:validate_source_contract(actual,incomplete,classes,dataset)
    except ValueError:pass
    else:raise AssertionError('Incomplete original roles accepted')
    print(json.dumps(dict(status='PASS',legacy_failure_reproduced='KeyError(classes)',
        checks=['all_original_contract_fields','physical_roles','ordered_class_map','received_IQ','scratch_provenance'],rejected_cases=len(mutations)+1)))


if __name__=='__main__':main()
