"""Focused source request and gate boundaries; no data/checkpoint/runtime."""
import copy
import json
from . import design as d


def main():
    c=d.request();c['analysis_code_commit']='0'*40;d.validate(c)
    for key,value in [('source_role','V'),('U_truth_read',True),('V_gradient',True),('target_access',True),('risk_ordinals',[0,1])]:
        bad=copy.deepcopy(c);bad[key]=value
        try:d.validate(bad)
        except ValueError:pass
        else:raise AssertionError('Illegal fixed source request accepted: '+key)
    rows=[]
    for state in d.STATES:
        for o in d.ORDINALS:
            rows.append(dict(state=state,ordinal=o,RAW_positive_directional_risk_RX=1,
                UPDATE_RAW_relative_displacement_difference=.1,UPDATE_retained_actual_displacement_ratio=.8,
                UPDATE_RAW_held_worst_RX_CE_delta_reduction=2e-6,UPDATE_RAW_held_mean_CE_increase=-2e-6,
                UPDATE_RAW_held_feature_RMS=2e-7,shallow_middle_tensors_changed_from_RAW=1,
                actual_risk_within_solver_and_rounding_bound=True,UPDATE_GRAD_relative_displacement_difference=.1,
                modes=dict(UPDATE_CONE=dict(actual_displacement=dict(actual_parameter_addition_measured=True)))))
    assert d.source_gate(rows)['passed']
    weak=copy.deepcopy(rows)
    for r in weak:r['UPDATE_retained_actual_displacement_ratio']=0.
    assert not d.source_gate(weak)['passed'] # Killing all movement cannot qualify.
    for r in weak:r['UPDATE_retained_actual_displacement_ratio']=.8;r['UPDATE_RAW_held_worst_RX_CE_delta_reduction']=0.
    assert not d.source_gate(weak)['passed'] # A feasible risk-only change cannot qualify.
    try:d.source_gate(rows[:-1])
    except ValueError:pass
    else:raise AssertionError('Incomplete matrix accepted')
    print(json.dumps(dict(status='PASS',fixed_role_query_truth_negatives=True,zero_update_not_useful=True,
        risk_only_not_useful=True,full_matrix_required=True,data_or_checkpoint_reads=0,actual_optimizer_updates=0)))


if __name__=='__main__':main()
