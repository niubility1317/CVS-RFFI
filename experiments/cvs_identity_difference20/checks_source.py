"""Focused source request and finite-step gate negatives, without data."""
import copy
import json
from . import design as d


def main():
    request=d.request();request['analysis_code_commit']='0'*40;d.validate(request)
    count=0
    for key,value in [('source_role','U_s'),('U_IQ_read',True),('U_truth_read',True),('V_IQ_read',True),
        ('target_access',True),('coefficient',.2),('direction_stop',.99),('source_row','BASE_SSL-s2026092701'),
        ('held_ordinals',[0,1]),('states',['FINAL20']),('current_pairs',[[0,0],[17,18],[35,36],[52,53]])]:
        bad=copy.deepcopy(request);bad[key]=value
        try:d.validate(bad)
        except ValueError:count+=1
        else:raise AssertionError('Invalid fixed request accepted: '+key)
    good=dict(CROSS_active_direction_pairs=300,minimum_class_difference_norm=.01,
        CROSS_SELF_auxiliary_gradient_relative_difference=.1,CROSS_ZERO_actual_F_displacement_relative_difference=.1,
        CROSS_ZERO_held_feature_RMS=1e-5,CROSS_ZERO_shallow_middle_tensors_changed=3,
        CROSS_ZERO_held_identity_margin_gain=1e-4,CROSS_SELF_held_identity_margin_gain=1e-4,
        CROSS_ZERO_held_mean_CE_increase=0.,CROSS_SELF_held_mean_CE_increase=0.)
    rows=[dict(good,state=s,ordinal=o) for s in d.STATES for o in d.ORDINALS]
    if not d.source_gate(rows)['passed']:raise AssertionError('Fixed sufficient signal rejected')
    initial_only=copy.deepcopy(rows)
    for row in initial_only:
        if row['state']=='FINAL20':row['CROSS_ZERO_held_identity_margin_gain']=0.
    if d.source_gate(initial_only)['passed']:raise AssertionError('Initial hypothesis substituted for final20 gate')
    for field,value in [('CROSS_active_direction_pairs',0),('minimum_class_difference_norm',0.),
        ('CROSS_ZERO_actual_F_displacement_relative_difference',0.),('CROSS_SELF_auxiliary_gradient_relative_difference',0.),
        ('CROSS_ZERO_held_feature_RMS',0.),('CROSS_ZERO_shallow_middle_tensors_changed',0),
        ('CROSS_ZERO_held_identity_margin_gain',0.),('CROSS_SELF_held_identity_margin_gain',0.),
        ('CROSS_ZERO_held_mean_CE_increase',2e-6),('CROSS_SELF_held_mean_CE_increase',2e-6)]:
        bad=copy.deepcopy(rows)
        for row in bad:row[field]=value
        if d.source_gate(bad)['passed']:raise AssertionError('Scientific false utility promoted: '+field)
        count+=1
    try:d.source_gate(rows[:-1])
    except ValueError:count+=1
    else:raise AssertionError('Incomplete source matrix promoted')
    print(json.dumps(dict(status='PASS',focused_negatives=count,INITIAL_not_FINAL_substitute=True,
        zero_aux_update_or_feature_difference_not_useful=True,SELF_control_and_original_CE_both_required=True,
        aux_loss_reduction_alone_not_gate=True,data_checkpoint_remote_reads=0,actual_optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
