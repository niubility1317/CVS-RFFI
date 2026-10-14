"""Focused fixed-stage request and interpretation negatives."""
import copy
import json
from . import design as d


def main():
    c=d.request();c['analysis_code_commit']='0'*40;d.validate(c);negatives=0
    for key,value in [('source_role','U_s'),('state','INITIAL'),('restart_lr',1e-3),('new_loss',True),
        ('head_fit_updates',1),('original_default_head_joint',False),('U_IQ_read',True),('V_IQ_read',True),
        ('U_truth_read',True),('target_access',True),('PRE_W_gains','pre-step fixed'),('source_row','BASE_SSL-s2026092701')]:
        bad=copy.deepcopy(c);bad[key]=value
        try:d.validate(bad)
        except ValueError:negatives+=1
        else:raise AssertionError('Invalid stage request accepted: '+key)
    good=dict(actual_F_step_norm=.01,held_feature_RMS_from_before=.001,shallow_middle_tensors_changed_from_before=3,
        PRE_W_held_CE_reduction_from_before=1e-4,held_raw_identity_margin_gain_from_before=1e-3)
    rows=[dict(good,ordinal=o,regime=m) for o in d.ORDINALS for m in d.REGIMES]
    if not d.source_gate(rows)['passed']:raise AssertionError('Sufficient reset stage rejected')
    for key,value in [('actual_F_step_norm',0.),('held_feature_RMS_from_before',0.),
        ('shallow_middle_tensors_changed_from_before',0),('PRE_W_held_CE_reduction_from_before',0.),
        ('held_raw_identity_margin_gain_from_before',0.)]:
        bad=copy.deepcopy(rows)
        for row in bad:
            if row['regime']=='RESTART_RESET':row[key]=value
        if d.source_gate(bad)['passed']:raise AssertionError('KEEP/END or no feature signal replaced intended reset')
        negatives+=1
    try:d.source_gate(rows[:-1])
    except ValueError:negatives+=1
    else:raise AssertionError('Incomplete stage matrix promoted')
    print(json.dumps(dict(status='PASS',focused_negatives=negatives,reset_not_KEEP_or_END_substitution=True,
        head_only_or_no_feature_change_not_gate=True,after_gains_required=True,
        dataset_checkpoint_remote_reads=0,actual_optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
