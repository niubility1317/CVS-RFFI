"""Focused identity-only source permission and fixed control negatives."""
import copy,json
from . import design as d

def main():
    c=d.request();c['analysis_code_commit']='a'*40;d.validate(c);negatives=0
    for key,value in [('target_access',True),('U_truth_read',True),('U_IQ_read',True),('V_IQ_read',True),
        ('source_role','V'),('head_fit_updates',1),('original_default_head_joint',False),
        ('source_row','BASE_SSL-s2026092701'),('states',['INITIAL']),('LR',1e-3),
        ('method',dict(d.METHOD,Sinkhorn_iterations=80)),('analysis_code_commit','uncommitted')]:
        bad=copy.deepcopy(c);bad[key]=value
        try:d.validate(bad)
        except ValueError:negatives+=1
        else:raise AssertionError(key)
    records=[dict(state=s,ordinal=o,mode=m,auxiliary_gradient_norm=1.,
        current_matching=dict(CROSS_ROW=dict(active_rows_and_columns=1))) for s in d.STATES for o in d.ORDINALS for m in d.MODES]
    q=dict(actual_F_difference_norm=2e-6,held_feature_RMS_difference=2e-5,
        shallow_middle_difference_tensors=1,held_correct_matching_mass_gain=2e-4,
        held_matching_errors_increase=0,ACTUAL_CE_difference=0.,PRE_W_CE_difference=0.,ACTUAL_accuracy_difference=0.,PRE_W_accuracy_difference=0.)
    pairs=[dict(state=s,ordinal=o,**{'vs_'+control:copy.deepcopy(q) for control in d.GATE['controls']}) for s in d.STATES for o in d.ORDINALS]
    assert d.source_gate(records,pairs)['stage_signal']=='E20_PLUS20_SIGNAL'
    for control in d.GATE['controls']:
        for key in q:
            bad=copy.deepcopy(pairs)
            for p in bad:
                p['vs_'+control][key]=1 if key in ('held_matching_errors_increase','ACTUAL_CE_difference','PRE_W_CE_difference') else -1. if 'accuracy_difference' in key else 0.
            assert not d.source_gate(records,bad)['passed'];negatives+=1
    bad=copy.deepcopy(records)
    for r in bad:r['auxiliary_gradient_norm']=0.
    assert not d.source_gate(bad,pairs)['passed'];negatives+=1
    for rs,ps in [(records[:-1],pairs),(records,pairs[:-1])]:
        try:d.source_gate(rs,ps)
        except ValueError:negatives+=1
        else:raise AssertionError('incomplete source matrix')
    print(json.dumps(dict(status='PASS',request_and_control_negatives=negatives,
        both_zero_and_same_ROW_controls_required=True,no_control_substitution=True,budget_and_class_head_fixed=True,
        method_specific_matching_signal_not_rawcos_proxy=True,data_checkpoint_remote_reads=0,actual_optimizer_updates=0)))

if __name__=='__main__':main()
