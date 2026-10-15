"""Fixed permissions, real identity gain and equal-loss routing qualification."""
import copy,json
from . import design as d
def main():
    c=d.request();c['analysis_code_commit']='a'*40;d.validate(c);n=0
    for key,value in [('target_access',True),('U_truth_read',True),('U_IQ_read',True),('V_IQ_read',True),('source_role','V'),('head_fit_updates',1),('original_default_head_joint',False),('source_row','BASE_SSL-s2026092701'),('states',['FINAL20']),('LR',1e-3),('method',dict(d.METHOD,tolerance=.01)),('analysis_code_commit','uncommitted')]:
        bad=copy.deepcopy(c);bad[key]=value
        try:d.validate(bad)
        except ValueError:n+=1
        else:raise AssertionError(key)
    records=[dict(state=s,ordinal=o,mode=m,auxiliary_gradient_norm=1.,auxiliary_F_gradient_cosine_main=.5,current_evidence=dict(ROUTED_GAP=dict(active_decisions=36,active_origin_packet_count=18))) for s in d.STATES for o in d.ORDINALS for m in d.MODES]
    q=dict(actual_F_difference_norm=2e-6,held_feature_RMS_difference=2e-5,shallow_middle_difference_tensors=1,held_full_CE_difference=-2e-5,held_positive_burden_difference=-2e-5,held_transfer_errors_increase=0,ACTUAL_CE_difference=-2e-5,PRE_W_CE_difference=-2e-5,ACTUAL_accuracy_difference=0.,PRE_W_accuracy_difference=0.)
    pairs=[dict(state=s,ordinal=o,auxiliary_F_cosine_JOINT=.5,**{'vs_'+control:copy.deepcopy(q) for control in d.GATE['controls']}) for s in d.STATES for o in d.ORDINALS]
    assert d.source_gate(records,pairs)['stage_signal']=='E20_PLUS20_SIGNAL'
    for control in d.GATE['controls']:
        for key in q:
            bad=copy.deepcopy(pairs)
            for p in bad:p['vs_'+control][key]=-1. if 'accuracy_difference' in key else 1. if key=='held_transfer_errors_increase' else 0.
            assert not d.source_gate(records,bad)['passed'];n+=1
    for field,value in [('auxiliary_gradient_norm',0.),('auxiliary_F_gradient_cosine_main',.999),('auxiliary_F_gradient_cosine_main',None),('active_decisions',35),('active_origin_packet_count',17)]:
        bad=copy.deepcopy(records)
        for r in bad:
            if field in ('auxiliary_gradient_norm','auxiliary_F_gradient_cosine_main'):r[field]=value
            else:r['current_evidence']['ROUTED_GAP'][field]=value
        assert not d.source_gate(bad,pairs)['passed'];n+=1
    rescale=copy.deepcopy(pairs)
    for p in rescale:p['auxiliary_F_cosine_JOINT']=.999
    assert not d.source_gate(records,rescale)['passed'];n+=1
    for rs,ps in [(records[:-1],pairs),(records,pairs[:-1])]:
        try:d.source_gate(rs,ps)
        except ValueError:n+=1
        else:raise AssertionError('incomplete matrix')
    early=copy.deepcopy(pairs)
    for p in early:
        if p['state']=='FINAL20':p['vs_ZERO']['held_full_CE_difference']=0.
    assert d.source_gate(records,early)['stage_signal']=='SCRATCH20_SIGNAL'
    print(json.dumps(dict(status='PASS',request_and_gate_negatives=n,both_controls_required=True,distinct_origin_coverage_required=True,actual_identity_and_burden_improvements_both_required=True,distinct_direction_against_main_and_JOINT_required=True,late_stage_priority_fixed=True,head_fit_updates=0,data_checkpoint_remote_reads=0,actual_optimizer_updates=0)))
if __name__=='__main__':main()
