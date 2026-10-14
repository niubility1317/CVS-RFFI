"""Fixed source legality, distinct-opportunity and two-state qualification."""
import copy,json
from . import design as d
def main():
    c=d.request();c['analysis_code_commit']='a'*40;d.validate(c);n=0
    for key,value in [('target_access',True),('U_truth_read',True),('U_IQ_read',True),('V_IQ_read',True),('source_role','V'),('head_fit_updates',1),('original_default_head_joint',False),('source_row','BASE_SSL-s2026092701'),('states',['FINAL20']),('LR',1e-3),('method',dict(d.METHOD,stop_probability=.99)),('analysis_code_commit','uncommitted')]:
        bad=copy.deepcopy(c);bad[key]=value
        try:d.validate(bad)
        except ValueError:n+=1
        else:raise AssertionError(key)
    records=[dict(state=s,ordinal=o,mode=m,auxiliary_gradient_norm=1.,current_transfer=dict(CROSS_RX=dict(active_decisions=180,active_origin_packet_count=18))) for s in d.STATES for o in d.ORDINALS for m in d.MODES]
    q=dict(actual_F_difference_norm=2e-6,held_feature_RMS_difference=2e-5,shallow_middle_difference_tensors=1,held_transfer_CE_difference=-2e-4,held_transfer_errors_increase=0,ACTUAL_CE_relative_difference=0.,PRE_W_CE_relative_difference=0.,ACTUAL_accuracy_difference=0.,PRE_W_accuracy_difference=0.)
    pairs=[dict(state=s,ordinal=o,**{'vs_'+control:copy.deepcopy(q) for control in d.GATE['controls']}) for s in d.STATES for o in d.ORDINALS]
    assert d.source_gate(records,pairs)['stage_signal']=='SCRATCH20_SIGNAL'
    for control in d.GATE['controls']:
        for key in q:
            bad=copy.deepcopy(pairs)
            for p in bad:p['vs_'+control][key]=-1. if 'accuracy_difference' in key else 1. if key in ('held_transfer_errors_increase','ACTUAL_CE_relative_difference','PRE_W_CE_relative_difference') else 0.
            assert not d.source_gate(records,bad)['passed'];n+=1
    for field,value in [('auxiliary_gradient_norm',0.),('active_decisions',179),('active_origin_packet_count',17)]:
        bad=copy.deepcopy(records)
        for r in bad:
            if field=='auxiliary_gradient_norm':r[field]=value
            else:r['current_transfer']['CROSS_RX'][field]=value
        assert not d.source_gate(bad,pairs)['passed'];n+=1
    for rs,ps in [(records[:-1],pairs),(records,pairs[:-1])]:
        try:d.source_gate(rs,ps)
        except ValueError:n+=1
        else:raise AssertionError('incomplete matrix')
    late=copy.deepcopy(pairs)
    for p in late:
        if p['state']=='INITIAL':p['vs_ZERO']['held_transfer_CE_difference']=0.
    assert d.source_gate(records,late)['stage_signal']=='E20_PLUS20_SIGNAL'
    edge=copy.deepcopy(pairs)
    for p in edge:
        for control in d.GATE['controls']:p['vs_'+control]['ACTUAL_CE_relative_difference']=.01
    assert d.source_gate(records,edge)['passed']
    print(json.dumps(dict(status='PASS',request_and_gate_negatives=n,both_controls_required=True,distinct_origin_coverage_required=True,relative_original_CE_guard_fixed_before_data=True,early_stage_priority_and_late_fallback_prespecified=True,head_fit_updates=0,data_checkpoint_remote_reads=0,actual_optimizer_updates=0)))
if __name__=='__main__':main()
