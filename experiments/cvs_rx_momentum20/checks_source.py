"""Focused query/data/scope negatives and fixed dual-control source rule."""
import copy,json
from . import design as d

def main():
    c=d.request();c['analysis_code_commit']='a'*40;d.validate(c);negatives=0
    for key,value in [('target_access',True),('U_truth_read',True),('U_IQ_read',True),('V_IQ_read',True),
        ('source_role','V'),('head_fit_updates',1),('original_default_head_joint',False),
        ('source_row','BASE_SSL-s2026092701'),('state','INITIAL'),('restart_lr',1e-3),
        ('current_gradient_preserved',False),('second_moment_and_step_preserved',False),
        ('rank_rule',dict(d.RANK_RULE,relative_eigenvalue_cutoff=1e-4)),('analysis_code_commit','uncommitted')]:
        bad=copy.deepcopy(c);bad[key]=value
        try:d.validate(bad)
        except ValueError:negatives+=1
        else:raise AssertionError(key)
    records=[dict(ordinal=o,regime=m,momentum_filter=dict(rank=4),
        held_raw_identity_margin_gain_from_before=2e-5,PRE_W_held_CE_reduction_from_before=0.,
        ACTUAL_held_CE_reduction_from_before=0.) for o in d.ORDINALS for m in d.REGIMES]
    q=dict(actual_F_difference_norm=2e-6,held_feature_RMS_difference=2e-5,
        raw_identity_margin_gain=2e-5,shallow_middle_difference_tensors=1,
        PRE_W_CE_difference=0.,ACTUAL_CE_difference=0.)
    pairs=[dict(ordinal=o,vs_KEEP=copy.deepcopy(q),vs_M_ZERO_F=copy.deepcopy(q)) for o in d.ORDINALS]
    assert d.source_gate(records,pairs)['passed']
    for key in q:
        bad=copy.deepcopy(pairs)
        for p in bad:p['vs_M_ZERO_F'][key]=1e-3 if key.endswith('CE_difference') else 0.
        assert not d.source_gate(records,bad)['passed'];negatives+=1
    for key in ['held_raw_identity_margin_gain_from_before','PRE_W_held_CE_reduction_from_before','ACTUAL_held_CE_reduction_from_before']:
        bad=copy.deepcopy(records)
        for r in bad:
            if r['regime']=='RX_M_FILTER':r[key]=-1e-3
        assert not d.source_gate(bad,pairs)['passed'];negatives+=1
    bad=copy.deepcopy(records)
    for r in bad:r['momentum_filter']['rank']=0
    assert not d.source_gate(bad,pairs)['passed'];negatives+=1
    for rs,ps in [(records[:-1],pairs),(records,pairs[:-1])]:
        try:d.source_gate(rs,ps)
        except ValueError:negatives+=1
        else:raise AssertionError('incomplete matrix')
    print(json.dumps(dict(status='PASS',request_and_gate_negatives=negatives,
        both_controls_required=True,head_readability_and_raw_geometry_required=True,
        data_checkpoint_remote_reads=0,actual_optimizer_updates=0)))

if __name__=='__main__':main()
