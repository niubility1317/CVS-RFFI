"""Focused permission, inheritance, absolute-pp and truth-closed checks."""
from copy import deepcopy
import tempfile
from pathlib import Path
from unittest.mock import patch
from . import design as d
from .protocol import final_metadata,checkpoint

def run():
    negatives=0
    def bad(fn):
        nonlocal negatives
        try:fn()
        except (ValueError,KeyError,TypeError):negatives+=1
        else:raise AssertionError('Invalid conditional-noise contract accepted')
    assert len(d.rows())==2
    configs=[d.config(r) for r in d.rows()]
    for c in configs:d.validate(c)
    assert configs[0]['parent_source']==configs[1]['parent_source']==d.old_config()['parent_source']
    assert configs[0]['parent_source']!=d.reference_config('B01')['parent_source']
    for key,value in dict(epochs=20,updates=4440,As=False,model_seed=2026092702,
            parent_source=d.reference_config('B01')['parent_source'],method='UNKNOWN').items():
        c=deepcopy(configs[1]);c[key]=value;bad(lambda c=c:d.validate(c))
    for key,value in dict(As_KL=True,original_dropout=False,same_physical_seed=False,
            same_hardware_object=False,total_As_CE_coefficient=.5,per_pass_As_CE_coefficient=.25,
            extra_NN=128,quality_rejection=True,U_changed=True,formal_test_changed=True,
            AGC='freeze first AGC',noise_energy='Exact perpacket energy normalization').items():
        c=deepcopy(configs[1]);c['noise_pair'][key]=value;bad(lambda c=c:d.validate(c))
    bad(lambda:d.config(d.rows()[1],20))
    c=configs[1];roles={'L':['l'],'U':['u'],'V':['v']}
    done=dict(config=c,epoch=10,optimizer_steps=2220,target_access=False,head_fit_updates=0)
    frozen=dict(config=c,selection='final_only',target_access=False)
    initial=dict(scratch_only=False,target_contact=False,checkpoint_sources=c['checkpoint_sources'],source_roles='EXACT_MATCH')
    final_metadata(c,done,frozen,initial,dict(role_ids=roles),roles)
    bad(lambda:final_metadata(c,dict(done,target_access=True),frozen,initial,dict(role_ids=roles),roles))
    bad(lambda:final_metadata(c,done,frozen,initial,dict(role_ids={'L':['other']}),roles))
    cp=dict(epoch=10,a1_ema_successful_updates=2220,run_id=c['run_id'],candidate_id=c['row_id'],
        checkpoint_selection='final_only',ema_model={},args=dict(baseline_ckpt=c['parent_source']+'/final_ssdg.pth',
        teacher_ckpt='',pseudo_temporal_mode='batch_neighbor',pseudo_temporal_gate=True))
    checkpoint(c,cp)
    bad(lambda:checkpoint(c,dict(cp,candidate_id='C01-s2026092701')))
    from . import report,evaluate
    names=('NOAS30','AS30','INDEP_FULL_CE6','SAME_NOISE_CE6','INDEP_NOISE_CE6')
    with tempfile.TemporaryDirectory() as temp:
        roots={n:Path(temp)/n for n in names}
        base=Path(temp)/'decision';base.mkdir()
        def populate(clean=.979,midlow=.725,old=.70,same=.72):
            for name,r in roots.items():
                cfg=report.cfg(name);v=.98 if name=='NOAS30' else clean
                value=midlow if name=='INDEP_NOISE_CE6' else same if name=='SAME_NOISE_CE6' else old
                d.write(r/'source/completion.json',dict(config=cfg,target_access=False,
                    final_source_metrics=dict(source_val_count=27000,source_val_accuracy=v)))
                d.write(r/'source_qualification/complete.json',dict(config=cfg,target_access=False,
                    formal_scores_read=False,training_updates=0,NN=12600,source_accuracies=[[.8,value,value]]*2))
        with patch.object(d,'BASE',base),patch.object(report,'roots',lambda:roots):
            populate();assert report.source_decision()['source_qualified']
            populate(clean=.97899);assert not report.source_decision()['source_qualified']
            populate(midlow=.72499);assert not report.source_decision()['source_qualified']
            populate(midlow=.731,old=.735);assert not report.source_decision()['source_qualified']
            (roots[names[-1]]/'source_qualification/complete.json').unlink()
            assert report.source_decision()['status']=='SOURCE_READOUT_PARTIAL'
        with patch.object(d,'BASE',base):
            try:evaluate.score(d.rows()[0]['row_id'])
            except FileNotFoundError:pass
            else:raise AssertionError('Missing prediction must keep truth closed')
            assert not (base/d.rows()[0]['row_id']/'scoring').exists()
    return dict(status='PASS',protocol_groups=6,negative_cases=negatives,source_criteria_cases=5,
        missing_prediction_truth_closed=True,NN=0,IQ=0,CP=0,training_updates=0)
