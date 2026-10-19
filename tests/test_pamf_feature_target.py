import json
import numpy as np
import pytest
from experiments.pamf_feature_refine.target import completed_predictions,SEEDS,READOUTS,source_check
from experiments.pamf_feature_refine import design as d


def packet(tmp_path):
    saved=tmp_path/'prediction';saved.mkdir()
    done=dict(status='PREDICTIONS_COMPLETE',views=['clean'],count=2,F_seeds=list(SEEDS),readouts=list(READOUTS),TX_truth_reads=0,state_updates=0)
    (saved/'complete.json').write_text(json.dumps(done))
    values=dict(ids=np.array(['a','b']),seeds=np.array(SEEDS),**{k:np.zeros((2,3),np.uint8) for k in READOUTS},
        parent_change=np.zeros(2),refined_change=np.zeros(2))
    np.savez(saved/'clean.npz',**values);np.save(saved/'clean_refined_iq.npy',np.zeros((2,2,256),np.float32))
    return saved,done,values


def test_all_original_heads_have_complete_predictions(tmp_path):
    packet(tmp_path);assert completed_predictions(tmp_path,2)['identity'].shape==(2,3)


@pytest.mark.parametrize('problem',['missingseed','badclass','badshape','earlytruth','missingIQ'])
def test_invalid_target_packets_fail_before_truth(tmp_path,problem):
    saved,done,values=packet(tmp_path)
    if problem=='missingseed':done['F_seeds']=list(SEEDS[:1])
    elif problem=='badclass':values['refined'][0,0]=6
    elif problem=='badshape':values['parent_R03']=np.zeros((1,3),np.uint8)
    elif problem=='earlytruth':done['TX_truth_reads']=1
    else:np.save(saved/'clean_refined_iq.npy',np.zeros((1,2,256),np.float32))
    (saved/'complete.json').write_text(json.dumps(done));np.savez(saved/'clean.npz',**values)
    with pytest.raises(ValueError):completed_predictions(tmp_path,2)


def test_unqualified_source_cannot_expand_to_three_F_heads(tmp_path):
    root=tmp_path/d.RUN;root.mkdir()
    (root/'launch.json').write_text(json.dumps(dict(commit='d896e0552f1f17e7b421fb7d94d45af9256899b4')))
    (root/'complete.json').write_text(json.dumps(dict(status='ANALYZED',arms=list(d.ARMS),target_reads=0,source_only=True)))
    (root/'source_decision.json').write_text(json.dumps(dict(qualified=False,target_reads=0)))
    with pytest.raises(ValueError):source_check(root)
