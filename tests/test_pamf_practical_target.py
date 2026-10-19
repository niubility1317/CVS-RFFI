import json
import numpy as np
import pytest
from experiments.pamf_feature_refine.practical_target import completed_predictions,waveform_gap,SCENES,SEEDS,READOUTS


def packet(tmp_path):
    saved=tmp_path/'prediction';saved.mkdir();ids=np.array(['a','b'])
    done=dict(status='PREDICTIONS_COMPLETE',views=list(SCENES),count=2,F_seeds=list(SEEDS),readouts=list(READOUTS),
        TX_truth_reads=0,paired_clean_reads=0,state_updates=0)
    (saved/'complete.json').write_text(json.dumps(done))
    values=dict(ids=ids,seeds=np.array(SEEDS),**{name:np.zeros((2,3),np.uint8) for name in READOUTS})
    for view in SCENES:
        np.savez(saved/(view+'.npz'),**values);np.save(saved/(view+'_refined_iq.npy'),np.zeros((2,2,256),np.float32))
    return saved,ids,done,values


def test_all_six_complete_before_scoring(tmp_path):
    _,ids,_,_=packet(tmp_path);assert set(completed_predictions(tmp_path,ids))==set(SCENES)


@pytest.mark.parametrize('problem',['missingview','earlytruth','earlyclean','stateupdate','missingseed','badclass','ids','missingIQ'])
def test_illegal_prediction_rejected_before_paired_clean_or_truth(tmp_path,problem):
    saved,ids,done,values=packet(tmp_path)
    if problem=='missingview':done['views']=list(SCENES[:-1])
    elif problem=='earlytruth':done['TX_truth_reads']=1
    elif problem=='earlyclean':done['paired_clean_reads']=1
    elif problem=='stateupdate':done['state_updates']=1
    elif problem=='missingseed':values['seeds']=np.array(SEEDS[:-1])
    elif problem=='badclass':values['refined'][0,0]=6
    elif problem=='ids':values['ids']=ids[::-1]
    else:np.save(saved/(SCENES[-1]+'_refined_iq.npy'),np.zeros((1,2,256),np.float32))
    (saved/'complete.json').write_text(json.dumps(done));np.savez(saved/(SCENES[-1]+'.npz'),**values)
    with pytest.raises(ValueError):completed_predictions(tmp_path,ids)


def test_waveform_gap_is_unaligned_and_retains_missing_tail():
    x=np.ones((2,2,256),np.float32);e=x.copy();e[:,:,232:]=0
    r=waveform_gap(e,x,batch=1)
    assert r['full_nmse']['mean']==24/256
    assert r['prefix232_nmse']['mean']==0 and r['tail24_nmse']['mean']==1
    assert waveform_gap(x,x)['full_nmse']['mean']==0
    assert waveform_gap(-x,x)['full_nmse']['mean']==4
