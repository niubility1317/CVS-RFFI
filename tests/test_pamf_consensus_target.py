import json
import numpy as np
import pytest
from experiments.pamf_consensus_target.runner import completed_predictions,MODES,reference_handles


def valid_packet(tmp_path):
    p=tmp_path/'prediction';p.mkdir()
    (p/'complete.json').write_text(json.dumps(dict(status='PREDICTIONS_COMPLETE',views=['clean'],count=2,
        readouts=list(MODES),TX_truth_reads=0,state_updates=0)))
    values=dict(ids=np.array(['a','b']),**{k:np.array([0,1],np.uint8) for k in MODES},
        keep=np.array([True,False]),change=np.array([0.,1.]),damage_logit=np.array([-1.,1.]))
    np.savez(p/'clean.npz',**values);return p,values


def test_complete_clean_prediction_contract(tmp_path):
    valid_packet(tmp_path);assert completed_predictions(tmp_path,2)['ids'].tolist()==['a','b']


@pytest.mark.parametrize('problem',['incomplete','badclass','nonfinite','shape'])
def test_predictions_rejected_before_truth(tmp_path,problem):
    p,values=valid_packet(tmp_path)
    if problem=='incomplete':
        (p/'complete.json').write_text(json.dumps(dict(status='RUNNING')))
    else:
        if problem=='badclass':values['consensus'][0]=6
        elif problem=='nonfinite':values['change'][0]=np.nan
        else:values['keep']=np.ones(1,bool)
        np.savez(p/'clean.npz',**values)
    with pytest.raises(ValueError):completed_predictions(tmp_path,2)


def test_wrong_reference_run_rejected_without_iq_read(tmp_path):
    (tmp_path/'residual_deterministic/prediction').mkdir(parents=True);(tmp_path/'reference').mkdir()
    for path in ('launch.json','complete.json','residual_deterministic/prediction/complete.json','reference/manifest.json'):
        (tmp_path/path).write_text('{}')
    with pytest.raises(ValueError):reference_handles(tmp_path,2)
