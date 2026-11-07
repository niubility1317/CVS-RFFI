import copy
import json
from unittest.mock import patch

import pytest

from experiments.pamf_phase3_restoration import covariance_family_source_validation as m


def parent_fixture(method):
    module,commit,_=m.SPECS[method]
    done=dict(status='SOURCE_TRAINED',code_commit=commit,epoch=30,updates=7410,
        source_Ls_exposures=189000,source_Us_exposures=1701000,F0_identity={'fixed':'reader'})
    for key in ('G_initialization_checkpoint_loads','TX_labels_read','V_reads','OE_reads','target_training_reads','model_state_F0_updates'):done[key]=0
    checkpoint=dict(config=module.configuration(module.ARM),code_commit=commit,epoch=30,updates=7410,state={})
    return done,checkpoint


@pytest.mark.parametrize('method',m.METHODS)
def test_parent_rejects_nonfinal_or_contaminated_checkpoint(method):
    done,checkpoint=parent_fixture(method)
    with patch.object(m,'valid_identity') as identity:
        m.validate_parent(method,done,checkpoint)
        identity.assert_called_once_with(done['F0_identity'])
    for field,value in [('target_training_reads',1),('V_reads',1),('TX_labels_read',1),('source_Us_exposures',0),('epoch',20)]:
        bad=copy.deepcopy(done);bad[field]=value
        with pytest.raises(ValueError):m.validate_parent(method,bad,checkpoint)
    bad=copy.deepcopy(checkpoint);bad['config']['run_id']='another-run'
    with pytest.raises(ValueError):m.validate_parent(method,done,bad)


def test_incomplete_or_changed_prediction_matrix_rejected_before_array_reads(tmp_path):
    complete=dict(status='SOURCE_FAMILY_V_PREDICTIONS_COMPLETE',code_commit='commit',methods=list(m.METHODS),views=list(m.SCENES),
        source_V_records=480,cells=21,TX_labels_read=0,target_inputs=0,optimizer_updates=0,model_state_updates=0)
    for key,value in [('cells',20),('methods',['signed']),('source_V_records',479),('target_inputs',1),('TX_labels_read',1)]:
        bad=dict(complete);bad[key]=value
        (tmp_path/'predictions_complete.json').write_text(json.dumps(bad),encoding='utf-8')
        with patch.object(m.np,'load') as reader,pytest.raises(ValueError):
            m.fixed_predictions(tmp_path,'commit')
        reader.assert_not_called()


def test_missing_cell_cannot_reach_source_truth(tmp_path):
    complete=dict(status='SOURCE_FAMILY_V_PREDICTIONS_COMPLETE',code_commit='commit',methods=list(m.METHODS),views=list(m.SCENES),
        source_V_records=480,cells=21,TX_labels_read=0,target_inputs=0,optimizer_updates=0,model_state_updates=0,F0_identity={})
    (tmp_path/'predictions_complete.json').write_text(json.dumps(complete),encoding='utf-8')
    # No reference/prediction exists: fixed_predictions must fail before score's truth boundary.
    with patch.object(m,'valid_identity'),pytest.raises(FileNotFoundError):m.fixed_predictions(tmp_path,'commit')
