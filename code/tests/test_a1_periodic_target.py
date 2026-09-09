import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from cvsrffi.a1_periodic_target import due, evaluate_checkpoint, run_epoch


def test_fixed_epoch_schedule():
    assert [e for e in range(1,201) if due(e,80,200)] == list(range(80,201,10))
    assert not due(80,0,200)


@pytest.mark.parametrize('fail', [False, True])
def test_rng_restored_and_truth_only_after_prediction(tmp_path, fail):
    py=random.getstate(); np_state=np.random.get_state(); cpu=torch.get_rng_state().clone()
    output=tmp_path/'eval'
    events=[]
    def predictor(argv):
        assert '--truth' not in argv
        assert str(tmp_path/'truth.json') not in argv
        random.random(); np.random.rand(); torch.rand(3)
        events.append('predict')
        if fail: raise RuntimeError('prediction failed')
        output.mkdir(); (output/'predictions.json').write_text('{}')
    def scorer(argv, env):
        assert (output/'predictions.json').is_file()
        assert env['CUDA_VISIBLE_DEVICES']==''
        events.append('score')
        (output/'score.json').write_text(json.dumps({'record_count':16}))
    kwargs=dict(output=output,input_package=tmp_path/'inputs',truth=tmp_path/'truth.json',
        run_id='r',row_id='row',device='cpu',predictor=predictor,scorer=scorer,expected_records=16)
    if fail:
        with pytest.raises(RuntimeError): evaluate_checkpoint(tmp_path/'own.pth',**kwargs)
        assert events==['predict']
    else:
        evaluate_checkpoint(tmp_path/'own.pth',**kwargs)
        assert events==['predict','score']
        assert json.loads((output/'evaluation_scope.json').read_text())['feeds_training'] is False
        with pytest.raises(FileExistsError): evaluate_checkpoint(tmp_path/'own.pth',**kwargs)
    assert random.getstate()==py
    np.testing.assert_array_equal(np.random.get_state()[1],np_state[1])
    assert np.random.get_state()[2:]==np_state[2:]
    assert torch.equal(torch.get_rng_state(),cpu)
