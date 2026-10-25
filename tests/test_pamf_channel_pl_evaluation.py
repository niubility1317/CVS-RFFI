from copy import deepcopy
import pytest
from experiments.pamf_channel_pl.evaluation import validate_prediction


def config():
    return dict(readouts=['raw','DURA-LU'],classes=list(range(6)),
        tests={k:dict(views_root='views/'+k) for k in ('cross_date','cross_rx','cross_both')},
        restored_caches=dict(LU='fixed-LU-cache'))


def test_prediction_requires_complete_source_frozen_readout_matrix():
    validate_prediction(config())


@pytest.mark.parametrize('bad',['truth','nested_truth','missing_population','wrong_route','G_weight','class_map'])
def test_prediction_rejects_truth_partial_matrix_and_wrong_restorer(bad):
    c=config()
    if bad=='truth':c['truth']='forbidden'
    elif bad=='nested_truth':c['tests']['cross_rx']['truth']='forbidden'
    elif bad=='missing_population':del c['tests']['cross_both']
    elif bad=='wrong_route':c['readouts']=['raw']
    elif bad=='G_weight':c['restored_caches']={'L':'not-LU'}
    else:c['classes']=[0,1,2]
    with pytest.raises(ValueError):validate_prediction(c)
