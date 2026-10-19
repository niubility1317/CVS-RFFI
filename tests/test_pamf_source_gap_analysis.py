import numpy as np
import pytest
from experiments.pamf_multireader_refine.source_gap_analysis import errors,summarize


def test_segment_error_does_not_hide_a_tail_only_change():
    clean=np.ones((2,2,256));estimate=clean.copy();estimate[:,:,232:]+=2
    values=errors(estimate,clean)
    assert np.allclose(values['prefix232'],0) and np.allclose(values['tail24'],4)
    assert np.allclose(values['full256'],4*24/256)


def test_original_clean_errors_do_not_enter_restoration_wrong_condition():
    values={k:np.array([1.,2.,3.]) for k in ('full256','prefix232','tail24')}
    result=summarize(values,np.array([0,1,2]),np.array([0,0,0]),np.array([0,1,0]))
    assert result['all']['count']==3 and result['clean_correct']['count']==2
    assert result['restored_correct_given_clean_correct']['count']==1
    assert result['restored_wrong_given_clean_correct']['regions']['tail24']['mean']==2
    with pytest.raises(ValueError):errors(np.zeros((2,2,255)),np.zeros((2,2,256)))
