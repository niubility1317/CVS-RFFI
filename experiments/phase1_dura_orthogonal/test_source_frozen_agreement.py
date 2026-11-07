import numpy as np
import pytest

from experiments.phase1_dura_orthogonal.source_frozen_agreement import summarize


def test_source_agreement_uses_clean_proxy_and_reports_paired_counts():
    proxy=np.array([0,1,0,1,1,0,1,0])
    rx=np.array([4,6,4,6,8,8,8,8])
    scene=np.array(['clean','clean','bad','bad','clean','clean','bad','bad'])
    pred={'raw':np.array([0,0,0,0,1,0,0,0]),
          'dura':np.array([0,1,1,1,1,0,1,1]),
          'candidate':np.array([1,1,0,1,1,0,0,0])}
    out=summarize(proxy,pred,rx,scene,views=('clean','bad'),per_group=2)
    assert out['RX4_6/bad']['matches']=={'raw':1,'dura':1,'candidate':2}
    assert out['RX4_6/bad']['delta_vs_dura']['candidate']==1
    assert out['RX8/bad']['matches']['candidate']==1


def test_source_agreement_rejects_invalid_classes_or_groups():
    x=np.array([0,1]);rx=np.array([4,8]);scene=np.array(['clean','clean'])
    with pytest.raises(ValueError,match='class'):
        summarize(x,{'raw':np.array([0,6]),'dura':x},rx,scene,views=('clean',),per_group=1)
    with pytest.raises(ValueError,match='group'):
        summarize(x,{'raw':x,'dura':x},rx,scene,views=('clean','bad'),per_group=1)
