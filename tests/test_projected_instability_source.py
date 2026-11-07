import numpy as np
import pytest
from experiments.pamf_phase3_restoration.projected_instability_source import select_groups,summary


def test_fixed_stress_groups_overlap_but_unique_panel_is_sorted():
    r=np.arange(10,dtype=float);c=r[::-1];v=r+1
    groups,selected=select_groups(r,c,v,n=2)
    assert groups['high_received_peak'].tolist()==[8,9]
    assert groups['high_coefficient_peak'].tolist()==[1,0]
    assert groups['low_noise'].tolist()==[0,1] and groups['high_noise'].tolist()==[8,9]
    assert selected.tolist()==[0,1,8,9]
    with pytest.raises(ValueError):select_groups(r,c,np.zeros(10),n=2)


def test_stress_summary_preserves_extreme_and_rejects_nonfinite():
    value=summary([0.,1.,2.,100.])
    assert value['count']==4 and value['median']==1.5 and value['maximum']==100.
    with pytest.raises(ValueError):summary([1.,float('nan')])
